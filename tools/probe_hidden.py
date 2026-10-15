"""What do trained GRUs' hidden states know? Linear/MLP probes on the top-layer state (Kaggle GPU job).

  python tools/probe_hidden.py runs/gru_all_full runs/x_base_s4000 --seqs 600 --name probe_hidden

On valid sequences [0, seqs), every `every`-th row after warm-up, 2-fold by sequence (even / odd):
  decode  : held-out corr of ridge(h -> target) vs ridge(inputs -> target), for past / future targets, past
            target means, future |y| (volatility), the sequence mean (regime) and |y|
  oracle  : V1-style WP gain of a |y|-weighted ridge on [pred, candidate] over pred alone (scored rows), i.e. how
            much each candidate would be worth if known at time t -- high value + low decodability = missing info
  residual: WP gain from ridge / small MLP on h (and on [h, raw]) predicting y - pred -> is the head underfit?
  class   : 4-class accuracy from h
Writes results/NAME.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from data import VALID, WARMUP, iter_sequences  # noqa: E402
from eval_torch import load  # noqa: E402
from gpu_features import torch_features  # noqa: E402

CLIP = 2.0
LAGS = (1, 3, 10, 30, 100, 300, 1000, 3000)
FUTS = (10, 100, 300)
PMEANS = (100, 1000)


def targets(y):
    """y: (T, 2) clipped -> dict name -> (T, 2) with NaN where undefined."""
    T = len(y)
    out = {}
    for k in LAGS:
        a = np.full_like(y, np.nan); a[k:] = y[:-k]; out[f'lag{k}'] = a
    for k in FUTS:
        a = np.full_like(y, np.nan); a[:-k] = y[k:]; out[f'fut{k}'] = a
    cs = np.concatenate([np.zeros((1, 2)), np.cumsum(y, 0)])
    for w in PMEANS:
        a = np.full_like(y, np.nan); a[w:] = (cs[w:T] - cs[:T - w]) / w; out[f'pmean{w}'] = a
    ca = np.concatenate([np.zeros((1, 2)), np.cumsum(np.abs(y), 0)])
    a = np.full_like(y, np.nan); a[:T - 100] = (ca[100 + 1:T + 1] - ca[1:T - 100 + 1]) / 100; out['fvol100'] = a
    out['absy'] = np.abs(y)
    out['seqmean'] = np.broadcast_to(y.mean(0), y.shape).copy()
    return out


def wp(y, p):
    """Pooled |y|-weighted Pearson, mean over the two targets (the metric on concatenated rows)."""
    w = np.abs(y)
    r = []
    for j in range(y.shape[1]):
        W = w[:, j].sum()
        my, mp = (w[:, j] * y[:, j]).sum() / W, (w[:, j] * p[:, j]).sum() / W
        cov = (w[:, j] * (y[:, j] - my) * (p[:, j] - mp)).sum()
        r.append(cov / np.sqrt((w[:, j] * (y[:, j] - my) ** 2).sum() * (w[:, j] * (p[:, j] - mp) ** 2).sum()))
    return float(np.mean(r))


def ridge(Xa, Ya, Xb, wa=None, lam=1e-2, device='cpu', chunk=100_000):
    """Ridge with intercept (standardised inputs), fitted on (Xa, Ya[, row weights wa]); returns predictions for Xb.
    Normal equations accumulated in row chunks so large probes fit in GPU memory."""
    mu, sd = Xa.mean(0, dtype=np.float64), np.maximum(Xa.std(0, dtype=np.float64), 1e-6)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    prep = lambda x: torch.cat([(t(x) - t(mu)) / t(sd), torch.ones(len(x), 1, dtype=torch.float64, device=device)], 1)
    k = Xa.shape[1] + 1
    A = torch.zeros(k, k, dtype=torch.float64, device=device)
    B = torch.zeros(k, Ya.shape[1], dtype=torch.float64, device=device)
    for i in range(0, len(Xa), chunk):
        x = prep(Xa[i:i + chunk])
        w = torch.ones(len(x), 1, dtype=torch.float64, device=device) if wa is None else t(wa[i:i + chunk])[:, None]
        A += x.T @ (w * x)
        B += x.T @ (w * t(Ya[i:i + chunk]))
    W = torch.linalg.solve(A + lam * len(Xa) * torch.eye(k, dtype=torch.float64, device=device), B)
    return np.concatenate([(prep(Xb[i:i + chunk]) @ W).cpu().numpy() for i in range(0, len(Xb), chunk)])


def corr(a, b):
    return float(np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]))


def mlp_resid(Xa, Ra, wa, Xb, device, epochs=3, seed=0):
    """Small MLP X -> residual, |y| x scored weighted MSE; returns predictions for Xb."""
    torch.manual_seed(seed)
    Xa, Xb, Ra, wa = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (Xa, Xb, Ra, wa))
    mu, sd = Xa.mean(0), Xa.std(0).clamp_min(1e-6)
    Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(Xa.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xa), device=device)
        for i in range(0, len(Xa), 8192):
            j = perm[i:i + 8192]
            loss = (wa[j, None] * (net(Xa[j]) - Ra[j]) ** 2).sum() / wa[j].sum().clamp_min(1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(Xb).cpu().numpy()


@torch.no_grad()
def collect(run, n_seqs, every, device, batch=16):
    model, _ = load(Path(run), 'best.pt', device)
    rows = {k: [] for k in ('h', 'raw', 'F', 'pred', 'y', 'scored', 'seq')}
    tg = {}
    seqs = list(iter_sequences(VALID, range(n_seqs)))
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        X = torch.from_numpy(np.stack([s.X for s in chunk])).to(device)
        F = torch_features(X)
        pred, _, top = model(F, None, return_top=True)
        for b, s in enumerate(chunk):
            idx = np.arange(WARMUP, len(s.X), every)
            y = np.clip(s.y, -CLIP, CLIP).astype(np.float64)
            rows['h'].append(top[b, idx].float().cpu().numpy())
            rows['F'].append(F[b, idx].float().cpu().numpy())
            rows['raw'].append(s.X[idx])
            rows['pred'].append(np.clip(pred[b, idx].float().cpu().numpy(), -CLIP, CLIP))
            rows['y'].append(y[idx])
            rows['scored'].append(s.scored[idx] & s.need[idx])
            rows['seq'].append(np.full(len(idx), i + b))
            for k, v in targets(y).items():
                tg.setdefault(k, []).append(v[idx])
    return {k: np.concatenate(v) for k, v in rows.items()}, {k: np.concatenate(v) for k, v in tg.items()}


def probe(run, n_seqs, every, device):
    d, tg = collect(run, n_seqs, every, device)
    cls = np.array(json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls'])[d['seq']]
    folds = [d['seq'] % 2 == 0, d['seq'] % 2 == 1]
    res = {'run': run, 'rows': int(len(d['y'])), 'decode': {}, 'oracle': {}, 'residual': {}}

    def two_fold(fn):
        """fn(train_mask, test_mask) -> predictions on test rows; returns predictions for all rows."""
        out = None
        for a, b in (folds, folds[::-1]):
            p = fn(a, b)
            if out is None:
                out = np.full((len(d['y']),) + p.shape[1:], np.nan)
            out[b] = p
        return out

    for name, t in tg.items():
        ok = np.isfinite(t).all(1)
        r = {}
        for src, X in (('h', d['h']), ('inputs', d['F']), ('h+inputs', np.concatenate([d['h'], d['F']], 1))):
            p = two_fold(lambda a, b: ridge(X[a & ok], t[a & ok], X[b], device=device))
            r[src] = corr(t[ok], p[ok])
        res['decode'][name] = r
        print(f"decode {name:9s} h {r['h']:+.3f}  inputs {r['inputs']:+.3f}  h+inputs {r['h+inputs']:+.3f}", flush=True)

    sc = d['scored']
    base = wp(d['y'][sc], d['pred'][sc])
    res['oracle']['base_wp'] = base
    w = np.abs(d['y']).mean(1) * sc   # one row weight for both targets (|y| averaged), scored rows only
    for name in [f'lag{k}' for k in LAGS] + [f'pmean{w_}' for w_ in PMEANS] + ['fvol100', 'seqmean']:
        t = tg[name]
        ok = np.isfinite(t).all(1)
        Z = np.concatenate([d['pred'], np.nan_to_num(t), ok[:, None].astype(float)], 1)
        p = two_fold(lambda a, b: ridge(Z[a], d['y'][a], Z[b], wa=w[a], lam=1e-6, device=device))
        res['oracle'][name] = wp(d['y'][sc], p[sc]) - base
        print(f'oracle {name:9s} WP gain {res["oracle"][name]:+.4f} (base {base:.4f})', flush=True)

    R = d['y'] - d['pred']
    for src, X in (('h', d['h']), ('h+raw', np.concatenate([d['h'], d['raw']], 1))):
        p = two_fold(lambda a, b: ridge(X[a], R[a], X[b], wa=w[a], device=device))
        res['residual'][f'ridge_{src}'] = wp(d['y'][sc], d['pred'][sc] + p[sc]) - base
        p = two_fold(lambda a, b: mlp_resid(X[a], R[a], w[a], X[b], device))
        res['residual'][f'mlp_{src}'] = wp(d['y'][sc], d['pred'][sc] + p[sc]) - base
    print('residual', {k: round(v, 4) for k, v in res['residual'].items()}, flush=True)

    onehot = np.eye(4)[cls]
    p = two_fold(lambda a, b: ridge(d['h'][a], onehot[a], d['h'][b], device=device))
    res['class_acc_h'] = float((p.argmax(1) == cls).mean())
    print(f"class accuracy from h {res['class_acc_h']:.4f}", flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--seqs', type=int, default=600)
    ap.add_argument('--every', type=int, default=20)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = []
    for run in args.runs:
        print(f'== {run}', flush=True)
        out.append(probe(run, args.seqs, args.every, device))
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(out, indent=2))
