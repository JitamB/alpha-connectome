"""Frozen random filter bank + weighted ridge (the "reservoir" baseline from Discord), and its blend with the GRUs.

  python tools/reservoir.py runs/gru_all_full runs/gru_all_full_seed1 --name reservoir [--channels 1024 512]

Model: u_t = x_t @ P (P = randn(112, C) / sqrt(112), fixed seed), causal EMA per channel with half-lives
geomspace(1, 3000, C): s_t = (1 - a) s_{t-1} + a u_t, prediction = [s_t, 1] @ beta, beta from |y|-weighted ridge on
train sequences (holdout excluded), ridge strength chosen on the train holdout (V3).
Scores V1 / V2 / V3 alone, then blends (1 - w) * GRU-ensemble + w * reservoir (GRU runs are train-only, so valid is
clean); w is chosen on valid half A and reported on half B. Writes results/NAME.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from data import TRAIN, VALID, WARMUP, holdout_groups, iter_sequences, train_groups  # noqa: E402
from eval_torch import load  # noqa: E402
from evaluate import seq_moments, wp_from_moments  # noqa: E402
from gpu_features import torch_features  # noqa: E402

CLIP = 2.0
SEED = 0
CHUNK = 64
LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2)
BLEND_W = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
HALF_A = 937


class Bank:
    def __init__(self, channels, device):
        self.P = torch.from_numpy(np.random.RandomState(SEED).randn(112, channels) / np.sqrt(112)).float().to(device)
        a = 1.0 - np.exp(-np.log(2.0) / np.geomspace(1.0, 3000.0, channels))
        d = 1.0 - a
        j = np.arange(CHUNK)
        expo = np.clip(j[:, None] - j[None, :], 0, None)
        M = np.where(j[:, None] >= j[None, :], a[:, None, None] * d[:, None, None] ** expo, 0.0)   # (C, L, L)
        self.M = torch.from_numpy(M).float().to(device)
        self.v = torch.from_numpy(d[:, None] ** (j[None, :] + 1)).float().to(device)              # (C, L)

    @torch.no_grad()
    def states(self, X):
        """X: (B, T, 112) -> (B, T, C) causal EMA states (zero initial state), chunked closed form."""
        U = X @ self.P
        B, T, C = U.shape
        s = torch.zeros(B, C, device=U.device)
        out = []
        for c in range(0, T, CHUNK):
            u = U[:, c:c + CHUNK]
            L = u.shape[1]
            e = torch.einsum('cjk,bkc->bjc', self.M[:, :L, :L], u) + self.v[None, :, :L].transpose(1, 2) * s[:, None, :]
            out.append(e)
            s = e[:, -1]
        return torch.cat(out, 1)


def batches(path, groups, size=16):
    chunk = []
    for s in iter_sequences(path, groups):
        chunk.append(s)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def design(S):
    return torch.cat([S, torch.ones_like(S[..., :1])], -1)


@torch.no_grad()
def fit(bank, groups, device, every=2):
    C = bank.P.shape[1] + 1
    G = torch.zeros(2, C, C, dtype=torch.float64, device=device)
    b = torch.zeros(2, C, dtype=torch.float64, device=device)
    for chunk in batches(TRAIN, groups):
        X = torch.from_numpy(np.stack([s.X for s in chunk])).to(device)
        Y = torch.from_numpy(np.stack([s.y for s in chunk])).to(device).clamp(-CLIP, CLIP)
        Z = design(bank.states(X))[:, WARMUP::every].reshape(-1, C).double()
        Y = Y[:, WARMUP::every].reshape(-1, 2).double()
        for j in range(2):
            w = Y[:, j].abs()
            G[j] += Z.T @ (Z * w[:, None])
            b[j] += Z.T @ (w * Y[:, j])
    return G, b


def solve(G, b, lam):
    C = G.shape[-1]
    scale = torch.diagonal(G, dim1=-2, dim2=-1).mean(-1)
    eye = torch.eye(C, dtype=G.dtype, device=G.device)
    return torch.stack([torch.linalg.solve(G[j] + lam * scale[j] * eye, b[j]) for j in range(2)], 1).float()   # (C, 2)


@torch.no_grad()
def collect(banks_betas, gru_models, path, groups, device):
    """-> per sequence: y, need, scored, group, reservoir preds {key: (T,2)}, gru pred (T,2)."""
    rows = []
    for chunk in batches(path, groups):
        X = torch.from_numpy(np.stack([s.X for s in chunk])).to(device)
        res = {}
        for key, (bank, betas) in banks_betas.items():
            Z = design(bank.states(X))
            for lam, beta in betas.items():
                res[(key, lam)] = (Z @ beta).cpu().numpy()
        g = None
        if gru_models:
            F = torch_features(X)
            g = torch.stack([m(F, None)[0] for m in gru_models]).mean(0).float().cpu().numpy()
        for i, s in enumerate(chunk):
            rows.append({'y': s.y, 'need': s.need, 'scored': s.scored, 'group': s.group,
                         'res': {k: v[i] for k, v in res.items()}, 'gru': None if g is None else g[i]})
    return rows


def wp(rows, pred, view, sel=lambda r: True):
    m = []
    for r in rows:
        if not sel(r):
            continue
        mask = r['need'] if view != 'V1' else r['need'] & r['scored']
        m.append(seq_moments(r['y'], pred(r), mask))
    return float(wp_from_moments(np.stack(m).sum(0)).mean()) if m else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gru_runs', nargs='*')
    ap.add_argument('--name', required=True)
    ap.add_argument('--channels', type=int, nargs='+', default=[1024, 512])
    ap.add_argument('--n-train', type=int, default=2000)
    ap.add_argument('--n-hold', type=int, default=800)
    ap.add_argument('--n-valid', type=int, default=1873)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(0)
    tr = sorted(rng.choice(train_groups(), args.n_train, replace=False).tolist())
    banks = {}
    for C in args.channels:
        bank = Bank(C, device)
        G, b = fit(bank, tr, device)
        banks[C] = (bank, {lam: solve(G, b, lam) for lam in LAMBDAS})
        print(f'fitted C={C} on {len(tr)} train seqs', flush=True)
    res = {'n_train': len(tr)}
    hold = collect(banks, [], TRAIN, holdout_groups()[:args.n_hold], device)
    best = {}
    for C in args.channels:
        scores = {lam: wp(hold, lambda r: r['res'][(C, lam)], 'V3') for lam in LAMBDAS}
        best[C] = max(scores, key=scores.get)
        res[f'C{C}_V3_by_lambda'] = scores
        print(f'C={C}: V3 by lambda {scores} -> {best[C]}', flush=True)
    gru = [load(Path(r), 'best.pt', device)[0] for r in args.gru_runs]
    valid = collect({C: (banks[C][0], {best[C]: banks[C][1][best[C]]}) for C in args.channels}, gru, VALID,
                    list(range(args.n_valid)), device)
    A, B = (lambda r: r['group'] < HALF_A), (lambda r: r['group'] >= HALF_A)
    for C in args.channels:
        key = (C, best[C])
        alone = {v: wp(valid, lambda r: r['res'][key], v) for v in ('V1', 'V2')}
        alone['V3'] = res[f'C{C}_V3_by_lambda'][best[C]]
        res[f'C{C}_alone'] = alone
        print(f'reservoir C={C} alone: ' + ' '.join(f'{k} {v:.4f}' for k, v in alone.items()), flush=True)
        if not gru:
            continue
        blend = {}
        for w in BLEND_W:
            f = lambda r, w=w: (1 - w) * r['gru'] + w * r['res'][key]
            blend[w] = {'V1': wp(valid, f, 'V1'), 'V1_A': wp(valid, f, 'V1', A), 'V1_B': wp(valid, f, 'V1', B),
                        'V2': wp(valid, f, 'V2')}
            print(f'  blend w={w}: ' + ' '.join(f'{k} {v:.5f}' for k, v in blend[w].items()), flush=True)
        w_star = max(BLEND_W, key=lambda w: blend[w]['V1_A'])
        res[f'C{C}_blend'] = {str(w): v for w, v in blend.items()}
        res[f'C{C}_w_star_from_A'] = w_star
        res[f'C{C}_half_B_gain'] = blend[w_star]['V1_B'] - blend[0.0]['V1_B']
        print(f'  w* (chosen on half A) = {w_star}: half-B V1 gain {res[f"C{C}_half_B_gain"]:+.5f}', flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
