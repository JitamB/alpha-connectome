"""Stateful GRU trainer (truncated BPTT over full 20k sequences), M2 (cpu/mps) or Colab (cuda).

  python train/train_gru.py --out runs/gru_base --max-train-seqs 500 --epochs 1      # M2 smoke run
  python train/train_gru.py --out runs/gru_base --epochs 4 --device cuda              # Colab full run

The model is split into one single-layer GRU per block so the ONNX export has one hidden-state
input per layer (hidden_0, hidden_1, ...), the same interface as the starterpack baseline.
Model selection uses the train holdout (V3 slice), keeping valid.parquet untouched until final evaluation.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('src', 'models')]
from data import TRAIN, VALID, WARMUP, holdout_groups, iter_sequences, train_groups  # noqa: E402
from evaluate import seq_moments, wp_from_moments  # noqa: E402
from features import N_FEATURES, batch_features  # noqa: E402
from gpu_features import torch_features  # noqa: E402
from mask_model import mask_prob  # noqa: E402

TRANSFORMS = {'raw': (112, None), 'eng': (N_FEATURES, batch_features)}

CLIP = 2.0
# Targets are integers times one of four per-sequence steps (found 2026-10-15); the clipped values therefore lie on
# a small set of atoms, which the optional distributional head classifies over.
STEPS = (0.12543297, 0.24132967, 0.4685359, 0.65686762)


def make_atoms():
    vals = {0.0, -CLIP, CLIP}
    for st in STEPS:
        for k in range(1, int(CLIP / st) + 1):
            vals.update({k * st, -k * st})
    return sorted(vals)


def decode_dist(logits, atoms):
    """(..., A) logits -> metric-optimal point prediction E[|y| y] / E[|y|] under the predicted distribution."""
    w = torch.softmax(logits, -1) * atoms.abs()
    return (w * atoms).sum(-1) / w.sum(-1).clamp_min(1e-8)


class Block(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)


class GRUModel(nn.Module):
    """Input standardisation (buffers) -> optional linear projection -> stacked GRUs -> head,
    plus an optional linear skip path from the inputs. The projection keeps the first GRU's input
    weights small (inference cost is bound by weight bytes per row)."""

    def __init__(self, n_in=112, hidden=128, layers=2, skip=False, proj=0, atoms=0):
        super().__init__()
        self.register_buffer('mean', torch.zeros(n_in))
        self.register_buffer('std', torch.ones(n_in))
        self.proj = nn.Linear(n_in, proj) if proj else None
        first = proj or n_in
        self.blocks = nn.ModuleList([Block(first if i == 0 else hidden, hidden) for i in range(layers)])
        self.reg_head = nn.Linear(hidden, 2)
        self.skip_head = nn.Linear(n_in, 2) if skip else None
        self.dist_head = nn.Linear(hidden, 2 * atoms) if atoms else None

    def forward(self, x, hidden, return_top=False):
        """x: (B, T, F); hidden: list of (1, B, H) or None -> prediction (B, T, 2), new hidden list
        (+ top GRU output (B, T, H) if return_top, for training-only auxiliary heads)."""
        x = (x - self.mean) / self.std
        z, new = (x if self.proj is None else self.proj(x)), []
        for i, block in enumerate(self.blocks):
            z, h = block.gru(z, None if hidden is None else hidden[i])
            new.append(h)
        pred = self.reg_head(z)
        if self.skip_head is not None:
            pred = pred + self.skip_head(x)
        return (pred, new, z) if return_top else (pred, new)


def weighted_mse(pred, y, row_w=None):
    """|y|-weighted MSE on clipped targets: the per-row loss that the weighted Pearson metric rewards.
    row_w: optional extra (B, T) row weights (e.g. scoring-mask proxy)."""
    y = y.clamp(-CLIP, CLIP)
    w = y.abs() if row_w is None else y.abs() * row_w[..., None]
    return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1e-8)


def pearson_loss(pred, y, anchor=0.05):
    """Negative |y|-weighted Pearson correlation per target over the chunk (the metric's form),
    plus a small weighted-MSE anchor so the prediction scale stays inside the metric's +/-2 clip."""
    yc = y.clamp(-CLIP, CLIP)
    w = yc.abs().reshape(-1, 2)
    p, t = pred.reshape(-1, 2), yc.reshape(-1, 2)
    W = w.sum(0).clamp_min(1e-8)
    pm, tm = (w * p).sum(0) / W, (w * t).sum(0) / W
    cov = (w * (p - pm) * (t - tm)).sum(0) / W
    vp, vt = (w * (p - pm) ** 2).sum(0) / W, (w * (t - tm) ** 2).sum(0) / W
    rho = cov / torch.sqrt(vp * vt + 1e-8)
    return -rho.mean() + anchor * weighted_mse(pred, y)


LOSSES = {'wmse': weighted_mse, 'pearson': pearson_loss}
AUX_COLS = list(range(52))  # i0 book + trade columns


def aux_targets(X, horizons):
    """Training-only targets from raw X (B, T, 112): mean(x[t+1..t+h]) - x[t] for the i0 columns and each h.
    Returns (B, T, len(AUX_COLS) * len(horizons)) and a (T, same) 0/1 mask of positions where the window fits."""
    x = X[..., AUX_COLS].double()
    T = x.shape[1]
    cs = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(1)], 1)
    out, valid = [], []
    for h in horizons:
        fut = torch.zeros_like(x)
        fut[:, :T - h] = (cs[:, h + 1:] - cs[:, 1:T - h + 1]) / h - x[:, :T - h]
        out.append(fut)
        v = torch.zeros(T, len(AUX_COLS), device=X.device)
        v[:T - h] = 1.0
        valid.append(v)
    return torch.cat(out, -1).float(), torch.cat(valid, -1)


VALID_OFFSET = 100_000  # training ids >= this are valid.parquet row groups (id - VALID_OFFSET)
VALID_SPLITS = {'none': [], 'A': range(0, 937), 'B': range(937, 1873), 'all': range(1873)}


def read_batch(path, groups):
    tr = [g for g in groups if g < VALID_OFFSET]
    va = [g - VALID_OFFSET for g in groups if g >= VALID_OFFSET]
    return list(iter_sequences(path, tr)) + list(iter_sequences(VALID, va))


def prefetch(groups_batches, transform=None, path=TRAIN, depth=1):
    """Background thread that reads batches of sequences -> (X (B,T,F), Y (B,T,2)) numpy arrays."""
    q = queue.Queue(depth)

    def work():
        for groups in groups_batches:
            seqs = read_batch(path, groups)
            X = np.stack([s.X if transform is None else transform(s.X) for s in seqs])
            q.put((X, np.stack([s.y for s in seqs])))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


@torch.no_grad()
def monitor_score(model, groups, device, transform, gpu_features=False, batch=16):
    """WP on all need_prediction rows of a train-holdout slice (model selection; valid stays untouched)."""
    model.eval()
    moments = []
    batches = [groups[i:i + batch] for i in range(0, len(groups), batch)]
    for X, Y in prefetch(batches, None if gpu_features else transform):
        X = torch.from_numpy(X).to(device)
        pred, _ = model(torch_features(X) if gpu_features else X, None)
        pred = pred.float().cpu().numpy()
        need = np.arange(X.shape[1]) >= WARMUP
        moments += [seq_moments(y, p, need) for y, p in zip(Y, pred)]
    model.train()
    return float(wp_from_moments(np.stack(moments).sum(0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--features', choices=list(TRANSFORMS), default='eng')
    ap.add_argument('--skip', action=argparse.BooleanOptionalAction, default=True, help='linear skip path from inputs')
    ap.add_argument('--proj', type=int, default=64, help='input projection size (0 = none)')
    ap.add_argument('--batch', type=int, default=16, help='sequences per batch')
    ap.add_argument('--chunk', type=int, default=1000, help='TBPTT window (steps)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--loss', choices=list(LOSSES), default='wmse')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--max-train-seqs', type=int)
    ap.add_argument('--monitor-seqs', type=int, default=200)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu-features', action=argparse.BooleanOptionalAction, default=None,
                    help='compute engineered features on the device (default: on for cuda)')
    ap.add_argument('--ema-decay', type=float, default=0.0,
                    help='also keep an exponential moving average of the weights (saved as ema.pt; 0 = off)')
    ap.add_argument('--aux-weight', type=float, default=0.0,
                    help='weight of the auxiliary future-i0-change loss (training only; 0 = off)')
    ap.add_argument('--aux-horizons', type=int, nargs='+', default=[10, 100])
    ap.add_argument('--lag-targets', type=int, nargs='+', default=[],
                    help='training-only head predicting the realised past targets y[t-H] for each H')
    ap.add_argument('--lag-weight', type=float, default=0.0)
    ap.add_argument('--mask-model', help='models/mask_mlp.pt: weight rows by eps + p(is_scored) (wmse, eng only)')
    ap.add_argument('--mask-eps', type=float, default=0.1)
    ap.add_argument('--dist-weight', type=float, default=0.0,
                    help='weight of a cross-entropy loss for a distributional head over the target atoms (0 = off)')
    ap.add_argument('--sum-weight', type=float, default=0.0,
                    help='extra MSE on the predicted sum t0 + t1 (the cost-like part; clipped targets; 0 = off)')
    ap.add_argument('--valid-groups', choices=list(VALID_SPLITS), default='none',
                    help='also train on valid.parquet sequences (A = row groups < 937, B = the rest)')
    ap.add_argument('--classes', type=int, nargs='+',
                    help='train and monitor only on these asset classes (splits/train_classes.json), e.g. 3')
    ap.add_argument('--init', help="fine-tune: start from another run's last.pt or best.pt (model incl. normalisation)")
    args = ap.parse_args()
    if args.gpu_features is None:
        args.gpu_features = args.device.startswith('cuda') and args.features == 'eng'
    assert args.mask_model is None or (args.loss == 'wmse' and args.features == 'eng'), 'mask weights need wmse + eng'

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)
    groups = train_groups()
    if args.max_train_seqs:
        groups = sorted(rng.choice(groups, args.max_train_seqs, replace=False).tolist())
    groups += [VALID_OFFSET + g for g in VALID_SPLITS[args.valid_groups]]
    monitor = holdout_groups()
    if args.classes:
        cls = json.loads((Path(__file__).resolve().parents[1] / 'splits' / 'train_classes.json').read_text())
        assert args.valid_groups == 'none', 'class filter covers train groups only'
        groups = [g for g in groups if cls[g] in args.classes]
        monitor = [g for g in monitor if cls[g] in args.classes]
    monitor = monitor[:args.monitor_seqs]

    n_in, transform = TRANSFORMS[args.features]
    atoms = torch.tensor(make_atoms())
    model = GRUModel(n_in, args.hidden, args.layers, args.skip, args.proj, len(atoms) if args.dist_weight else 0)
    train_only = [g for g in groups if g < VALID_OFFSET]
    stats_groups = sorted(np.random.default_rng(args.seed + 1).choice(train_only, min(64, len(train_only)), replace=False).tolist())
    sample = np.concatenate([(s.X if transform is None else transform(s.X))[WARMUP::10]
                             for s in iter_sequences(TRAIN, stats_groups)]).astype(np.float64)
    model.mean.copy_(torch.from_numpy(sample.mean(0)).float())
    model.std.copy_(torch.from_numpy(sample.std(0)).float().clamp_min(1e-6))
    del sample
    model = model.to(device)
    atoms = atoms.to(device)
    mids = (atoms[1:] + atoms[:-1]) / 2
    aux_head = nn.Linear(args.hidden, len(AUX_COLS) * len(args.aux_horizons)).to(device) if args.aux_weight else None
    aux_std = None
    lag_head = nn.Linear(args.hidden, 2 * len(args.lag_targets)).to(device) if args.lag_weight and args.lag_targets else None
    mw = {k: v.to(device) for k, v in torch.load(args.mask_model).items()} if args.mask_model else None
    if args.init:  # last.pt restores the training heads too; a plain best.pt leaves them freshly initialised
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck.get('model', ck))
        if aux_head is not None and ck.get('aux_head') is not None:
            aux_head.load_state_dict(ck['aux_head']); aux_std = ck['aux_std']
        if lag_head is not None and ck.get('lag_head') is not None:
            lag_head.load_state_dict(ck['lag_head'])
    params = [p for m in (model, aux_head, lag_head) if m is not None for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    n_batches = (len(groups) + args.batch - 1) // args.batch
    steps_per_batch = (20_000 + args.chunk - 1) // args.chunk
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * n_batches * steps_per_batch,
                                                pct_start=0.05)
    start_epoch, best, history = 0, -1.0, []
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema_decay else None
    ckpt_path = args.out / 'last.pt'
    if ckpt_path.exists():  # resume (Colab disconnects)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start_epoch, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        if ema is not None and ck.get('ema') is not None:
            ema = ck['ema']
        if aux_head is not None:
            aux_head.load_state_dict(ck['aux_head']); aux_std = ck['aux_std']
        if lag_head is not None:
            lag_head.load_state_dict(ck['lag_head'])
        print(f'resumed after epoch {ck["epoch"]}')
    (args.out / 'config.json').write_text(json.dumps({**vars(args), 'out': str(args.out), 'n_train': len(groups)}, indent=2))

    for epoch in range(start_epoch, args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(groups)
        batches = [sorted(order[i:i + args.batch].tolist()) for i in range(0, len(order), args.batch)]
        t0, losses, aux_losses, lag_losses, dist_losses = time.time(), [], [], [], []
        for b, (X, Y) in enumerate(prefetch(batches, None if args.gpu_features else transform)):
            X, Y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            if aux_head is not None:
                A, V = aux_targets(X[..., :112], args.aux_horizons)  # raw block comes first in both layouts
                if aux_std is None:
                    aux_std = ((A ** 2 * V).sum((0, 1)) / (V.sum(0) * len(A))).sqrt().clamp_min(1e-6)
                A = A / aux_std
            if args.gpu_features:
                X = torch_features(X)
            if mw is not None:
                with torch.no_grad():
                    R = args.mask_eps + mask_prob(X, mw)
            if lag_head is not None:  # (B, T, 2 * n_lags) clipped y[t-H], zero (and zero weight) for t < H
                YL = torch.cat([torch.cat([torch.zeros_like(Y[:, :h]), Y[:, :-h]], 1) for h in args.lag_targets], -1).clamp(-CLIP, CLIP)
            hidden = None
            for c in range(0, X.shape[1], args.chunk):
                pred, hidden, top = model(X[:, c:c + args.chunk], hidden, return_top=True)
                hidden = [h.detach() for h in hidden]
                lo = max(0, WARMUP - c)
                if mw is None:
                    loss = LOSSES[args.loss](pred[:, lo:], Y[:, c + lo:c + args.chunk])
                else:
                    loss = weighted_mse(pred[:, lo:], Y[:, c + lo:c + args.chunk], R[:, c + lo:c + args.chunk])
                total = loss
                if aux_head is not None:
                    v = V[c:c + args.chunk]
                    aux = (((aux_head(top) - A[:, c:c + args.chunk]) ** 2) * v).sum() / (v.sum() * len(top)).clamp_min(1.0)
                    total = loss + args.aux_weight * aux
                    aux_losses.append(aux.item())
                if args.sum_weight:
                    ys = Y[:, c + lo:c + args.chunk].clamp(-CLIP, CLIP).sum(-1)
                    se = (pred[:, lo:].sum(-1) - ys) ** 2
                    if mw is not None:
                        se = se * R[:, c + lo:c + args.chunk]
                    total = total + args.sum_weight * se.mean()
                if model.dist_head is not None:
                    yc = Y[:, c + lo:c + args.chunk].clamp(-CLIP, CLIP)
                    logits = model.dist_head(top[:, lo:]).unflatten(-1, (2, len(atoms)))
                    ce = nn.functional.cross_entropy(logits.reshape(-1, len(atoms)), torch.bucketize(yc, mids).reshape(-1),
                                                     reduction='none').view(yc.shape)
                    if mw is not None:
                        ce = ce * R[:, c + lo:c + args.chunk, None]
                    dist = ce.mean()
                    total = total + args.dist_weight * dist
                    dist_losses.append(dist.item())
                if lag_head is not None:
                    yl = YL[:, c + lo:c + args.chunk]
                    wl = yl.abs()
                    lag = (wl * (lag_head(top[:, lo:]) - yl) ** 2).sum() / wl.sum().clamp_min(1e-8)
                    total = total + args.lag_weight * lag
                    lag_losses.append(lag.item())
                opt.zero_grad(set_to_none=True)
                total.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                if ema is not None:
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            ema[k].lerp_(v, 1.0 - args.ema_decay)
                losses.append(loss.item())
            if (b + 1) % 20 == 0 or b + 1 == len(batches):
                rate = (b + 1) * args.batch * 20_000 / (time.time() - t0)
                aux_note = (f' aux {np.mean(aux_losses[-200:]):.4f}' if aux_losses else '') + \
                           (f' lag {np.mean(lag_losses[-200:]):.4f}' if lag_losses else '') + \
                           (f' ce {np.mean(dist_losses[-200:]):.4f}' if dist_losses else '')
                print(f'epoch {epoch} batch {b + 1}/{len(batches)} loss {np.mean(losses[-200:]):.4f}{aux_note} '
                      f'lr {sched.get_last_lr()[0]:.2e} {rate / 1e3:.0f}k rows/s', flush=True)
        score = monitor_score(model, monitor, device, transform, args.gpu_features)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'monitor_wp': score, 'minutes': (time.time() - t0) / 60})
        print(f'== epoch {epoch}: monitor WP (holdout, all rows) {score:.5f}', flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), args.out / 'best.pt')
        if ema is not None:
            torch.save(ema, args.out / 'ema.pt')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch, 'best': best, 'history': history, 'ema': ema,
                    'aux_head': aux_head.state_dict() if aux_head is not None else None, 'aux_std': aux_std,
                    'lag_head': lag_head.state_dict() if lag_head is not None else None}, ckpt_path)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
