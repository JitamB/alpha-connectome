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
from shape_lookup import N_FEAT as SHAPE_N, features as shape_features, load_table  # noqa: E402

TRANSFORMS = {'raw': (112, None), 'eng': (N_FEATURES, batch_features)}
BLOCKS = ('raw', 'diff', 'ema10', 'ema100', 'ema1000', 'vol')  # 112-column blocks of the engineered features, in order

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


def make_class_atoms():
    """Exact clipped target support for each asset tick class."""
    out = []
    for step in STEPS:
        n = int(np.ceil(CLIP / step)) + 1
        vals = {-CLIP, 0.0, CLIP}
        for k in range(-n, n + 1):
            vals.add(float(np.clip(k * step, -CLIP, CLIP)))
        out.append(sorted(vals))
    return out


def decode_dist(logits, atoms):
    """(..., A) logits -> metric-optimal point prediction E[|y| y] / E[|y|] under the predicted distribution."""
    w = torch.softmax(logits, -1) * atoms.abs()
    return (w * atoms).sum(-1) / w.sum(-1).clamp_min(1e-8)


class Block(nn.Module):
    """One recurrent layer: GRU (attribute name kept for existing checkpoints) or LSTM (state = (h, c) tuple)."""

    def __init__(self, n_in, hidden, cell='gru'):
        super().__init__()
        if cell == 'lstm':
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        else:
            self.gru = nn.GRU(n_in, hidden, batch_first=True)

    def forward(self, z, h):
        return self.lstm(z, h) if hasattr(self, 'lstm') else self.gru(z, h)


class GRUModel(nn.Module):
    """Input standardisation (buffers) -> optional linear projection -> stacked GRUs -> head,
    plus an optional linear skip path from the inputs. The projection keeps the first GRU's input
    weights small (inference cost is bound by weight bytes per row).
    Optional causal CNN front-end (conv=(channels, kernel, dilations)): dilated Conv1d + ReLU layers over the
    standardised raw block (first 112 inputs), concatenated to the projection. Its state is the last
    R = (kernel - 1) * sum(dilations) raw rows, carried as the last element of the hidden list (zeros at
    sequence start), so chunked training equals row-by-row streaming."""

    def __init__(self, n_in=112, hidden=128, layers=2, skip=False, proj=0, atoms=0, conv=None, cell='gru', drop=(),
                 dropout=0.0, pair_atoms=None):
        super().__init__()
        self.p_drop = dropout  # training-only dropout on the input of every recurrent layer (no parameters)
        self.register_buffer('mean', torch.zeros(n_in))
        self.register_buffer('std', torch.ones(n_in))
        # optional: engineered blocks zeroed after standardisation (export zeroes their weights)
        self.register_buffer('drop', None, persistent=False)
        if drop:
            m = torch.ones(n_in)
            for b in drop:
                m[BLOCKS.index(b) * 112:(BLOCKS.index(b) + 1) * 112] = 0.0
            self.register_buffer('drop', m, persistent=False)
        self.proj = nn.Linear(n_in, proj) if proj else None
        first = proj or n_in
        self.conv, self.conv_ctx = None, 0
        if conv:
            ch, k, dils = conv
            self.conv = nn.ModuleList([nn.Conv1d(112 if i == 0 else ch, ch, k, dilation=d) for i, d in enumerate(dils)])
            self.conv_ctx = (k - 1) * sum(dils)
            first += ch
        self.blocks = nn.ModuleList([Block(first if i == 0 else hidden, hidden, cell) for i in range(layers)])
        self.reg_head = nn.Linear(hidden, 2)
        self.skip_head = nn.Linear(n_in, 2) if skip else None
        self.dist_head = nn.Linear(hidden, 2 * atoms) if atoms else None
        self.class_head = nn.Linear(hidden, len(STEPS)) if pair_atoms is not None else None
        self.pair_heads = nn.ModuleList([nn.Linear(hidden, len(a) ** 2) for a in pair_atoms]) if pair_atoms is not None else None

    def forward(self, x, hidden, return_top=False):
        """x: (B, T, F); hidden: list of (1, B, H) or None -> prediction (B, T, 2), new hidden list
        (+ top GRU output (B, T, H) if return_top, for training-only auxiliary heads)."""
        x = (x - self.mean) / self.std
        if self.drop is not None:
            x = x * self.drop
        z, new = (x if self.proj is None else self.proj(x)), []
        if self.conv is not None:
            raw = x[..., :112]
            buf = torch.zeros_like(raw[:, :self.conv_ctx]) if hidden is None else hidden[-1]
            c = torch.cat([buf, raw], 1)
            ctx = c[:, -self.conv_ctx:]
            c = c.transpose(1, 2)
            for conv in self.conv:
                c = torch.relu(conv(c))
            z = torch.cat([z, c.transpose(1, 2)], -1)
        for i, block in enumerate(self.blocks):
            if self.p_drop and self.training:
                z = nn.functional.dropout(z, self.p_drop, True)
            z, h = block(z, None if hidden is None else hidden[i])
            new.append(h)
        if self.conv is not None:
            new.append(ctx)
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


ACT_PER_H = 22 + 22 + 1 + 2  # up / down excursion per i0 price column, log1p(i0 price changes), mean |y| per target


@torch.no_grad()
def act_targets(X, Y, horizons):
    """Training-only activity/range targets over the future window (t, t+h] (raw X (B, T, 112), targets Y (B, T, 2)):
    per i0 price column max(x[t+1..t+h]) - x[t] and x[t] - min(x[t+1..t+h]), log1p(number of rows in the window where
    an i0 price column changes), and the window mean of |clipped y| per target.
    Returns (B, T, ACT_PER_H * len(horizons)) and a (T, same) 0/1 mask of positions where the window fits."""
    x = X[..., :22].float()
    B, T, C = x.shape
    xs = x.transpose(1, 2).reshape(B * C, 1, T)
    tr = torch.zeros(B, T, device=X.device, dtype=torch.float64)
    tr[:, 1:] = (x[:, 1:] != x[:, :-1]).any(-1).double()
    cs_tr = torch.cat([torch.zeros_like(tr[:, :1]), tr.cumsum(1)], 1)          # cs[k] = sum over s < k
    ay = Y.clamp(-CLIP, CLIP).abs().double()
    cs_y = torch.cat([torch.zeros_like(ay[:, :1]), ay.cumsum(1)], 1)
    idx = torch.arange(T, device=X.device)
    out, valid = [], []
    for h in horizons:
        ok = (idx <= T - 1 - h)[None, :, None]
        hi = torch.cat([xs[..., 1:], xs.new_full((B * C, 1, h), float('-inf'))], -1)   # x[t+1..], padded at the end
        lo = torch.cat([xs[..., 1:], xs.new_full((B * C, 1, h), float('inf'))], -1)
        mx = nn.functional.max_pool1d(hi, h, 1).reshape(B, C, T).transpose(1, 2)
        mn = -nn.functional.max_pool1d(-lo, h, 1).reshape(B, C, T).transpose(1, 2)
        a, b = (idx + 1).clamp(max=T), (idx + h + 1).clamp(max=T)
        n_tr = (cs_tr[:, b] - cs_tr[:, a]).float()[..., None]
        m_y = ((cs_y[:, b] - cs_y[:, a]) / h).float()
        blk = torch.cat([mx - x, x - mn, torch.log1p(n_tr), m_y], -1)
        out.append(torch.where(ok, blk, torch.zeros_like(blk)))
        valid.append(ok[0].float().expand(T, ACT_PER_H))
    return torch.cat(out, -1), torch.cat(valid, -1)


@torch.no_grad()
def future_features(X, horizons):
    """NON-CAUSAL teacher inputs from raw X (B, T, 112): mean(x[t+1..t+h]) - x[t] for each h (window cut at the
    sequence end, 0 on the last row) -> (B, T, 112 * len(horizons)). Only for privileged teachers, never exported."""
    x = X.double()
    T = x.shape[1]
    cs = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(1)], 1)
    idx = torch.arange(T, device=X.device)
    out = []
    for h in horizons:
        end = (idx + h).clamp(max=T - 1)
        n = (end - idx).clamp_min(1).double()[None, :, None]
        m = (cs[:, end + 1] - cs[:, idx + 1]) / n - x
        m[:, -1] = 0.0
        out.append(m.float())
    return torch.cat(out, -1)


def load_teacher(run, device):
    """Frozen privileged teacher (a run trained with --future-inputs)."""
    cfg = json.loads((Path(run) / 'config.json').read_text())
    n_in = TRANSFORMS[cfg['features']][0] + 112 * len(cfg['future_inputs'])
    m = GRUModel(n_in, cfg['hidden'], cfg['layers'], cfg['skip'], cfg['proj'], len(make_atoms()) if cfg.get('dist_weight') else 0)
    m.load_state_dict(torch.load(Path(run) / 'best.pt', map_location=device))
    return m.to(device).eval(), cfg['future_inputs']


VALID_OFFSET = 100_000  # training ids >= this are valid.parquet row groups (id - VALID_OFFSET)
VALID_SPLITS = {'none': [], 'A': range(0, 937), 'B': range(937, 1873), 'all': range(1873)}


def read_batch(path, groups):
    tr = [g for g in groups if g < VALID_OFFSET]
    va = [g - VALID_OFFSET for g in groups if g >= VALID_OFFSET]
    return list(iter_sequences(path, tr)) + list(iter_sequences(VALID, va))


def prefetch(groups_batches, transform=None, path=TRAIN, depth=1, table=None):
    """Background thread that reads batches of sequences -> (X (B,T,F), Y (B,T,2), M (B,T)) numpy arrays;
    M = official is_scored mask (0/1) for valid sequences, -1 for train sequences (no mask).
    table (--shape-table): the exact-shape lookup features are appended as the last SHAPE_N columns of X."""
    q = queue.Queue(depth)

    def work():
        for groups in groups_batches:
            seqs = read_batch(path, groups)
            X = np.stack([s.X if transform is None else transform(s.X) for s in seqs])
            if table is not None:
                X = np.concatenate([X, np.stack([shape_features(s.X, *table)[0] for s in seqs])], -1)
            M = np.stack([s.scored.astype(np.float32) if s.scored is not None else np.full(len(s.y), -1, np.float32)
                          for s in seqs])
            q.put((X, np.stack([s.y for s in seqs]), M))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def engineer(X):
    """GPU engineered features of raw rows (B, T, 112), keeping columns appended by prefetch (shape lookup) at the end."""
    return torch_features(X) if X.shape[-1] == 112 else torch.cat([torch_features(X[..., :112]), X[..., 112:]], -1)


@torch.no_grad()
def monitor_score(model, groups, device, transform, gpu_features=False, batch=16, future=(), table=None):
    """WP on all need_prediction rows of a train-holdout slice (model selection; valid stays untouched)."""
    model.eval()
    moments = []
    batches = [groups[i:i + batch] for i in range(0, len(groups), batch)]
    for X, Y, _ in prefetch(batches, None if gpu_features else transform, table=table):
        X = torch.from_numpy(X).to(device)
        F = engineer(X) if gpu_features else X
        if future:
            F = torch.cat([F, future_features(X, future)], -1)
        pred, _ = model(F, None)
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
                    help='training-only head predicting the realised past targets y[t-H] for each H (H < 0: future targets)')
    ap.add_argument('--lag-weight', type=float, default=0.0)
    ap.add_argument('--pmean-targets', type=int, nargs='+', default=[],
                    help='extra lag-head outputs: mean of the clipped targets over [t-W, t) for each W')
    ap.add_argument('--mask-model', help='models/mask_mlp.pt: weight rows by eps + p(is_scored) (wmse, eng only)')
    ap.add_argument('--mask-eps', type=float, default=0.1)
    ap.add_argument('--valid-true-mask', action='store_true',
                    help='with --mask-model and --valid-groups: weight valid rows by eps + the official is_scored mask')
    ap.add_argument('--dist-weight', type=float, default=0.0,
                    help='weight of a cross-entropy loss for a distributional head over the target atoms (0 = off)')
    ap.add_argument('--pair-weight', type=float, default=0.0,
                    help='weight of class-conditional joint CE over clipped (t0,t1) tick pairs (0 = off)')
    ap.add_argument('--class-weight', type=float, default=0.1,
                    help='per-row asset-class CE weight when --pair-weight is enabled')
    ap.add_argument('--sum-weight', type=float, default=0.0,
                    help='extra MSE on the predicted sum t0 + t1 (the cost-like part; clipped targets; 0 = off)')
    ap.add_argument('--valid-groups', choices=list(VALID_SPLITS), default='none',
                    help='also train on valid.parquet sequences (A = row groups < 937, B = the rest)')
    ap.add_argument('--classes', type=int, nargs='+',
                    help='train and monitor only on these asset classes (splits/train_classes.json), e.g. 3')
    ap.add_argument('--conv', type=int, nargs='+', metavar=('CHANNELS', 'KERNEL'),
                    help='causal CNN front-end on the raw block: CHANNELS KERNEL DILATION... (e.g. 32 4 1 4 16)')
    ap.add_argument('--cell', choices=['gru', 'lstm'], default='gru', help='recurrent cell of every layer')
    ap.add_argument('--init', help="fine-tune: start from another run's last.pt or best.pt (model incl. normalisation)")
    ap.add_argument('--future-inputs', type=int, nargs='+', default=[],
                    help='PRIVILEGED TEACHER: also feed mean(x[t+1..t+h]) - x[t] for each h (non-causal, never exported)')
    ap.add_argument('--train-half', type=int, choices=[0, 1],
                    help='train only on train row groups with group %% 2 == this (cross-fitted teachers)')
    ap.add_argument('--teacher', nargs=2, metavar=('RUN_HALF0', 'RUN_HALF1'),
                    help='distil from two cross-fitted teachers: a sequence with group %% 2 == k gets the teacher trained on '
                         'the other half (valid sequences: half-0 teacher)')
    ap.add_argument('--teacher-weight', type=float, default=1.0,
                    help='weight of the |y|-weighted MSE to the clipped teacher prediction (added to the target loss)')
    ap.add_argument('--drop-blocks', nargs='+', choices=BLOCKS, default=[],
                    help='zero these engineered feature blocks (e.g. raw vol: sequence-level levels that shift between periods)')
    ap.add_argument('--valid-repeat', type=int, default=1, help='repeat the --valid-groups sequences this often per epoch')
    ap.add_argument('--no-train', action='store_true', help='no train.parquet sequences (fine-tune on --valid-groups only)')
    ap.add_argument('--dropout', type=float, default=0.0, help='training-only dropout on the input of every recurrent layer')
    ap.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay')
    ap.add_argument('--act-weight', type=float, default=0.0,
                    help='weight of the training-only activity/range head (future i0 excursions, price changes, mean |y|)')
    ap.add_argument('--act-horizons', type=int, nargs='+', default=[100, 300])
    ap.add_argument('--shape-table', help='exact-shape lookup table (tools/build_shape_table.py): append its features as inputs')
    args = ap.parse_args()
    if args.gpu_features is None:
        args.gpu_features = args.device.startswith('cuda') and args.features == 'eng'
    assert args.mask_model is None or (args.loss == 'wmse' and args.features == 'eng'), 'mask weights need wmse + eng'
    assert not (args.future_inputs or args.teacher) or args.gpu_features, 'teacher paths need --gpu-features'
    assert not (args.shape_table and (args.future_inputs or args.teacher or args.conv)), 'shape table: plain recipe only'
    table = load_table(args.shape_table) if args.shape_table else None

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
    if args.train_half is not None:
        groups = [g for g in groups if g % 2 == args.train_half]
    if args.no_train:
        groups = []
    groups += [VALID_OFFSET + g for g in VALID_SPLITS[args.valid_groups]] * args.valid_repeat
    monitor = holdout_groups()
    train_classes = json.loads((Path(__file__).resolve().parents[1] / 'splits' / 'train_classes.json').read_text())
    valid_classes = json.loads((Path(__file__).resolve().parents[1] / 'splits' / 'valid_classes.json').read_text())['cls']
    def class_of(g):
        return train_classes[g] if g < VALID_OFFSET else valid_classes[g - VALID_OFFSET]
    if args.classes:
        cls = json.loads((Path(__file__).resolve().parents[1] / 'splits' / 'train_classes.json').read_text())
        assert args.valid_groups == 'none', 'class filter covers train groups only'
        groups = [g for g in groups if cls[g] in args.classes]
        monitor = [g for g in monitor if cls[g] in args.classes]
    monitor = monitor[:args.monitor_seqs]

    n_in, transform = TRANSFORMS[args.features]
    atoms = torch.tensor(make_atoms())
    pair_atoms = make_class_atoms() if args.pair_weight else None
    conv = (args.conv[0], args.conv[1], args.conv[2:]) if args.conv else None
    model = GRUModel(n_in + SHAPE_N * (table is not None) + 112 * len(args.future_inputs), args.hidden, args.layers, args.skip, args.proj,
                     len(atoms) if args.dist_weight else 0, conv, args.cell, args.drop_blocks, args.dropout, pair_atoms)
    train_only = [g for g in groups if g < VALID_OFFSET] or train_groups()
    stats_groups = sorted(np.random.default_rng(args.seed + 1).choice(train_only, min(64, len(train_only)), replace=False).tolist())

    def stats_rows(x):
        f = x if transform is None else transform(x)
        if table is not None:
            f = np.hstack([f, shape_features(x, *table)[0]])
        if args.future_inputs:
            f = np.hstack([f, future_features(torch.from_numpy(x)[None], args.future_inputs)[0].numpy()])
        return f[WARMUP::10]
    sample = np.concatenate([stats_rows(s.X) for s in iter_sequences(TRAIN, stats_groups)]).astype(np.float64)
    model.mean.copy_(torch.from_numpy(sample.mean(0)).float())
    model.std.copy_(torch.from_numpy(sample.std(0)).float().clamp_min(1e-6))
    del sample
    model = model.to(device)
    atoms = atoms.to(device)
    pair_atom_tensors = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in pair_atoms] if pair_atoms else None
    mids = (atoms[1:] + atoms[:-1]) / 2
    aux_head = nn.Linear(args.hidden, len(AUX_COLS) * len(args.aux_horizons)).to(device) if args.aux_weight else None
    aux_std = None
    n_lag = len(args.lag_targets) + len(args.pmean_targets)
    lag_head = nn.Linear(args.hidden, 2 * n_lag).to(device) if args.lag_weight and n_lag else None
    act_head = nn.Linear(args.hidden, ACT_PER_H * len(args.act_horizons)).to(device) if args.act_weight else None
    act_std = None
    mw = {k: v.to(device) for k, v in torch.load(args.mask_model).items()} if args.mask_model else None
    teachers = [load_teacher(r, device) for r in args.teacher] if args.teacher else None
    for t, _ in teachers or []:
        t.requires_grad_(False)
    if args.init:  # last.pt restores the training heads too; a plain best.pt leaves them freshly initialised
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck.get('model', ck))
        if aux_head is not None and ck.get('aux_head') is not None:
            aux_head.load_state_dict(ck['aux_head']); aux_std = ck['aux_std']
        if lag_head is not None and ck.get('lag_head') is not None:
            lag_head.load_state_dict(ck['lag_head'])
    params = [p for m in (model, aux_head, lag_head, act_head) if m is not None for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
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
        if act_head is not None:
            act_head.load_state_dict(ck['act_head']); act_std = ck['act_std']
        print(f'resumed after epoch {ck["epoch"]}')
    prov_file = Path(__file__).resolve().parents[1] / 'provenance.json'  # written by kaggle/push.sh
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    prov.update({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                 'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else str(device)})
    (args.out / 'config.json').write_text(json.dumps({**vars(args), 'out': str(args.out), 'n_train': len(groups),
                                                      'provenance': prov}, indent=2))

    for epoch in range(start_epoch, args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(groups)
        batches = [sorted(order[i:i + args.batch].tolist()) for i in range(0, len(order), args.batch)]
        t0, losses, aux_losses, lag_losses, dist_losses, teach_losses, act_losses, pair_losses, class_losses = time.time(), [], [], [], [], [], [], [], []
        for b, (X, Y, M) in enumerate(prefetch(batches, None if args.gpu_features else transform, table=table)):
            X, Y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            batch_cls = np.asarray([class_of(g) for g in batches[b]], dtype=np.int64) if pair_atoms is not None else None
            if aux_head is not None:
                A, V = aux_targets(X[..., :112], args.aux_horizons)  # raw block comes first in both layouts
                if aux_std is None:
                    aux_std = ((A ** 2 * V).sum((0, 1)) / (V.sum(0) * len(A))).sqrt().clamp_min(1e-6)
                A = A / aux_std
            if act_head is not None:
                AT, AV = act_targets(X[..., :112], Y, args.act_horizons)
                if act_std is None:
                    act_std = ((AT ** 2 * AV).sum((0, 1)) / (AV.sum(0) * len(AT))).sqrt().clamp_min(1e-6)
                AT = AT / act_std
            Xraw = X
            if args.gpu_features:
                X = engineer(X)
            if teachers is not None:  # per sequence: the teacher that did not see it (see --teacher)
                order = [g for g in batches[b] if g < VALID_OFFSET] + [g for g in batches[b] if g >= VALID_OFFSET]
                pick = [0 if g >= VALID_OFFSET else 1 - g % 2 for g in order]
                tsel = []
                for j, (t, fut) in enumerate(teachers):
                    ix = torch.tensor([i for i, k in enumerate(pick) if k == j], device=device, dtype=torch.long)
                    if len(ix):
                        tsel.append((t, ix, torch.cat([X[ix], future_features(Xraw[ix], fut)], -1)))
                th = [None] * len(tsel)
            if args.future_inputs:
                X = torch.cat([X, future_features(Xraw, args.future_inputs)], -1)
            if mw is not None:
                with torch.no_grad():
                    R = args.mask_eps + mask_prob(X, mw)
                    if args.valid_true_mask:  # valid sequences: the official mask instead of the proxy
                        M = torch.from_numpy(M).to(device)
                        R = torch.where(M >= 0, args.mask_eps + M, R)
            if lag_head is not None:  # (B, T, 2 * n_lags) clipped y[t-H] (H < 0: future y[t+|H|]), zero (and zero weight) off the ends
                YL = torch.cat([torch.cat([torch.zeros_like(Y[:, :h]), Y[:, :-h]], 1) if h > 0 else
                                torch.cat([Y[:, -h:], torch.zeros_like(Y[:, :-h])], 1) for h in args.lag_targets], -1).clamp(-CLIP, CLIP)
                if args.pmean_targets:  # zero (and zero weight) for t < W
                    cs = torch.cat([torch.zeros_like(Y[:, :1]), Y.clamp(-CLIP, CLIP).cumsum(1)], 1)
                    YL = torch.cat([YL] + [torch.cat([torch.zeros_like(Y[:, :w]), (cs[:, w:-1] - cs[:, :-w - 1]) / w], 1)
                                           for w in args.pmean_targets], -1)
            hidden = None
            for c in range(0, X.shape[1], args.chunk):
                pred, hidden, top = model(X[:, c:c + args.chunk], hidden, return_top=True)
                hidden = [tuple(t.detach() for t in h) if isinstance(h, tuple) else h.detach() for h in hidden]
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
                if teachers is not None:  # after the aux block, which resets total = loss + aux
                    with torch.no_grad():
                        tp = torch.zeros_like(pred)
                        for k, (t, ix, XT) in enumerate(tsel):
                            tp[ix], th[k] = t(XT[:, c:c + args.chunk], th[k])
                    yc = Y[:, c + lo:c + args.chunk].clamp(-CLIP, CLIP)
                    wt = yc.abs() if mw is None else yc.abs() * R[:, c + lo:c + args.chunk, None]
                    distil = (wt * (pred[:, lo:] - tp[:, lo:].clamp(-CLIP, CLIP)) ** 2).sum() / wt.sum().clamp_min(1e-8)
                    total = total + args.teacher_weight * distil
                    teach_losses.append(distil.item())
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
                if model.pair_heads is not None:
                    class_target = torch.as_tensor(batch_cls, device=device, dtype=torch.long)
                    class_logits = model.class_head(top)
                    class_loss = nn.functional.cross_entropy(
                        class_logits.reshape(-1, len(STEPS)),
                        class_target[:, None].expand(-1, top.shape[1]).reshape(-1))
                    pair_loss = pred.new_zeros(())
                    target_y = Y[:, c + lo:c + args.chunk].clamp(-CLIP, CLIP)
                    for ci, (support, av) in enumerate(zip(pair_atoms, pair_atom_tensors)):
                        ix = torch.where(class_target == ci)[0]
                        if ix.numel() == 0:
                            continue
                        yy = target_y[ix]
                        j0 = (yy[..., 0, None] - av).abs().argmin(-1)
                        j1 = (yy[..., 1, None] - av).abs().argmin(-1)
                        labels = j0 * len(support) + j1
                        logits = model.pair_heads[ci](top[ix, lo:])
                        ce = nn.functional.cross_entropy(
                            logits.reshape(-1, len(support) ** 2), labels.reshape(-1), reduction='none').view(labels.shape)
                        rw = R[ix, c + lo:c + args.chunk] if mw is not None else torch.ones_like(ce)
                        pair_loss = pair_loss + (ce * rw).sum() / rw.sum().clamp_min(1.0) * (ix.numel() / len(batch_cls))
                    total = total + args.pair_weight * pair_loss + args.class_weight * class_loss
                    pair_losses.append(pair_loss.item())
                    class_losses.append(class_loss.item())
                if lag_head is not None:
                    yl = YL[:, c + lo:c + args.chunk]
                    wl = yl.abs()
                    lag = (wl * (lag_head(top[:, lo:]) - yl) ** 2).sum() / wl.sum().clamp_min(1e-8)
                    total = total + args.lag_weight * lag
                    lag_losses.append(lag.item())
                if act_head is not None:
                    v = AV[c:c + args.chunk]
                    act = (((act_head(top) - AT[:, c:c + args.chunk]) ** 2) * v).sum() / (v.sum() * len(top)).clamp_min(1.0)
                    total = total + args.act_weight * act
                    act_losses.append(act.item())
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
                           (f' ce {np.mean(dist_losses[-200:]):.4f}' if dist_losses else '') + \
                           (f' teach {np.mean(teach_losses[-200:]):.4f}' if teach_losses else '') + \
                           (f' act {np.mean(act_losses[-200:]):.4f}' if act_losses else '') + \
                           (f' pair {np.mean(pair_losses[-200:]):.4f} class {np.mean(class_losses[-200:]):.4f}' if pair_losses else '')
                print(f'epoch {epoch} batch {b + 1}/{len(batches)} loss {np.mean(losses[-200:]):.4f}{aux_note} '
                      f'lr {sched.get_last_lr()[0]:.2e} {rate / 1e3:.0f}k rows/s', flush=True)
        score = monitor_score(model, monitor, device, transform, args.gpu_features, future=args.future_inputs, table=table)
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
                    'lag_head': lag_head.state_dict() if lag_head is not None else None,
                    'act_head': act_head.state_dict() if act_head is not None else None, 'act_std': act_std}, ckpt_path)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
