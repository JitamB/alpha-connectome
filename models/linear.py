"""Phase 3 linear diagnostic: |y|-weighted ridge on causal feature blocks (not for submission).

One streaming pass accumulates the weighted Gram matrix over all blocks; every feature set is a
column subset of it, so all variants are fitted from the same pass and evaluated on V1/V2/V3.

  python models/linear.py [--train-seqs 1000] [--every 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import ROOT, TRAIN, WARMUP, iter_sequences, train_groups  # noqa: E402
from evaluate import evaluate, print_summary  # noqa: E402

EMA_SPANS = (10, 100, 1000)
BLOCKS = ['raw', 'diff', *[f'ema_dev_{h}' for h in EMA_SPANS], 'absdiff_ema_100']
FEATURE_SETS = {
    'raw': ['raw'],
    'raw+diff': ['raw', 'diff'],
    'raw+diff+ema': ['raw', 'diff', *[f'ema_dev_{h}' for h in EMA_SPANS]],
    'all': BLOCKS,
}
RIDGE = 1e-3
CLIP = 2.0


def ema(x, span):
    """Causal EMA along time, initialised at the first row."""
    a = 2.0 / (span + 1)
    return lfilter([a], [1, a - 1], x, axis=0, zi=((1 - a) * x[:1]))[0]


def feature_blocks(X):
    """X: (T, 112) -> dict of (T, 112) causal feature blocks."""
    diff = np.vstack([np.zeros((1, X.shape[1]), X.dtype), np.diff(X, axis=0)])
    out = {'raw': X, 'diff': diff}
    for h in EMA_SPANS:
        out[f'ema_dev_{h}'] = X - ema(X, h)
    out['absdiff_ema_100'] = ema(np.abs(diff), 100)
    return out


def design(X, blocks):
    f = feature_blocks(X)
    return np.hstack([np.ones((len(X), 1))] + [f[b] for b in blocks]).astype(np.float64)


def columns(blocks):
    """Column indices of `blocks` (plus intercept) inside the full design matrix."""
    idx = [0]
    for b in blocks:
        start = 1 + BLOCKS.index(b) * 112
        idx += range(start, start + 112)
    return np.array(idx)


def accumulate(groups, every):
    n = 1 + len(BLOCKS) * 112
    G, b = np.zeros((2, n, n)), np.zeros((2, n))
    for i, s in enumerate(iter_sequences(TRAIN, groups)):
        rows = np.arange(WARMUP, len(s.X), every)
        A = design(s.X, BLOCKS)[rows]
        y = np.clip(s.y[rows], -CLIP, CLIP).astype(np.float64)
        for k in range(2):
            Aw = A * np.abs(y[:, k])[:, None]
            G[k] += A.T @ Aw
            b[k] += Aw.T @ y[:, k]
        if (i + 1) % 100 == 0:
            print(f'  accumulated {i + 1}/{len(groups)} sequences', flush=True)
    return G, b


def solve(G, b, cols):
    coef = np.zeros((len(cols), 2))
    for k in range(2):
        g = G[k][np.ix_(cols, cols)]
        reg = RIDGE * np.diag(np.diag(g))
        reg[0, 0] = 0
        coef[:, k] = np.linalg.solve(g + reg, b[k][cols])
    return coef


class LinearPredictor:
    def __init__(self, blocks, coef):
        self.blocks, self.coef = blocks, coef

    def __call__(self, Xb):
        return np.stack([design(X, self.blocks) @ self.coef for X in Xb]).astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-seqs', type=int, default=1000)
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--sets', nargs='+', default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    args = ap.parse_args()
    groups = sorted(np.random.default_rng(0).choice(train_groups(), args.train_seqs, replace=False).tolist())
    print(f'fitting on {len(groups)} train sequences, every {args.every}th row')
    G, b = accumulate(groups, args.every)
    summaries = {}
    for name in args.sets:
        blocks = FEATURE_SETS[name]
        coef = solve(G, b, columns(blocks))
        run = f'linear_{name}'
        np.save(ROOT / 'results' / f'{run}_coef.npy', coef)
        summaries[run] = evaluate(LinearPredictor(blocks, coef), run)
        print_summary(run, summaries[run])
    if len(args.sets) == len(FEATURE_SETS):
        (ROOT / 'results' / 'linear_summary.json').write_text(json.dumps(summaries, indent=2))
