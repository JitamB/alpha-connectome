"""Class-3 error anatomy: dump per-row predictions and covariates of the train-only pair on scored valid rows (Kaggle).

  python tools/c3_anatomy.py runs/lr7e4_full runs/lr7e4_full_seed1 --name c3_anatomy [--classes 3 2]

Writes results/<name>.npz: one row per scored valid row of the chosen classes, columns in `cols` (float32), plus
`X` = the 112 raw features (float16) on every 4th of those rows (`xrow` = index into the main table).
Causal covariates: time since the last i0 / i1 price-shape change, i0 / i1 changes in the last 100 / 1000 rows,
shape index (order of first appearance in the sequence), distinct shapes so far. Future-looking columns (analysis
only, prefixed f_): rows until the next i0 change, i0 changes in the next 100 rows, ticks at t+100 / t+300, ticks at
the next i0 change. Ticks are y / step rounded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'src'), str(ROOT / 'train')]
import gate_paths as G  # noqa: E402
from data import VALID, iter_sequences  # noqa: E402
from eval_torch import load  # noqa: E402
from train_gru import STEPS  # noqa: E402


def since(flag):
    t = np.arange(len(flag))
    return t - np.maximum.accumulate(np.where(flag, t, 0))


def count_back(flag, w):
    c = np.cumsum(flag)
    return c - np.r_[np.zeros(w), c][:len(c)]


def covariates(X, y, step):
    """-> dict name -> (T,) array for every row of one sequence."""
    T = len(X)
    t = np.arange(T)
    tr0, _ = G.transitions(X[:, 0:22])
    tr1, _ = G.transitions(X[:, 52:74])
    ids = G.shape_ids(X[:, 0:22])
    _, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first))                       # shape index by order of first appearance
    shape = rank[inv]
    n = np.rint(y.astype(np.float64) / step)
    c0 = np.cumsum(tr0)
    nxt = np.minimum.accumulate(np.where(tr0, t, T)[::-1])[::-1]          # first change at or after t
    nxt = np.r_[nxt[1:], T]                                              # first change strictly after t
    at_next = n[np.minimum(nxt, T - 1)]
    return {
        't': t, 'age0': since(tr0), 'age1': since(tr1),
        'tr0_100': count_back(tr0, 100), 'tr0_1000': count_back(tr0, 1000),
        'tr1_100': count_back(tr1, 100), 'tr1_1000': count_back(tr1, 1000),
        'shape': shape, 'nshapes': np.maximum.accumulate(shape) + 1,
        'n0': n[:, 0], 'n1': n[:, 1], 'y0': y[:, 0], 'y1': y[:, 1],
        'f_next0': nxt - t, 'f_tr0_100': c0[np.minimum(t + 100, T - 1)] - c0,
        'f_n0_100': n[np.minimum(t + 100, T - 1), 0], 'f_n1_100': n[np.minimum(t + 100, T - 1), 1],
        'f_n0_300': n[np.minimum(t + 300, T - 1), 0], 'f_n1_300': n[np.minimum(t + 300, T - 1), 1],
        'f_n0_next': at_next[:, 0], 'f_n1_next': at_next[:, 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--name', required=True)
    ap.add_argument('--classes', type=int, nargs='+', default=[3])
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    groups = [g for g in range(len(vcls)) if vcls[g] in args.classes]
    models = [load(Path(r), 'best.pt', device)[0] for r in args.runs]
    tables, xs, chunk = [], [], []
    it = iter_sequences(VALID, groups)
    while True:
        s = next(it, None)
        if s is not None:
            chunk.append(s)
        if chunk and (s is None or len(chunk) == args.batch):
            for s_, q in zip(chunk, G.predict(models, chunk, device)):
                c = vcls[s_.group]
                m = s_.need & s_.scored
                cov = covariates(s_.X, s_.y, STEPS[c])
                cov.update({'seq': np.full(len(m), s_.group), 'cls': np.full(len(m), c), 'p0': q[:, 0], 'p1': q[:, 1]})
                tables.append({k: np.asarray(v)[m].astype(np.float32) for k, v in cov.items()})
                xs.append(s_.X[m][::4].astype(np.float16))
            chunk = []
        if s is None:
            break
    cols = list(tables[0])
    data = np.stack([np.concatenate([tb[k] for tb in tables]) for k in cols], 1)
    offs = np.cumsum([0] + [len(tb['t']) for tb in tables])
    xrow = np.concatenate([offs[i] + np.arange(0, len(tb['t']), 4) for i, tb in enumerate(tables)])
    (ROOT / 'results').mkdir(exist_ok=True)
    np.savez_compressed(ROOT / 'results' / f'{args.name}.npz', data=data, cols=np.array(cols), X=np.concatenate(xs),
                        xrow=xrow, runs=np.array(args.runs))
    print(f'saved {args.name}: {data.shape[0]} scored rows from {len(groups)} sequences, {len(cols)} columns', flush=True)


if __name__ == '__main__':
    main()
