"""Build the i1 lead-lag tables (src/i1_lead.py): mean future i0-target change after each i1 transition type.

  python tools/build_i1_table.py --out results/i1_table.npz [--workers 4]

Tables 0..3 are out-of-fold: table k uses train groups with g % 4 != k (holdout groups are never used), so a training
sequence never sees its own future; table 4 uses every non-holdout train group (validation, holdout and test time).
Types seen fewer than --min-count times in a table are dropped (they are shrunk to ~0 anyway).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from data import TRAIN, holdout_groups, iter_sequences, train_groups  # noqa: E402
import i1_lead as L  # noqa: E402


def agg(typ, fut):
    keys, inv = np.unique(typ, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=fut[:, j], minlength=len(keys)) for j in range(2)], 1)
    return keys, sums, np.bincount(inv, minlength=len(keys)).astype(np.float64)


def work(groups):
    """-> {fold: (keys, sums, counts)} over the given train groups, fold = g % N_FOLDS."""
    per = {k: ([], []) for k in range(L.N_FOLDS)}
    for s in iter_sequences(TRAIN, groups):
        tr, typ = L.transition_types(s.X[:, L.PRICE_I1])
        idx = np.flatnonzero(tr)
        per[s.group % L.N_FOLDS][0].append(typ[idx])
        per[s.group % L.N_FOLDS][1].append(L.future_change(s.y)[idx])
    return {k: agg(np.concatenate(t), np.concatenate(f)) for k, (t, f) in per.items() if t}


def merge(parts):
    keys = np.concatenate([p[0] for p in parts])
    sums = np.concatenate([p[1] for p in parts])
    cnt = np.concatenate([p[2] for p in parts])
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.stack([np.bincount(inv, weights=sums[:, j], minlength=len(u)) for j in range(2)], 1), \
        np.bincount(inv, weights=cnt, minlength=len(u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min-count', type=int, default=2)
    ap.add_argument('--max-groups', type=int, default=0, help='smoke test only')
    args = ap.parse_args()
    groups = train_groups()
    if args.max_groups:
        groups = groups[:args.max_groups]
    assert not set(groups) & set(holdout_groups())
    chunks = [groups[i::args.workers * 4] for i in range(args.workers * 4)]
    with Pool(args.workers) as p:
        results = p.map(work, chunks)
    folds = {k: merge([r[k] for r in results if k in r]) for k in range(L.N_FOLDS)}
    out = {}
    for k in range(L.N_FOLDS + 1):
        parts = [folds[j] for j in range(L.N_FOLDS) if j != k and j in folds]
        keys, sums, cnt = merge(parts)
        keep = cnt >= args.min_count
        out[f'keys{k}'] = keys[keep]
        out[f'vals{k}'] = (sums[keep] / (cnt[keep, None] + L.SHRINK)).astype(np.float32)
        print(f'table {k}: {int(keep.sum())} types kept of {len(keys)}, transitions {int(cnt.sum())}, '
              f'mean |value| {np.abs(out[f"vals{k}"]).mean():.4f}', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print('saved', args.out, Path(args.out).stat().st_size // 1024, 'KB', flush=True)


if __name__ == '__main__':
    main()
