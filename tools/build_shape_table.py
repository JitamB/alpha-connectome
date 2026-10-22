"""Build the exact-shape lookup table (src/shape_lookup.py) from labelled sequences.

  python tools/build_shape_table.py --out models/shape_table_train.npz [--valid] [--every 2] [--min-count 2]

Default sources: all train row groups except the holdout (splits/holdout.json), so V3 and valid stay out of sample.
--valid also adds all valid sequences (final train+valid models; then only V3 stays clean).
Stored: keys (uint64, sorted), stats (float32: count, sum S*step, sum D*step, count S<=-3, =-2, =-1, >=0),
cls (uint8 majority class), prior (float64, (5, 6): per majority class 0..3 and global: E[S*step], E[D*step], P(bucket) x 4), info.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from data import TRAIN, VALID, iter_sequences, train_groups  # noqa: E402
from shape_lookup import shape_ids  # noqa: E402

STEPS = (0.12543297, 0.24132967, 0.4685359, 0.65686762)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--valid', action='store_true')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--min-count', type=int, default=2)
    args = ap.parse_args()
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    sources = [(TRAIN, train_groups(), tcls)] + ([(VALID, list(range(1873)), vcls)] if args.valid else [])
    parts = []
    for path, groups, cls_of in sources:
        for s in iter_sequences(path, groups):
            c = cls_of[s.group]
            st = STEPS[c]
            n = np.rint(s.y.astype(np.float64) / st).astype(np.int64)[::args.every]
            sid = shape_ids(s.X[::args.every, :22])
            # aggregate per sequence first (keeps memory small): unique shapes with their sums
            k, inv = np.unique(sid, return_inverse=True)
            S, D = n.sum(1), n[:, 0] - n[:, 1]
            b = np.select([S <= -3, S == -2, S == -1], [0, 1, 2], 3)
            cols = [np.bincount(inv, minlength=len(k)).astype(np.float64),
                    np.bincount(inv, weights=S * st, minlength=len(k)), np.bincount(inv, weights=D * st, minlength=len(k))]
            cols += [np.bincount(inv[b == j], minlength=len(k)).astype(np.float64) for j in range(4)]
            parts.append((k, np.stack(cols, 1), np.full(len(k), c, np.uint8)))
    keys = np.concatenate([p[0] for p in parts])
    stats = np.concatenate([p[1] for p in parts])
    cls = np.concatenate([p[2] for p in parts])
    uk, inv = np.unique(keys, return_inverse=True)
    agg = np.zeros((len(uk), 7))
    np.add.at(agg, inv, stats)
    cls_cnt = np.zeros((len(uk), 4))
    np.add.at(cls_cnt, (inv, cls.astype(np.int64)), stats[:, 0])
    major = cls_cnt.argmax(1)
    prior = np.zeros((5, 6))
    for c in range(5):  # per class (majority class of the shape), then global
        tot = agg[major == c].sum(0) if c < 4 else agg.sum(0)
        prior[c] = np.r_[tot[1] / tot[0], tot[2] / tot[0], tot[3:] / tot[0]] if tot[0] > 0 else 0.0
    tot = agg.sum(0)
    keep = agg[:, 0] >= args.min_count
    info = {'rows': float(tot[0]), 'shapes': int(len(uk)), 'kept': int(keep.sum()),
            'row_coverage_of_kept': float(agg[keep, 0].sum() / tot[0]), 'mixed_class_shapes': int(((cls_cnt > 0).sum(1) > 1).sum()),
            'per_class_shapes': [int(((cls_cnt.argmax(1) == c) & keep).sum()) for c in range(4)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, keys=uk[keep], stats=agg[keep].astype(np.float32), cls=major[keep].astype(np.uint8),
                        prior=prior, info=json.dumps(info))
    out = Path(args.out)
    print(f'{out} ({out.stat().st_size / 1e6:.1f} MB): {info}', flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{out.stem}.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
