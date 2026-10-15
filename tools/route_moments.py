"""Score class routing from saved per-sequence moments: sequences of the given asset classes take a specialist's
moments, all others the base's. Uses the true class labels (the warm-up detector is 99.9% accurate).

  python tools/route_moments.py BASE SPECIALIST --classes 3 --name NAME   # writes results/NAME.npz/json
  python tools/route_moments.py RUN [RUN ...] --per-class                 # per-class WP of saved runs

Then `python src/evaluate.py compare BASE NAME` gives the paired bootstrap.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import holdout_groups  # noqa: E402
from evaluate import RESULTS, summarize, wp_from_moments  # noqa: E402


def view_classes():
    """Class label per sequence, in the order of each view's moments."""
    valid = np.array(json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls'])
    train = np.array(json.loads((ROOT / 'splits' / 'train_classes.json').read_text()))
    return {'V1': valid, 'V2': valid, 'V3': train[holdout_groups()]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--classes', type=int, nargs='+')
    ap.add_argument('--name')
    ap.add_argument('--per-class', action='store_true')
    args = ap.parse_args()
    cls = view_classes()
    if args.per_class:
        for run in args.runs:
            m = np.load(RESULTS / f'{run}.npz')
            print(run, '  '.join(f"{v}: " + ' '.join(f'c{c} {wp_from_moments(m[v][cls[v] == c].sum(0)).mean():.4f}'
                                                     for c in range(4)) for v in m.files))
        sys.exit()
    base, spec = (np.load(RESULTS / f'{r}.npz') for r in args.runs)
    out = {v: np.where(np.isin(cls[v], args.classes)[:, None, None], spec[v], base[v]) for v in base.files}
    np.savez(RESULTS / f'{args.name}.npz', **out)
    summary = {v: summarize(m) for v, m in out.items()}
    (RESULTS / f'{args.name}.json').write_text(json.dumps(summary, indent=2))
    print(args.name, '  '.join(f"{v} {s['wp']:.4f}" for v, s in summary.items()))
