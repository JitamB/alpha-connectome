"""Seed-aware judge for Connectome experiments (procedure: .claude/skills/connectome-experiment/SKILL.md).

  python tools/judge.py --base lr7e4_full lr7e4_full_seed1 --new X X_seed1 [--valid-from 937]
  python tools/judge.py --cells lr7e4_full,lr7e4_full_seed1 A,A_seed1 B,B_seed1 AB,AB_seed1     (2x2: 00 10 01 11)

Evidence unit: the same-seed paired delta (names listed in the same seed order). For the seed mean of n deltas,
SE = sqrt(sigma_t^2 / n + sigma_b^2): sigma_t = training noise of one same-seed paired delta (replicate spread of the
full-data recipe, 2026-10-22 review), sigma_b = paired bootstrap over sequences. z_fix uses sigma_t only.
Views: V1, V2, V3; V1B/V2B = valid row groups >= --valid-from (clean half when valid half A is in training);
V1t = V1 reweighted to the fitted public-test class mix (secondary check only); V1c0..3 = V1 per asset class.
Verdict (train-only runs: V1 decides; with --valid-from: V1B decides; V2/V3 can only veto):
  1 seed : STOP if delta < -0.0006, CONTINUE if > +0.0006, else WEAK
  2 seeds: CANDIDATE if mean >= +0.0009 (2x2 main effect: +0.0006), no seed < -0.0006 and V2/V3 (V2B/V3) not below -2 SE; else REJECT
  3+     : ADOPT if mean >= +0.0007 (count only seeds not used to pick the variant)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate import RESULTS, wp_from_moments  # noqa: E402

SIGMA_T = {'V1': 0.00062, 'V2': 0.0007, 'V3': 0.00045, 'V1B': 0.0007, 'V2B': 0.0008, 'V1t': 0.00065}
TEST_MIX = np.array([0.22, 0.20, 0.04, 0.54])  # public-test class shares fitted from 7 LB scores (unproven)
VCLS = np.array(json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls'])


def views(name, valid_from):
    d = np.load(RESULTS / f'{name}.npz')
    w = (TEST_MIX / (np.bincount(VCLS, minlength=4) / len(VCLS)))[VCLS][:, None, None]
    out = {'V1': d['V1'], 'V2': d['V2'], 'V3': d['V3'], 'V1t': d['V1'] * w}
    if valid_from:
        out['V1B'], out['V2B'] = d['V1'][valid_from:], d['V2'][valid_from:]
    out.update({f'V1c{c}': d['V1'][VCLS == c] for c in range(4)})
    return out


def wp(m):
    return wp_from_moments(m).mean(-1)


def contrast(groups, coef, n_boot=1000, seed=0):
    """groups: per seed, a list of per-sequence moment arrays (one per cell); coef: weights of the cells.
    Returns per-seed contrasts, their mean and the bootstrap sd of the mean (sequences resampled jointly)."""
    n = len(groups[0][0])
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot).astype(np.float64)
    per_seed, boot = [], np.zeros(n_boot)
    for cells in groups:
        per_seed.append(sum(c * wp(m.sum(0)) for c, m in zip(coef, cells)))
        boot += sum(c * wp((counts @ m.reshape(n, 12)).reshape(n_boot, 6, 2)) for c, m in zip(coef, cells))
    return np.array(per_seed), float(np.mean(per_seed)), float((boot / len(groups)).std())


def report(title, runs, coef, valid_from, sigma_scale):
    """runs: list over seeds of lists of names (one per cell)."""
    vs = [[views(n, valid_from) for n in cells] for cells in runs]
    res = {}
    for v in vs[0][0]:
        per, mean, sb = contrast([[cv[v] for cv in cells] for cells in vs], coef)
        st = SIGMA_T.get(v)
        se = np.sqrt((st * sigma_scale) ** 2 / len(runs) + sb ** 2) if st else sb
        res[v] = {'per_seed': per.round(5).tolist(), 'mean': round(mean, 5), 'se': round(float(se), 5),
                  'z_fix': round(mean / (st * sigma_scale / np.sqrt(len(runs))), 1) if st else None,
                  'z_gen': round(mean / se, 1)}
    print(f'== {title}')
    for v, r in res.items():
        print(f"  {v:5s} per seed {r['per_seed']}  mean {r['mean']:+.5f}  SE {r['se']:.5f}  z_fix {r['z_fix']}  z_gen {r['z_gen']}")
    return res


def verdict(res, primary, thr=0.0009):
    r, n = res[primary], len(res[primary]['per_seed'])
    vetoes = [v for v in ('V2B' if primary == 'V1B' else 'V2', 'V3') if v in res and res[v]['mean'] < -2 * res[v]['se']]
    if n == 1:
        d = r['mean']
        return 'STOP' if d < -0.0006 else 'CONTINUE' if d > 0.0006 else 'WEAK'
    if n == 2:
        ok = r['mean'] >= thr and min(r['per_seed']) >= -0.0006 and not vetoes
        return 'CANDIDATE' if ok else f'REJECT{" (veto " + ",".join(vetoes) + ")" if vetoes else ""}'
    return 'ADOPT (if these seeds were not used to pick the variant)' if r['mean'] >= 0.0007 and not vetoes else 'REJECT'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', nargs='+')
    ap.add_argument('--new', nargs='+')
    ap.add_argument('--cells', nargs=4, metavar=('C00', 'C10', 'C01', 'C11'), help='comma-separated names per cell')
    ap.add_argument('--valid-from', type=int, default=0)
    args = ap.parse_args()
    primary = 'V1B' if args.valid_from else 'V1'
    if args.cells:
        cells = [c.split(',') for c in args.cells]
        runs = [list(r) for r in zip(*cells)]
        # main effects / interaction of a 2x2 per seed block; one cell-run has sd sigma_t / sqrt(2)
        for title, coef in (('factor 1 (C10, C11 vs C00, C01)', (-.5, .5, -.5, .5)),
                            ('factor 2 (C01, C11 vs C00, C10)', (-.5, -.5, .5, .5)),
                            ('interaction', (.5, -.5, -.5, .5))):
            print(f"   verdict: {verdict(report(title, runs, coef, args.valid_from, np.sqrt(0.5)), primary, 0.0006)}")
        print(f"   combined C11 vs C00 verdict: {verdict(report('C11 vs C00', [[r[0], r[3]] for r in runs], (-1, 1), args.valid_from, 1.0), primary)}")
    else:
        assert len(args.base) == len(args.new), 'list base and new runs in the same seed order'
        res = report(f"{','.join(args.new)} vs {','.join(args.base)}", [list(p) for p in zip(args.base, args.new)],
                     (-1, 1), args.valid_from, 1.0)
        print(f'   verdict: {verdict(res, primary)}')


if __name__ == '__main__':
    main()
