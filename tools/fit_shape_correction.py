"""Post-hoc exact-shape correction for a packaged ensemble: per-class |y|-weighted ridge on [p0, p1, lookup features],
fitted on the train holdout (out of sample for train-only and train+valid models alike).

  python tools/fit_shape_correction.py runs/a runs/b --table results/shape_table_train.npz --name corr_X [--eval-valid]

Writes results/<name>.npz (mu, sd, W per class: deployment coefficients, see src/fast_gru.py ShapeCorrection) and
results/<name>.json. --eval-valid (train-only runs only): V1/V2 on valid with and without the correction, plus the
same with a correction fitted on valid half A and scored on half B (protocol G1 / G2 of tools/shape_gate.py).
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
from data import TRAIN, VALID, holdout_groups, iter_sequences  # noqa: E402
from eval_torch import load  # noqa: E402
from shape_lookup import features as shape_features, load_table  # noqa: E402


def build(path, groups, cls_of, table, models, device, batch=16):
    rows, chunk = [], []
    it = iter_sequences(path, groups)
    while True:
        s = next(it, None)
        if s is not None:
            chunk.append(s)
        if chunk and (s is None or len(chunk) == batch):
            for s_, q in zip(chunk, G.predict(models, chunk, device)):
                f, _ = shape_features(s_.X, *table)
                rows.append({'y': np.clip(s_.y, -G.CLIP, G.CLIP).astype(np.float64), 'p': q, 'need': s_.need,
                             'scored': s_.scored if s_.scored is not None else np.ones_like(s_.need),
                             'cls': cls_of[s_.group], 'group': s_.group, 'f': {'shape': f}})
            chunk = []
        if s is None:
            return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--table', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--eval-valid', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    table = load_table(args.table)
    models = [load(Path(r), 'best.pt', device)[0] for r in args.runs]
    G.GROUPS = {'shape': ['shape'], 'all': ['shape']}
    hold = build(TRAIN, holdout_groups(), tcls, table, models, device)
    for r in hold:
        r['fitmask'] = r['need'].copy()
    fit = G.fit_ridge(hold, 'shape', 'fitmask')
    out = {'mu': np.stack([fit[c][0] for c in range(4)]), 'sd': np.stack([fit[c][1] for c in range(4)]),
           'W': np.stack([fit[c][2] for c in range(4)])}
    (ROOT / 'results').mkdir(exist_ok=True)
    np.savez(ROOT / 'results' / f'{args.name}.npz', **out)
    res = {'runs': args.runs, 'table': args.table, 'V3_holdout_in_sample': {
        k: G.score(hold, f, 'V2')['all'] for k, f in (('pred', lambda r: r['p']),
                                                     ('corrected', lambda r: G.apply_ridge(fit, r, 'shape')))}}
    if args.eval_valid:
        valid = build(VALID, list(range(1873)), vcls, table, models, device)
        for r in valid:
            r['fitmask'] = r['need'] & r['scored']
        for view in ('V1', 'V2'):
            res[f'{view}_pred'] = G.score(valid, lambda r: r['p'], view)
            res[f'{view}_corrected_G1'] = G.score(valid, lambda r: G.apply_ridge(fit, r, 'shape'), view)
        A = [r for r in valid if r['group'] < 937]
        B = [r for r in valid if r['group'] >= 937]
        res['G2_A_to_B'] = G.gate(A, B, 'fitmask')
        res['G2_B_to_A'] = G.gate(B, A, 'fitmask')
        d = {v: res[f'{v}_corrected_G1']['all'] - res[f'{v}_pred']['all'] for v in ('V1', 'V2')}
        print(f"G1 (fit holdout, score valid): V1 {res['V1_pred']['all']:.5f} -> {res['V1_corrected_G1']['all']:.5f} "
              f"({d['V1']:+.5f}); V2 {d['V2']:+.5f}; per class V1: " +
              ' '.join(f"c{c} {res['V1_corrected_G1'][f'c{c}'] - res['V1_pred'][f'c{c}']:+.4f}" for c in range(4)), flush=True)
        for k in ('G2_A_to_B', 'G2_B_to_A'):
            g = res[k]['V1']
            print(f"{k} V1: ridge[pred] {g['none']['all']:.5f} +shape {g['shape']['all'] - g['none']['all']:+.5f}", flush=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2, default=float))
    print('saved', args.name, res['V3_holdout_in_sample'], flush=True)


if __name__ == '__main__':
    main()
