"""Gate for exact-shape lookup features on top of the train-only pair (stacking, same protocol as tools/gate_paths.py).

  python tools/shape_gate.py runs/lr7e4_full runs/lr7e4_full_seed1 --name shape_gate [--n-table 4000]

Forensics A1 (2026-10-22, results/forensics_a1.json): S = n0 + n1 is a near-deterministic function of the exact current
i0 price shape (22 float32 bit patterns): a global modal-S table fitted on other sequences is right on 92% (class 3),
75% (class 2), 65% / 61% (classes 1 / 0) of segments, with 95-99.99% coverage. The GRU's p0 + p1 barely tracks S on
class 3 (corr 0.03), and true S was worth +0.028 class-3 t0 WP (experiments.md 2026-10-15). The old gate only tested
window sums of decoded target *changes*, never the per-shape level.
Features per row (current shape; train tables exclude the holdout and all valid sequences), in target units:
  hit, log1p(count), E[S], E[D] (shrunk to the class prior), P(S >= 0), P(S = -1), P(S = -2), P(S <= -3).
Protocols G1 (fit holdout, test valid) and G2 (valid half A <-> B); baseline = per-class ridge on the predictions.
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
from data import TRAIN, VALID, holdout_groups, iter_sequences, train_groups  # noqa: E402
from eval_torch import load  # noqa: E402
from train_gru import STEPS  # noqa: E402

SHRINK = 20.0
BUCKETS = ((None, -3), (-2, -2), (-1, -1), (0, None))   # S <= -3, S = -2, S = -1, S >= 0


def learn_table(groups, cls_of, every=4):
    """exact i0 shape -> [count, sum S, sum D, counts per S bucket] over every `every`-th row of the given train groups."""
    sids, S, D = [], [], []
    for s in iter_sequences(TRAIN, groups):
        n = np.rint(s.y.astype(np.float64) / STEPS[cls_of[s.group]]).astype(np.int64)[::every]
        sids.append(G.shape_ids(s.X[::every, 0:22]))
        S.append(n.sum(1))
        D.append(n[:, 0] - n[:, 1])
    sids, S, D = np.concatenate(sids), np.concatenate(S), np.concatenate(D)
    keys, inv = np.unique(sids, return_inverse=True)
    cols = [np.bincount(inv, minlength=len(keys)).astype(np.float64),
            np.bincount(inv, weights=S, minlength=len(keys)), np.bincount(inv, weights=D, minlength=len(keys))]
    for lo, hi in BUCKETS:
        m = np.ones(len(S), bool) if lo is None else S >= lo
        m &= np.ones(len(S), bool) if hi is None else S <= hi
        cols.append(np.bincount(inv[m], minlength=len(keys)).astype(np.float64))
    return keys, np.stack(cols, 1), {'rows': int(len(S)), 'shapes': int(len(keys)), 'mean_S': float(S.mean()),
                                     'mean_D': float(D.mean())}


def shape_features(X, step, table, prior):
    keys, stats = table
    sid = G.shape_ids(X[:, 0:22])
    pos = np.clip(np.searchsorted(keys, sid), 0, len(keys) - 1)
    hit = keys[pos] == sid
    st = np.where(hit[:, None], stats[pos], 0.0)
    cnt = st[:, 0]
    eS = (st[:, 1] + SHRINK * prior[0]) / (cnt + SHRINK)
    eD = (st[:, 2] + SHRINK * prior[1]) / (cnt + SHRINK)
    pb = (st[:, 3:] + SHRINK * prior[2:]) / (cnt + SHRINK)[:, None]
    f = np.column_stack([hit, np.log1p(cnt), eS * step, eD * step, pb])
    return f.astype(np.float32), sid


def build(path, groups, cls_of, tables, priors, models, device, batch=16):
    rows, chunk, diag = [], [], {'S_acc': {c: [0, 0] for c in range(4)}, 'hit': {c: [0, 0] for c in range(4)}}
    it = iter_sequences(path, groups)
    while True:
        s = next(it, None)
        if s is not None:
            chunk.append(s)
        if chunk and (s is None or len(chunk) == batch):
            for s_, q in zip(chunk, G.predict(models, chunk, device)):
                c = cls_of[s_.group]
                f, _ = shape_features(s_.X, STEPS[c], tables[c], priors[c])
                n = np.rint(s_.y.astype(np.float64) / STEPS[c]).astype(np.int64)
                need = s_.need & (s_.scored if s_.scored is not None else True)
                modal = np.argmax(f[:, 4:], 1)                     # bucket index of the most likely S
                true_b = np.select([n.sum(1) <= -3, n.sum(1) == -2, n.sum(1) == -1], [0, 1, 2], 3)
                diag['S_acc'][c][0] += int(((modal == true_b) & need).sum())
                diag['S_acc'][c][1] += int(need.sum())
                diag['hit'][c][0] += int((f[:, 0] > 0)[need].sum())
                diag['hit'][c][1] += int(need.sum())
                rows.append({'y': np.clip(s_.y, -G.CLIP, G.CLIP).astype(np.float64), 'p': q, 'need': s_.need,
                             'scored': s_.scored, 'cls': c, 'group': s_.group, 'f': {'shape': f}})
            chunk = []
        if s is None:
            return rows, diag


def sum_corr(rows, cls=3):
    """corr(p0 + p1, y0 + y1) and corr(E[S] lookup, y0 + y1) on scored rows of one class (headroom check)."""
    a, b, l = [], [], []
    for r in rows:
        if r['cls'] != cls:
            continue
        m = r['need'] & r['scored']
        a.append(r['p'][m].sum(1)); b.append(r['y'][m].sum(1)); l.append(r['f']['shape'][m, 2])
    a, b, l = np.concatenate(a), np.concatenate(b), np.concatenate(l)
    return {'corr_pred_sum_vs_S': float(np.corrcoef(a, b)[0, 1]), 'corr_lookup_vs_S': float(np.corrcoef(l, b)[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--name', required=True)
    ap.add_argument('--n-table', type=int, default=4000)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    res = {}
    tg = train_groups()[:args.n_table]
    tables, priors = {}, {}
    for c in range(4):
        keys, stats, info = learn_table([g for g in tg if tcls[g] == c], tcls)
        tables[c] = (keys, stats)
        tot = stats.sum(0)
        priors[c] = np.r_[tot[1] / tot[0], tot[2] / tot[0], tot[3:] / tot[0]]
        res[f'table_c{c}'] = info
        print(f'class {c} table: {info}', flush=True)
    models = [load(Path(r), 'best.pt', device)[0] for r in args.runs]
    hold, res['diag_holdout'] = build(TRAIN, holdout_groups(), tcls, tables, priors, models, device)
    for r in hold:
        r['scored'] = np.ones_like(r['need'])
    valid, res['diag_valid'] = build(VALID, list(range(1873)), vcls, tables, priors, models, device)
    for k in ('diag_holdout', 'diag_valid'):
        res[k] = {m: {c: (v[0] / max(v[1], 1)) for c, v in d.items()} for m, d in res[k].items()}
    res['class3_sum_corr_valid'] = sum_corr(valid, 3)
    res['class2_sum_corr_valid'] = sum_corr(valid, 2)
    print('diagnostics', res['diag_valid'], res['class3_sum_corr_valid'], flush=True)
    for r in hold + valid:
        r['fitmask'] = r['need'] & r['scored']
    G.GROUPS = {'shape': ['shape'], 'all': ['shape']}
    res['G1_holdout_to_valid'] = G.gate(hold, valid, 'fitmask')
    A = [r for r in valid if r['group'] < 937]
    B = [r for r in valid if r['group'] >= 937]
    res['G2_A_to_B'] = G.gate(A, B, 'fitmask')
    res['G2_B_to_A'] = G.gate(B, A, 'fitmask')
    for k in ('G1_holdout_to_valid', 'G2_A_to_B', 'G2_B_to_A'):
        for view in ('V1', 'V2'):
            g = res[k][view]
            base = g['none']['all']
            print(f"{k} {view}: pred {g['pred']['all']:.5f} ridge[pred] {base:.5f} | +shape {g['shape']['all'] - base:+.5f} "
                  f"| per class: " + ' '.join(f"c{c} {g['shape'][f'c{c}'] - g['none'][f'c{c}']:+.4f}" for c in range(4)),
                  flush=True)
    res['histgb_G1'] = G.histgb_check(hold, valid)
    print('HistGB G1 (pred + shape + class):', {v: round(x['all'], 5) for v, x in res['histgb_G1'].items()}, flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
