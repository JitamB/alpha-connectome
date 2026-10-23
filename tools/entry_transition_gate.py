"""Gate whether a class-3 target depends on the causal transition *into* its current i0 book state.

The static exact-shape table is closed.  Here an entry key is carried until the next i0
price-shape transition: (the state immediately before entry, current state).  Thus rows
inside the same current shape can receive different target statistics according to how that
shape was reached.  All lookup statistics come from train sequences outside the fixed
holdout; G1 and G2 fit only a residual ridge after the lookup has been frozen.
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
MIX_A, MIX_B = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xD6E8FEB86659FD93)


def entry_ids(X):
    """One causal key per row: (state immediately before the current state was entered, current state)."""
    sid = G.shape_ids(X[:, :22])
    prev = np.r_[sid[:1], sid[:-1]]
    changed = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(changed)
    # The first segment uses (initial state, initial state); thereafter origin is the preceding state.
    origin = np.repeat(prev[starts], np.diff(np.r_[starts, len(sid)]))
    with np.errstate(over='ignore'):
        return origin * MIX_A + sid * MIX_B


def learn_entry_table(groups, cls_of, every):
    keys_all, sums_all = [], []
    for s in iter_sequences(TRAIN, groups):
        step = STEPS[cls_of[s.group]]
        n = np.rint(s.y.astype(np.float64) / step).astype(np.int64)[::every]
        keys_all.append(entry_ids(s.X)[::every])
        sums_all.append(np.stack([n.sum(1), n[:, 0] - n[:, 1]], 1))
    key, val = np.concatenate(keys_all), np.concatenate(sums_all)
    keys, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    stats = np.column_stack([cnt, np.bincount(inv, weights=val[:, 0]), np.bincount(inv, weights=val[:, 1])])
    prior = stats[:, 1:].sum(0) / stats[:, 0].sum()
    return keys, stats, prior, {'rows': int(len(key)), 'keys': int(len(keys)), 'prior_S_D': prior.tolist()}


def entry_features(X, step, table):
    keys, stats, prior = table
    q = entry_ids(X)
    pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    hit = keys[pos] == q
    st = np.where(hit[:, None], stats[pos], 0.0)
    cnt = st[:, 0]
    mean = (st[:, 1:] + SHRINK * prior) / (cnt[:, None] + SHRINK)
    return np.column_stack([hit, np.log1p(cnt), mean * step]).astype(np.float32)


def build(path, groups, cls_of, table, models, device, batch=16):
    rows, chunk = [], []
    it = iter_sequences(path, groups)
    while True:
        s = next(it, None)
        if s is not None:
            chunk.append(s)
        if chunk and (s is None or len(chunk) == batch):
            for s_, p in zip(chunk, G.predict(models, chunk, device)):
                c = cls_of[s_.group]
                # The gate deliberately leaves non-c3 rows unchanged; their per-class ridge sees zeros.
                f = entry_features(s_.X, STEPS[c], table) if c == 3 else np.zeros((len(s_.X), 4), np.float32)
                rows.append({'y': np.clip(s_.y, -G.CLIP, G.CLIP).astype(np.float64), 'p': p, 'need': s_.need,
                             'scored': s_.scored, 'cls': c, 'group': s_.group, 'f': {'entry': f}})
            chunk = []
        if s is None:
            return rows


def report(res, key):
    for view in ('V1', 'V2'):
        g = res[key][view]
        base, new = g['none'], g['entry']
        print(f"{key} {view}: all {new['all'] - base['all']:+.5f}  c3 {new['c3'] - base['c3']:+.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--name', required=True)
    ap.add_argument('--n-table', type=int, default=4000)
    ap.add_argument('--every', type=int, default=4)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    groups = [g for g in train_groups()[:args.n_table] if tcls[g] == 3]
    keys, stats, prior, info = learn_entry_table(groups, tcls, args.every)
    table = (keys, stats, prior)
    print('entry table', info, flush=True)
    models = [load(Path(r), 'best.pt', device)[0] for r in args.runs]
    hold = build(TRAIN, holdout_groups(), tcls, table, models, device)
    for r in hold:
        r['scored'] = np.ones_like(r['need'])
    valid = build(VALID, list(range(1873)), vcls, table, models, device)
    for r in hold + valid:
        r['fitmask'] = r['need'] & r['scored']
    G.GROUPS = {'none': [], 'entry': ['entry']}
    res = {'table': info}
    res['G1_holdout_to_valid'] = G.gate(hold, valid, 'fitmask')
    A, B = [r for r in valid if r['group'] < 937], [r for r in valid if r['group'] >= 937]
    res['G2_A_to_B'] = G.gate(A, B, 'fitmask')
    res['G2_B_to_A'] = G.gate(B, A, 'fitmask')
    for key in ('G1_holdout_to_valid', 'G2_A_to_B', 'G2_B_to_A'):
        report(res, key)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
