"""Gate for decoded tick-path / event-clock features: do they add anything on top of the GRU ensemble's predictions?

  python tools/gate_paths.py runs/gru_all_full runs/gru_all_full_seed1 --name gate_paths [--n-table 3000]

1. Tables, learned on train sequences (holdout excluded). A book "shape" = the 22 price columns of an instrument
   (exact float32 bits, hashed); a transition = (previous shape, new shape).
   - i0 decode: transition type -> most common tick change (dn0, dn1) of the targets on that row.
   - Target encodings (i0 and i1): transition type -> shrunk mean target change (ticks) over the next H rows.
     i1 has no targets of its own, so this measures what i1's moves predict for i0 (lead-lag).
2. Causal features per row, all multiplied by the class step:
   - window sums (10 / 100 / 1000 rows) of decoded i0 moves and of both encodings;
   - log time since the last i0 / i1 price change and since the last i0 / i1 trade;
   - transition counts over 100 / 1000 rows.
   The class (and step) comes from splits/*_classes.json; the warm-up detector is 99.9% accurate.
3. Stacking with ridge, per class, |y|-weighted, compared with ridge on the predictions alone:
   G1 fits on the train holdout (800 seqs, all rows; the models never saw them) and tests on all of valid (V1, V2).
   G2 fits on valid half A (scored rows) and tests on half B, then the reverse.
   Feature groups: i0 (decode + i0 encoding), i1 (i1 encoding), clock (timers + counts), all.
   Cross-check: HistGB on [pred, all features, class] in G1.
Writes results/NAME.json. The predictions come from train-only models, so valid is clean.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from data import TRAIN, VALID, WARMUP, holdout_groups, iter_sequences, train_groups  # noqa: E402
from eval_torch import load  # noqa: E402
from evaluate import seq_moments, wp_from_moments  # noqa: E402
from gpu_features import torch_features  # noqa: E402
from train_gru import STEPS  # noqa: E402

CLIP = 2.0
PRICE = {'i0': slice(0, 22), 'i1': slice(52, 74)}
TRADE = {'i0': [44, 48], 'i1': [96, 100]}          # dp0, dv0: one trade price / volume number each
WINDOWS = (10, 100, 1000)
H = 100                                             # horizon of the target encodings
SHRINK = 20.0
MULT = np.random.default_rng(12345).integers(1, 2 ** 63, 22, dtype=np.uint64) | np.uint64(1)
PAIR = np.uint64(0x9E3779B97F4A7C15)
GROUPS = {'i0': ['i0dec', 'i0enc'], 'i1': ['i1enc'], 'clock': ['clock'], 'all': ['i0dec', 'i0enc', 'i1enc', 'clock']}


def shape_ids(P):
    u = np.ascontiguousarray(P, dtype=np.float32).view(np.uint32).astype(np.uint64)
    with np.errstate(over='ignore'):
        return (u * MULT).sum(1)


def transitions(P):
    ids = shape_ids(P)
    tr = np.r_[False, ids[1:] != ids[:-1]]
    typ = np.zeros(len(ids), np.uint64)
    with np.errstate(over='ignore'):
        typ[1:] = ids[:-1] * PAIR + ids[1:]
    return tr, typ


def ticks(y, step):
    return np.round(np.clip(y, -1e6, 1e6) / step).astype(np.int64)


def learn_tables(groups, cls_of):
    dec, enc = [], {'i0': [], 'i1': []}
    for s in iter_sequences(TRAIN, groups):
        n = ticks(s.y.astype(np.float64), STEPS[cls_of[s.group]])
        T = len(n)
        fut = n[np.minimum(np.arange(T) + H, T - 1)] - n
        for inst in ('i0', 'i1'):
            tr, typ = transitions(s.X[:, PRICE[inst]])
            idx = np.flatnonzero(tr)
            enc[inst].append((typ[idx], fut[idx]))
            if inst == 'i0':
                dec.append((typ[idx], n[idx] - n[idx - 1]))
    typ = np.concatenate([t for t, _ in dec])
    d = np.concatenate([x for _, x in dec])
    combo, counts = np.unique(np.stack([typ, (d[:, 0] + 10_000).astype(np.uint64), (d[:, 1] + 10_000).astype(np.uint64)], 1),
                              axis=0, return_counts=True)
    order = np.lexsort((counts, combo[:, 0]))                  # by type, then count: the last row of each type wins
    combo = combo[order]
    last = np.r_[combo[1:, 0] != combo[:-1, 0], True]
    best = combo[last]
    tables = {'i0dec': (best[:, 0], best[:, 1:].astype(np.float64) - 10_000)}
    for inst in ('i0', 'i1'):
        typ = np.concatenate([t for t, _ in enc[inst]])
        fut = np.concatenate([f for _, f in enc[inst]]).astype(np.float64)
        keys, inv, cnt = np.unique(typ, return_inverse=True, return_counts=True)
        sums = np.stack([np.bincount(inv, weights=fut[:, j], minlength=len(keys)) for j in range(2)], 1)
        tables[f'{inst}enc'] = (keys, sums / (cnt[:, None] + SHRINK))
    sizes = {k: int(len(v[0])) for k, v in tables.items()}
    return tables, sizes


def lookup(table, typ):
    keys, vals = table
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    pos = np.clip(np.searchsorted(keys, typ), 0, len(keys) - 1)
    hit = keys[pos] == typ
    return np.where(hit[:, None], vals[pos], 0.0), hit


def window_sums(v, w):
    c = np.cumsum(v, 0)
    out = c.copy()
    out[w:] -= c[:-w]
    return out


def since(flag):
    t = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, t, 0))
    return np.log1p(t - last)


def features(X, step, tables):
    """(T, 112) raw rows -> dict group -> (T, k) causal features (tick quantities in target units)."""
    f, cover = {}, {}
    tr0, typ0 = transitions(X[:, PRICE['i0']])
    tr1, typ1 = transitions(X[:, PRICE['i1']])
    for name, tr, typ in (('i0dec', tr0, typ0), ('i0enc', tr0, typ0), ('i1enc', tr1, typ1)):
        v, hit = lookup(tables[name], typ)
        v = v * tr[:, None] * step
        cover[name] = (int((hit & tr).sum()), int(tr.sum()))
        f[name] = np.concatenate([window_sums(v, w) for w in WINDOWS], 1)
    clock = [since(tr0), since(tr1)]
    for inst in ('i0', 'i1'):
        cols = X[:, TRADE[inst]]
        clock.append(since(np.r_[False, (cols[1:] != cols[:-1]).any(1)]))
    for tr in (tr0, tr1):
        clock += [window_sums(tr.astype(np.float64), w) / w for w in (100, 1000)]
    f['clock'] = np.stack(clock, 1)
    return {k: v.astype(np.float32) for k, v in f.items()}, cover


@torch.no_grad()
def predict(models, chunk, device):
    F = torch_features(torch.from_numpy(np.stack([s.X for s in chunk])).to(device))
    return torch.stack([m(F, None)[0] for m in models]).mean(0).float().cpu().numpy()


def build(path, groups, cls_of, tables, models, device, batch=16):
    """-> list of per-sequence dicts with y, pred, need, scored, cls, features (raw rows are not kept)."""
    rows, cover, chunk = [], {}, []
    it = iter_sequences(path, groups)
    while True:
        s = next(it, None)
        if s is not None:
            chunk.append(s)
        if chunk and (s is None or len(chunk) == batch):
            for s_, q in zip(chunk, predict(models, chunk, device)):
                c = cls_of[s_.group]
                f, cv = features(s_.X, STEPS[c], tables)
                for k, (h, t) in cv.items():
                    a = cover.setdefault(k, [0, 0]); a[0] += h; a[1] += t
                rows.append({'y': np.clip(s_.y, -CLIP, CLIP).astype(np.float64), 'p': q, 'need': s_.need,
                             'scored': s_.scored, 'cls': c, 'group': s_.group, 'f': f})
            chunk = []
        if s is None:
            return rows, {k: v[0] / max(v[1], 1) for k, v in cover.items()}


def design(r, group):
    cols = [r['p']] + [r['f'][g] for g in GROUPS.get(group, [])]
    return np.concatenate(cols, 1).astype(np.float64)


def fit_ridge(rows, group, mask_key, lam=1e-3):
    """Per-class |y|-weighted ridge per target on rows[...][mask]; returns {cls: (mu, sd, W (k+1, 2))}."""
    models = {}
    for c in range(4):
        A = b = None
        sub = [r for r in rows if r['cls'] == c]
        if not sub:
            continue
        Z = np.concatenate([design(r, group)[r[mask_key]] for r in sub])
        Y = np.concatenate([r['y'][r[mask_key]] for r in sub])
        mu, sd = Z.mean(0), Z.std(0) + 1e-9
        Z = np.concatenate([(Z - mu) / sd, np.ones((len(Z), 1))], 1)
        W = np.zeros((Z.shape[1], 2))
        for j in range(2):
            w = np.abs(Y[:, j])
            A = Z.T @ (Z * w[:, None]) + lam * w.sum() * np.eye(Z.shape[1])
            W[:, j] = np.linalg.solve(A, Z.T @ (w * Y[:, j]))
        models[c] = (mu, sd, W)
    return models


def apply_ridge(models, r, group):
    if r['cls'] not in models:  # class absent from the fit set (tiny smoke runs only)
        return r['p']
    mu, sd, W = models[r['cls']]
    Z = (design(r, group) - mu) / sd
    return Z @ W[:-1] + W[-1]


def score(rows, pred_fn, view):
    """Global WP over rows (V1: need & scored, V2: need), overall and per class."""
    m = {c: [] for c in range(4)}
    for r in rows:
        mask = r['need'] & r['scored'] if view == 'V1' else r['need']
        m[r['cls']].append(seq_moments(r['y'], pred_fn(r), mask))
    per = {c: float(wp_from_moments(np.stack(v).sum(0)).mean()) for c, v in m.items() if v}
    allm = np.stack([x for v in m.values() for x in v]).sum(0)
    return {'all': float(wp_from_moments(allm).mean()), **{f'c{c}': v for c, v in per.items()}}


def gate(fit_rows, test_rows, fit_mask):
    out = {}
    for view in ('V1', 'V2'):
        out[view] = {'pred': score(test_rows, lambda r: r['p'], view)}
    for group in ['none'] + list(GROUPS):
        models = fit_ridge(fit_rows, group, fit_mask)
        for view in ('V1', 'V2'):
            out[view][group] = score(test_rows, lambda r: apply_ridge(models, r, group), view)
    return out


def histgb_check(fit_rows, test_rows, n_fit=1_500_000, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)

    def mat(r):
        return np.concatenate([design(r, 'all'), np.full((len(r['y']), 1), r['cls'])], 1)
    Z = np.concatenate([mat(r)[r['need']] for r in fit_rows])
    Y = np.concatenate([r['y'][r['need']] for r in fit_rows])
    keep = rng.choice(len(Z), min(n_fit, len(Z)), replace=False)
    out = {}
    hgb = []
    for j in range(2):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                          l2_regularization=1.0, categorical_features=[Z.shape[1] - 1], random_state=seed)
        m.fit(Z[keep], Y[keep, j], sample_weight=np.abs(Y[keep, j]) + 1e-3)
        hgb.append(m)
    cache = {}

    def pred(r):
        if r['group'] not in cache:
            Zr = mat(r)
            cache[r['group']] = np.stack([m.predict(Zr) for m in hgb], 1)
        return cache[r['group']]
    for view in ('V1', 'V2'):
        out[view] = score(test_rows, pred, view)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--name', required=True)
    ap.add_argument('--n-table', type=int, default=3000)
    ap.add_argument('--n-hold', type=int, default=800)
    ap.add_argument('--n-valid', type=int, default=1873)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    res = {}
    tables, res['table_sizes'] = learn_tables(train_groups()[:args.n_table], tcls)
    print('table sizes', res['table_sizes'], flush=True)
    models = [load(Path(r), 'best.pt', device)[0] for r in args.runs]
    hold, res['coverage_holdout'] = build(TRAIN, holdout_groups()[:args.n_hold], tcls, tables, models, device)
    for r in hold:
        r['scored'] = np.ones_like(r['need'])
    valid, res['coverage_valid'] = build(VALID, list(range(args.n_valid)), vcls, tables, models, device)
    print('coverage holdout', res['coverage_holdout'], 'valid', res['coverage_valid'], flush=True)
    for r in hold + valid:
        r['fitmask'] = r['need'] & r['scored']
    res['G1_holdout_to_valid'] = gate(hold, valid, 'fitmask')
    A = [r for r in valid if r['group'] < 937]
    B = [r for r in valid if r['group'] >= 937]
    if A and B:
        res['G2_A_to_B'] = gate(A, B, 'fitmask')
        res['G2_B_to_A'] = gate(B, A, 'fitmask')
    for k in [k for k in ('G1_holdout_to_valid', 'G2_A_to_B', 'G2_B_to_A') if k in res]:
        for view in ('V1', 'V2'):
            g = res[k][view]
            base = g['none']['all']
            print(f"{k} {view}: pred {g['pred']['all']:.5f} ridge[pred] {base:.5f} | " +
                  ' '.join(f"+{grp} {g[grp]['all'] - base:+.5f}" for grp in GROUPS) +
                  f" | per class (all - none): " +
                  ' '.join(f"c{c} {g['all'][f'c{c}'] - g['none'][f'c{c}']:+.4f}" for c in range(4) if f'c{c}' in g['all']),
                  flush=True)
    res['histgb_G1'] = histgb_check(hold, valid)
    print('HistGB G1 (pred + all + class):', {v: round(x['all'], 5) for v, x in res['histgb_G1'].items()}, flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
