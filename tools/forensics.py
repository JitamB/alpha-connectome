"""A1 forensic probes on the raw data (CPU only): target anatomy, clock test, regime persistence, a0-a7 semantics and
the scoring-mask mechanism. Analysis only; writes results/<name>.json and prints a summary.

  python tools/forensics.py --name forensics_a1 [--per-class 60] [--mask-seqs 120]

An i0 "shape" is the exact float32 bits of the 22 i0 price-like columns (hashed, as in tools/gate_paths.py); a
transition is a row whose shape differs from the previous row; a segment is a maximal run of rows without a
transition. Targets are n * step with integer n (step per asset class); S = n0 + n1, D = n0 - n1.
Hypotheses under test (experiments.md, 2026-10-22 review):
  H1  t0 = F - ask_now, t1 = bid_now - F       -> S = -spread_now: a function of the current shape
  H3  t0 = max future bid - ask_now, t1 = bid_now - min future ask  -> n >= -spread (floor), S grows with activity
  clock B: label window [segment start, + H rows] -> segments longer than H carry the "no move" label exactly
  clock C: label window = next K i0 events        -> no such length threshold
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from data import TRAIN, VALID, iter_sequences, train_groups  # noqa: E402

STEPS = (0.12543297, 0.24132967, 0.4685359, 0.65686762)
MULT = np.random.default_rng(12345).integers(1, 2 ** 63, 22, dtype=np.uint64) | np.uint64(1)
P0, P1 = slice(0, 22), slice(52, 74)            # i0 / i1 price-like columns
TRADE0, TRADE1 = [44, 48], [96, 100]            # dp0, dv0 of each instrument
A_COLS = list(range(104, 112))
L_EDGES = [1, 3, 10, 30, 100, 300, 1000, 3000, 10 ** 9]
F_EDGES = [0, 1, 2, 4, 8, 16, 10 ** 9]          # buckets of transition counts in a 300-row window


def shape_ids(P):
    u = np.ascontiguousarray(P, dtype=np.float32).view(np.uint32).astype(np.uint64)
    with np.errstate(over='ignore'):
        return (u * MULT).sum(1)


def flags(X, cols):
    f = np.zeros(len(X), bool)
    f[1:] = (X[1:, cols] != X[:-1, cols]).any(1)
    return f


def since(flag):
    t = np.arange(len(flag))
    return t - np.maximum.accumulate(np.where(flag, t, 0))


def window_count(flag, h, future=True):
    """number of flagged rows in (t, t+h] (future) or (t-h, t] (past); -1 where the window leaves the sequence"""
    c = np.concatenate([[0], np.cumsum(flag)])
    T = len(flag)
    t = np.arange(T)
    if future:
        out = c[np.minimum(t + h + 1, T)] - c[t + 1]
        out[t + h > T - 1] = -1
    else:
        out = c[t + 1] - c[np.maximum(t + 1 - h, 0)]
        out[t - h + 1 < 0] = -1
    return out


def bucket_stats(keys, edges, vals):
    """mean and count of vals per bucket of keys (edges: left-closed bucket bounds)"""
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (keys >= lo) & (keys < hi)
        if m.any():
            out[f'{lo}-{hi - 1 if hi < 10 ** 9 else "inf"}'] = [int(m.sum()), float(vals[m].mean())]
    return out


def corr(a, b, w=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    if len(a) < 3 or w.sum() <= 0:
        return None
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    va, vb = np.average((a - ma) ** 2, weights=w), np.average((b - mb) ** 2, weights=w)
    return float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(va * vb)) if va > 0 and vb > 0 else None


def analyze(X, y, step):
    """Per-sequence quantities (all numpy, small)."""
    n = np.rint(y.astype(np.float64) / step).astype(np.int64)
    integ = float(np.abs(y.astype(np.float64) / step - n).max())
    sid = shape_ids(X[:, P0])
    tr = np.r_[False, sid[1:] != sid[:-1]]
    changed = np.r_[False, (n[1:] != n[:-1]).any(1)]
    starts = np.r_[0, np.flatnonzero(tr)]
    lens = np.diff(np.r_[starts, len(X)])
    lab = n[starts]                                  # label of each segment (targets are constant inside, checked)
    _, local = np.unique(sid, return_inverse=True)
    a6 = X[:, 110]
    return {
        'n': n, 'integ': integ, 'sid': sid, 'tr': tr, 'local': local, 'starts': starts, 'lens': lens, 'lab': lab,
        'seg_sid': sid[starts], 'seg_local': local[starts],
        'a6_end': a6[np.r_[starts[1:] - 1, len(X) - 1]], 'a6_start': a6[starts],
        'chg_no_tr': int((changed & ~tr).sum()), 'chg': int(changed.sum()), 'n_tr': int(tr.sum()),
        'chg_at_tr': int((changed & tr).sum()),
        'fut300': window_count(tr, 300), 'past300': window_count(tr, 300, future=False),
    }


def anatomy(rows):
    """rows: list of (cls, seq dict) for one class -> anatomy + clock + persistence summary."""
    out = {}
    S_all = np.concatenate([r['n'][99:].sum(1) for r in rows])
    n0 = np.concatenate([r['n'][99:, 0] for r in rows])
    n1 = np.concatenate([r['n'][99:, 1] for r in rows])
    vals, cnt = np.unique(S_all, return_counts=True)
    top = np.argsort(-cnt)[:10]
    out['S_counts'] = {int(vals[i]): round(float(cnt[i] / cnt.sum()), 4) for i in top}
    out['P_S_pos'], out['P_S_zero'] = float((S_all > 0).mean()), float((S_all == 0).mean())
    out['n0_q'] = [int(v) for v in np.quantile(n0, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])]
    out['n1_q'] = [int(v) for v in np.quantile(n1, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])]
    out['corr_n0_n1'] = corr(n0, n1)
    out['target_change_without_transition'] = [sum(r['chg_no_tr'] for r in rows), sum(r['chg'] for r in rows)]
    out['transitions_changing_target'] = [sum(r['chg_at_tr'] for r in rows), sum(r['n_tr'] for r in rows)]
    out['integrality_max_err'] = max(r['integ'] for r in rows)
    out['segments_per_seq'] = float(np.mean([len(r['starts']) for r in rows]))
    out['distinct_shapes_per_seq'] = float(np.mean([r['local'].max() + 1 for r in rows]))
    # inner segments only (the first and last are cut by the sequence ends)
    seg = [(r, slice(1, len(r['starts']) - 1)) for r in rows if len(r['starts']) > 2]
    L = np.concatenate([r['lens'][s] for r, s in seg])
    lab = np.concatenate([r['lab'][s] for r, s in seg])
    S = lab.sum(1)
    out['segment_len_q'] = [int(v) for v in np.quantile(L, [0.1, 0.5, 0.9, 0.99])] if len(L) else None

    # S purity given the (sequence, shape): share of segments whose S equals the group's modal S (groups >= 3 segments)
    pur_num = pur_den = 0
    eq_min = eq_den = 0
    q_flags, q_L, q_a6 = [], [], []
    for r, s in seg:
        ls, lb, ln, a6e = r['seg_local'][s], r['lab'][s], r['lens'][s], r['a6_end'][s]
        Sg = lb.sum(1)
        for g in np.unique(ls):
            m = ls == g
            if m.sum() < 3:
                continue
            v, c = np.unique(Sg[m], return_counts=True)
            pur_num += c.max()
            pur_den += m.sum()
            eq_min += int(lb[m, 0].min() == lb[m, 1].min())
            eq_den += 1
            q_flags.append(Sg[m] == Sg[m].min())          # "no move" candidate: the group's lowest S (H3: U = L = 0)
            q_L.append(ln[m])
            q_a6.append(a6e[m])
    out['S_purity_seq_shape'] = [float(pur_num / max(pur_den, 1)), int(pur_den)]
    out['floor_min_n0_eq_min_n1'] = [float(eq_min / max(eq_den, 1)), int(eq_den)]
    if q_flags:
        q, qL, qa = np.concatenate(q_flags), np.concatenate(q_L), np.concatenate(q_a6)
        out['clock_P_lowestS_by_len'] = bucket_stats(qL, L_EDGES, q.astype(float))
        qs = np.quantile(qa, np.linspace(0, 1, 9))
        out['clock_P_lowestS_by_a6end'] = {f'{qs[i]:.3f}..{qs[i + 1]:.3f}': [int(m.sum()), float(q[m].mean())]
                                           for i in range(8) if (m := (qa >= qs[i]) & (qa <= qs[i + 1])).any()}
    # S across the length of a segment: does the label depend on how long the segment lasts?
    if len(L):
        out['S_by_len'] = bucket_stats(L, L_EDGES, S.astype(float))
    # S against activity in the next / last 300 rows (row level, need rows)
    fut = np.concatenate([r['fut300'][99:] for r in rows])
    past = np.concatenate([r['past300'][99:] for r in rows])
    ok = fut >= 0
    out['S_by_future_tr300'] = bucket_stats(fut[ok], F_EDGES, S_all[ok].astype(float))
    out['S_by_past_tr300'] = bucket_stats(past[past >= 0], F_EDGES, S_all[past >= 0].astype(float))
    out['corr_S_future_tr300'] = corr(S_all[ok], fut[ok])
    out['corr_S_past_tr300'] = corr(S_all[past >= 0], past[past >= 0])
    # label autocorrelation in event lags (segments) vs row lags
    ev = {}
    for k in (1, 2, 3, 5, 10, 20):
        a = np.concatenate([r['lab'][1:-1 - k, 0] for r in rows if len(r['starts']) > k + 2] or [np.zeros(0)])
        b = np.concatenate([r['lab'][1 + k:-1, 0] for r in rows if len(r['starts']) > k + 2] or [np.zeros(0)])
        ev[k] = corr(a, b)
    out['autocorr_n0_event_lag'] = ev
    out['autocorr_n0_row_lag'] = {k: corr(np.concatenate([r['n'][99:-k, 0] for r in rows]),
                                          np.concatenate([r['n'][99 + k:, 0] for r in rows])) for k in (10, 100, 300, 1000, 3000)}
    # regime persistence: first vs second half, |y|-weighted mean of y per sequence
    halves = []
    for r in rows:
        yy = r['n'] * r['step']
        yc = np.clip(yy, -2, 2)
        h = []
        for lo, hi in ((99, 10_000), (10_000, 20_000)):
            w = np.abs(yc[lo:hi])
            h.append([float((w[:, j] * yc[lo:hi, j]).sum() / max(w[:, j].sum(), 1e-9)) for j in range(2)]
                     + [float(w.mean()), float(yy[lo:hi].sum(1).mean())])
        halves.append(h)
    hv = np.array(halves)
    out['persistence_corr_half1_half2'] = {k: corr(hv[:, 0, i], hv[:, 1, i]) for i, k in
                                           enumerate(['wmean_y0', 'wmean_y1', 'mean_abs_y', 'mean_S'])}
    return out


def global_shape_tests(rows, seed=0):
    """Across sequences: S purity / floor of the global shape, fitted on half of the sequences, tested on the rest."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    fit, test = [rows[i] for i in idx[: len(rows) // 2]], [rows[i] for i in idx[len(rows) // 2:]]

    def seg_table(rs):
        sid = np.concatenate([r['seg_sid'] for r in rs])
        lab = np.concatenate([r['lab'] for r in rs])
        return sid, lab
    sid, lab = seg_table(fit)
    S = lab.sum(1)
    order = np.lexsort((S, sid))
    sid_o, S_o = sid[order], S[order]
    keys, first = np.unique(sid_o, return_index=True)
    # modal S per shape
    modal, mins0, mins1 = {}, {}, {}
    for k, a, b in zip(keys, first, np.r_[first[1:], len(sid_o)]):
        v, c = np.unique(S_o[a:b], return_counts=True)
        modal[k] = v[c.argmax()]
    sidt, labt = seg_table(test)
    hit = np.array([s in modal for s in sidt])
    acc = np.array([modal.get(s, 10 ** 6) == v for s, v in zip(sidt, labt.sum(1))])
    # floor: min n0 / n1 per shape on the fit half; violation rate on the test half
    for k_, n0_, n1_ in zip(sid, lab[:, 0], lab[:, 1]):
        mins0[k_] = min(mins0.get(k_, 10 ** 6), n0_)
        mins1[k_] = min(mins1.get(k_, 10 ** 6), n1_)
    viol0 = np.array([n < mins0[s] for s, n in zip(sidt, labt[:, 0]) if s in mins0])
    viol1 = np.array([n < mins1[s] for s, n in zip(sidt, labt[:, 1]) if s in mins1])
    eq = np.mean([mins0[k_] == mins1[k_] for k_ in mins0]) if mins0 else None
    return {'coverage': float(hit.mean()) if len(hit) else None,
            'S_modal_acc_on_hits': float(acc[hit].mean()) if hit.any() else None,
            'floor_violation_n0': float(viol0.mean()) if len(viol0) else None,
            'floor_violation_n1': float(viol1.mean()) if len(viol1) else None,
            'shapes_min_n0_eq_min_n1': float(eq) if eq is not None else None,
            'min_n0_counts': {int(v): int(c) for v, c in zip(*np.unique(list(mins0.values()), return_counts=True))}
            if mins0 else None}


def a_semantics(seqs):
    """a0..a7: distinct values, floor share, change rate, relation to event timers (Spearman on every 10th row)."""
    from scipy.stats import spearmanr
    cols = {f'a{j}': [] for j in range(8)}
    timers = {k: [] for k in ('since_i0_tr', 'since_i1_tr', 'since_i0_trade', 'since_i1_trade', 'step')}
    dec_at_tr, dec_tot, chg = np.zeros(8), np.zeros(8), np.zeros(8)
    for X in seqs:
        f = {'since_i0_tr': since(np.r_[False, shape_ids(X[1:, P0]) != shape_ids(X[:-1, P0])]),
             'since_i1_tr': since(np.r_[False, shape_ids(X[1:, P1]) != shape_ids(X[:-1, P1])]),
             'since_i0_trade': since(flags(X, TRADE0)), 'since_i1_trade': since(flags(X, TRADE1)),
             'step': np.arange(len(X))}
        tr0 = np.r_[False, (X[1:, P0] != X[:-1, P0]).any(1)]
        for k in timers:
            timers[k].append(f[k][::10])
        for j, c in enumerate(A_COLS):
            cols[f'a{j}'].append(X[::10, c])
            d = np.diff(X[:, c])
            dec = np.r_[False, d < 0]
            dec_tot[j] += dec.sum()
            dec_at_tr[j] += (dec & tr0).sum()
            chg[j] += (d != 0).sum()
    T = sum(len(X) for X in seqs)
    out = {}
    tm = {k: np.concatenate(v) for k, v in timers.items()}
    for j in range(8):
        a = np.concatenate(cols[f'a{j}'])
        out[f'a{j}'] = {
            'distinct': int(len(np.unique(a))), 'share_at_min': float((a == a.min()).mean()),
            'change_rate': float(chg[j] / T), 'P_i0_transition_given_decrease': float(dec_at_tr[j] / max(dec_tot[j], 1)),
            'decrease_rate': float(dec_tot[j] / T),
            'spearman': {k: round(float(spearmanr(a, v).correlation), 3) for k, v in tm.items()},
        }
    return out


def mask_probe(seqs_a, seqs_b, seed=0):
    """AUC of is_scored from causal row features vs causal + next-2-row features (HistGB; fit half A, test half B)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from features import batch_features
    rng = np.random.default_rng(seed)

    def rows(seqs):
        Fc, Ff, y, w, cond = [], [], [], [], []
        for s in seqs:
            X = s.X
            F = batch_features(X)[:, :336]                       # raw, diff, x - EMA10 (causal)
            nxt = np.zeros((len(X), 224), np.float32)
            nxt[:-1, :112] = X[1:] - X[:-1]
            nxt[:-2, 112:] = X[2:] - X[1:-1]
            tr0 = np.r_[False, shape_ids(X[1:, P0]) != shape_ids(X[:-1, P0])]
            tr1 = np.r_[False, shape_ids(X[1:, P1]) != shape_ids(X[:-1, P1])]
            fl = np.stack([tr0, tr1, flags(X, TRADE0), flags(X, TRADE1)], 1).astype(np.float32)
            fl_next = np.zeros_like(fl)
            fl_next[:-1] = fl[1:]
            idx = np.arange(99, len(X) - 2)
            m = s.scored[idx]
            sel = idx[m | (rng.random(len(idx)) < 0.1)]
            Fc.append(np.hstack([F[sel], fl[sel]]))
            Ff.append(np.hstack([nxt[sel], fl_next[sel]]))
            y.append(s.scored[sel])
            w.append(np.where(s.scored[sel], 1.0, 10.0))
            cond.append(np.hstack([fl[idx], fl_next[idx], s.scored[idx, None]]))
        return (np.concatenate(Fc), np.concatenate(Ff), np.concatenate(y).astype(int), np.concatenate(w),
                np.concatenate(cond))
    Fa, Na, ya, wa, ca = rows(seqs_a)
    Fb, Nb, yb, wb, cb = rows(seqs_b)
    out = {'rows_fit': int(len(ya)), 'rows_test': int(len(yb))}
    for name, (A, B) in {'causal': (Fa, Fb), 'causal_plus_next2': (np.hstack([Fa, Na]), np.hstack([Fb, Nb]))}.items():
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, random_state=seed)
        clf.fit(A, ya, sample_weight=wa)
        out[f'auc_{name}'] = float(roc_auc_score(yb, clf.predict_proba(B)[:, 1], sample_weight=wb))
    c = np.concatenate([ca, cb])
    names = ['i0_tr_t', 'i1_tr_t', 'i0_trade_t', 'i1_trade_t', 'i0_tr_t+1', 'i1_tr_t+1', 'i0_trade_t+1', 'i1_trade_t+1']
    out['P_scored'] = float(c[:, -1].mean())
    out['P_scored_given'] = {k: [float(c[c[:, i] > 0, -1].mean()) if (c[:, i] > 0).any() else None,
                                 float(c[c[:, i] == 0, -1].mean())] for i, k in enumerate(names)}
    return out


def examples(rows, k_seg=60):
    """Readable segment lists and distinct-shape tables for a few sequences."""
    ex = []
    for cls, r, X in rows:
        segs = [[int(a), int(b), int(c), int(d[0]), int(d[1]), round(float(e), 3), round(float(f), 3)]
                for a, b, c, d, e, f in zip(r['starts'][:k_seg], r['lens'][:k_seg], r['seg_local'][:k_seg],
                                            r['lab'][:k_seg], r['a6_start'][:k_seg], r['a6_end'][:k_seg])]
        loc, first, cnt = np.unique(r['local'], return_index=True, return_counts=True)
        top = np.argsort(-cnt)[:12]
        shapes = {int(loc[i]): {'rows': int(cnt[i]), 'bid': np.round(X[first[i], 0:11], 4).tolist(),
                                'ask': np.round(X[first[i], 11:22], 4).tolist()} for i in top}
        ex.append({'class': cls, 'step': r['step'], 'segments[start,len,shape,n0,n1,a6_start,a6_end]': segs,
                   'shapes': shapes})
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--per-class', type=int, default=60)
    ap.add_argument('--mask-seqs', type=int, default=120, help='valid sequences per half for the mask probe')
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    groups = train_groups()
    pick = []
    for c in range(4):
        gc = [g for g in groups if tcls[g] == c]
        pick += sorted(rng.choice(gc, min(args.per_class, len(gc)), replace=False).tolist())
    res = {'per_class': {}, 'n_train_seqs': len(pick)}
    by_cls, raw_a, ex_rows = {c: [] for c in range(4)}, [], []
    for s in iter_sequences(TRAIN, sorted(pick)):
        c = tcls[s.group]
        r = analyze(s.X, s.y, STEPS[c])
        r['step'] = STEPS[c]
        by_cls[c].append(r)
        if len(raw_a) < 40 and len(by_cls[c]) <= 10:
            raw_a.append(s.X)
        if (c == 3 and sum(e[0] == 3 for e in ex_rows) < 3) or (c == 0 and sum(e[0] == 0 for e in ex_rows) < 1):
            ex_rows.append((c, r, s.X))
    for c in range(4):
        res['per_class'][c] = anatomy(by_cls[c])
        res['per_class'][c]['global_shape'] = global_shape_tests(by_cls[c])
        print(f'class {c} done', flush=True)
    res['a_features'] = a_semantics(raw_a)
    res['examples'] = examples(ex_rows)
    del by_cls
    va = sorted(rng.choice(937, args.mask_seqs, replace=False).tolist())
    vb = sorted((937 + rng.choice(1873 - 937, args.mask_seqs, replace=False)).tolist())
    res['mask'] = mask_probe(iter_sequences(VALID, va), iter_sequences(VALID, vb))
    res['mask']['class_mix_test'] = np.bincount([vcls[g] for g in vb], minlength=4).tolist()
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=1, default=str))
    for c in range(4):
        p = res['per_class'][c]
        print(f"== class {c}: S {p['S_counts']} P(S>0) {p['P_S_pos']:.3f} | S purity (seq,shape) {p['S_purity_seq_shape']} "
              f"| floor min n0 == min n1 {p['floor_min_n0_eq_min_n1']} | global {p['global_shape']}")
        print(f"   clock P(lowest S | segment length): {p.get('clock_P_lowestS_by_len')}")
        print(f"   S by future transitions (300): {p['S_by_future_tr300']} | by past: {p['S_by_past_tr300']}")
        print(f"   persistence: {p['persistence_corr_half1_half2']}")
    print('mask:', {k: v for k, v in res['mask'].items() if k.startswith('auc')}, flush=True)


if __name__ == '__main__':
    main()
