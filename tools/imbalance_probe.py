"""Queue-imbalance probe for the class-3 direction part (screen + transition lift + stacking gate), Kaggle only.

  python tools/imbalance_probe.py runs/lr7e4_full runs/lr7e4_full_seed1 --name imbalance_probe

The docs say column indices are identifiers, not depth levels, so we do not know which volume column is the best-level
queue. The probe is therefore role-agnostic:
  1. Residual screen. Candidates: 44 raw volume columns, 44 five-row volume changes, all 11x11 bid-minus-ask volume pairs
     for i0 and for i1, and 36 permutation-invariant side aggregates (mean/max/min/std, 5-row change, deviation from
     EMA10; bid, ask and bid-minus-ask, both instruments). Target: the |y|-weighted correlation with the pair's residual,
     direction d = ((y0-p0)-(y1-p1))/2 and cost s = ((y0-p0)+(y1-p1))/2 (base = mean of the given runs). Null: the same
     statistic with (y, p, mask) rolled in time by a random offset inside each sequence.
  2. Transition lift: P(i0 price shape changes within the next 5 rows | z > +1.28 / z < -1.28) per volume column.
  3. Cross-column structure: within-sequence correlation of the 22 i0 volume and 22 i0 price columns, per class.
  4. Stacking gate (G1 holdout -> valid, G2 both valid halves) and HistGB on [pred, aggregates], per tools/shape_gate.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'src'), str(ROOT / 'train')]
import gate_paths as G  # noqa: E402
from data import TRAIN, VALID, holdout_groups, iter_sequences  # noqa: E402
from eval_torch import load  # noqa: E402

VOL = {'i0': (slice(22, 33), slice(33, 44)), 'i1': (slice(74, 85), slice(85, 96))}   # (bid, ask) volume-like columns
N_NULL, MIN_SHIFT, HORIZON, Z_HI = 30, 3000, 5, 1.2816
ALPHA = 2 / 11


def side_agg(V):
    m = V.mean(1)
    ema = lfilter([ALPHA], [1, -(1 - ALPHA)], m, zi=[(1 - ALPHA) * m[0]])[0]
    d5 = m - np.r_[np.full(5, m[0]), m[:-5]]
    return np.stack([m, V.max(1), V.min(1), V.std(1), d5, m - ema], 1)


def candidates(X):
    """(T, K) float32 candidate matrix and the matching names; the last 36 columns are the aggregates."""
    cols, names = [], []
    raw = np.concatenate([X[:, VOL[i][s]] for i in ('i0', 'i1') for s in (0, 1)], 1)       # 44 volumes
    rn = [f'{i} v{j if s == 0 else 11 + j}' for i in ('i0', 'i1') for s in (0, 1) for j in range(11)]
    cols += [raw, raw - np.r_[np.repeat(raw[:1], 5, 0), raw[:-5]]]
    names += rn + ['d5 ' + n for n in rn]
    for i in ('i0', 'i1'):
        b, a = X[:, VOL[i][0]], X[:, VOL[i][1]]
        cols.append((b[:, :, None] - a[:, None, :]).reshape(len(X), 121))
        names += [f'{i} v{j}-v{11 + k}' for j in range(11) for k in range(11)]
    agg = []
    for i in ('i0', 'i1'):
        sb, sa = side_agg(X[:, VOL[i][0]]), side_agg(X[:, VOL[i][1]])
        agg += [sb, sa, sb - sa]
    cols.append(np.concatenate(agg, 1))
    stats = ('mean', 'max', 'min', 'std', 'd5', 'dEMA')
    names += [f'{i} {w} {s}' for i in ('i0', 'i1') for w in ('bid', 'ask', 'bid-ask') for s in stats]
    return np.concatenate(cols, 1).astype(np.float32), names


class Acc:
    """Moments for the residual screen (real + N_NULL rolled copies), transition lift, and column structure."""

    def __init__(self, K):
        z = lambda *s: np.zeros(s)
        self.n = z(N_NULL + 1); self.Swf = z(K, N_NULL + 1); self.Swff = z(K, N_NULL + 1)
        self.Swr = z(2, N_NULL + 1); self.Swrr = z(2, N_NULL + 1); self.Swfr = z(2, K, N_NULL + 1)
        self.lift = z(4, 44)                                  # n_hi, ev_hi, n_lo, ev_lo
        self.cv = z(22, 22); self.cp = z(22, 22); self.nr = 0.0

    def add(self, C, r, w, m, tr, X, rng):
        T = len(C)
        shifts = np.r_[0, rng.integers(MIN_SHIFT, T - MIN_SHIFT, N_NULL)]
        ws = np.stack([np.roll(w * m, s) for s in shifts], 1)                         # (T, S)
        rs = [np.stack([np.roll(r[:, c], s) for s in shifts], 1) for c in (0, 1)]
        F = C
        self.n += ws.sum(0)
        self.Swf += (F.T @ ws.astype(np.float32)).astype(np.float64)
        self.Swff += ((F * F).T @ ws.astype(np.float32)).astype(np.float64)
        for c in (0, 1):
            wr = ws * rs[c]
            self.Swr[c] += wr.sum(0); self.Swrr[c] += (wr * rs[c]).sum(0)
            self.Swfr[c] += (F.T @ wr.astype(np.float32)).astype(np.float64)
        fut = self._future(tr)
        z = C[:, :44]; need = m > 0
        hi, lo = (z > Z_HI) & need[:, None], (z < -Z_HI) & need[:, None]
        self.lift += np.stack([hi.sum(0), hi[fut].sum(0), lo.sum(0), lo[fut].sum(0)])
        S = slice(None, None, 10)
        for A, idx in ((self.cv, slice(22, 44)), (self.cp, slice(0, 22))):
            Y = X[S, idx].astype(np.float64)
            Y = Y - Y.mean(0)
            A += Y.T @ Y
        self.nr += len(X[S])

    @staticmethod
    def _future(tr):
        c = np.r_[0, np.cumsum(tr)]                                  # transitions in rows (t, t+HORIZON]
        T = len(tr)
        hi = np.minimum(np.arange(T) + 1 + HORIZON, T)
        return (c[hi] - c[np.minimum(np.arange(T) + 1, T)]) > 0

    def corr(self):
        """-> (2, K, S) weighted correlation of each candidate with each residual channel, real = [..., 0]."""
        mf = self.Swf / self.n
        vf = self.Swff / self.n - mf ** 2
        out = []
        for c in (0, 1):
            mr = self.Swr[c] / self.n
            vr = self.Swrr[c] / self.n - mr ** 2
            cov = self.Swfr[c] / self.n - mf * mr
            out.append(cov / np.sqrt(np.maximum(vf * vr, 1e-18)))
        return np.stack(out)


def run(path, groups, cls_of, models, device, names, scored_view, rng, batch=16):
    K = len(names)
    acc = {c: Acc(K) for c in range(4)}
    rows, chunk = [], []
    it = iter_sequences(path, groups)
    while True:
        s = next(it, None)
        if s is not None:
            chunk.append(s)
        if chunk and (s is None or len(chunk) == batch):
            for s_, q in zip(chunk, G.predict(models, chunk, device)):
                c = cls_of[s_.group]
                C, _ = candidates(s_.X)
                y = np.clip(s_.y, -G.CLIP, G.CLIP).astype(np.float64)
                p = np.clip(q, -G.CLIP, G.CLIP).astype(np.float64)
                m = (s_.need & s_.scored) if scored_view else s_.need
                res = y - p
                r = np.stack([(res[:, 0] - res[:, 1]) / 2, (res[:, 0] + res[:, 1]) / 2], 1)
                tr, _ = G.transitions(s_.X[:, 0:22])
                acc[c].add(C, r, np.abs(y).sum(1) / 2, m.astype(np.float64), tr, s_.X, rng)
                rows.append({'y': y, 'p': q, 'need': s_.need, 'scored': s_.scored if scored_view else np.ones_like(s_.need),
                             'cls': c, 'group': s_.group, 'f': {'imb': C[:, -36:].copy()}})
            chunk = []
        if s is None:
            return acc, rows


def summarize(acc, names, top=12):
    out = {}
    for c, a in acc.items():
        if a.n[0] <= 0:
            continue
        R = a.corr()                                                    # (2, K, S)
        null_max = np.abs(R[:, :, 1:]).max(1)                           # (2, N_NULL): max over candidates per replicate
        entry = {'weight_rows': float(a.n[0]), 'null_max_abs_corr': {'mean': float(null_max.mean()), 'max': float(null_max.max())}}
        for ch, nm in enumerate(('direction_resid', 'cost_resid')):
            order = np.argsort(-np.abs(R[ch, :, 0]))[:top]
            entry[nm] = [{'name': names[k], 'corr': float(R[ch, k, 0]), 'null_sd': float(R[ch, k, 1:].std())} for k in order]
        lift = a.lift
        entry['lift_top'] = sorted(({'name': names[j], 'p_hi': float(lift[1, j] / max(lift[0, j], 1)),
                                     'p_lo': float(lift[3, j] / max(lift[2, j], 1))} for j in range(44)),
                                   key=lambda d: -abs(d['p_hi'] - d['p_lo']))[:8]
        den = max(a.nr, 1)
        for nm, M in (('corr_vol_i0', a.cv), ('corr_price_i0', a.cp)):
            sd = np.sqrt(np.diag(M) / den) + 1e-12
            entry[nm] = np.round(M / den / np.outer(sd, sd), 2).tolist()
        out[f'c{c}'] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--name', required=True)
    ap.add_argument('--n-valid', type=int, default=1873)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    vcls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    models = [load(Path(r), 'best.pt', device)[0] for r in args.runs]
    rng = np.random.default_rng(7)
    names = candidates(next(iter_sequences(VALID, [0])).X)[1]
    res = {'candidates': len(names), 'runs': args.runs}
    acc_h, hold = run(TRAIN, holdout_groups(), tcls, models, device, names, False, rng)
    acc_v, valid = run(VALID, list(range(args.n_valid)), vcls, models, device, names, True, rng)
    res['screen_holdout_V3rows'] = summarize(acc_h, names)
    res['screen_valid_scored'] = summarize(acc_v, names)
    for v in ('screen_holdout_V3rows', 'screen_valid_scored'):
        for c in ('c3', 'c0'):
            e = res[v].get(c)
            if e:
                print(v, c, 'null max |corr|', {k: round(x, 4) for k, x in e['null_max_abs_corr'].items()},
                      'top dir', [(d['name'], round(d['corr'], 4)) for d in e['direction_resid'][:4]],
                      'top cost', [(d['name'], round(d['corr'], 4)) for d in e['cost_resid'][:3]], flush=True)
    for r in hold + valid:
        r['fitmask'] = r['need'] & r['scored']
    G.GROUPS = {'all': ['imb']}
    res['G1_holdout_to_valid'] = G.gate(hold, valid, 'fitmask')
    A = [r for r in valid if r['group'] < 937]
    B = [r for r in valid if r['group'] >= 937]
    res['G2_A_to_B'] = G.gate(A, B, 'fitmask')
    res['G2_B_to_A'] = G.gate(B, A, 'fitmask')
    for k in ('G1_holdout_to_valid', 'G2_A_to_B', 'G2_B_to_A'):
        for view in ('V1', 'V2'):
            g = res[k][view]
            base = g['none']['all']
            print(f"{k} {view}: pred {g['pred']['all']:.5f} ridge[pred] {base:.5f} | +imb {g['all']['all'] - base:+.5f} | per class: " +
                  ' '.join(f"c{c} {g['all'][f'c{c}'] - g['none'][f'c{c}']:+.4f}" for c in range(4)), flush=True)
    res['histgb_G1'] = G.histgb_check(hold, valid)
    print('HistGB G1 (pred + imb + class):', {v: round(x['all'], 5) for v, x in res['histgb_G1'].items()}, flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
