"""Episode probe: is the label a fixed reference minus the current quote, re-anchored at rare events? (Kaggle CPU)

  python tools/episode_probe.py --name episode_probe [--classes 3 2] [--n-table 1500] [--n-test 200]

Theory (from results/c3_anatomy.npz): in class 3 the label pairs are almost only (0,-1), (-1,0), (k,-3) and (-3,k).
After a label of (0,-1) the next change gives (-1,-1) 79% of the time. After an up label (1,-3) it gives (0,-2) 44%
of the time. That looks like n0 = F - a_t and n1 = b_t - F', in ticks, where a_t and b_t are the current ask and bid
and F and F' are references that stay fixed while the quote moves.
Method:
  1. On n_table train sequences per class, learn the modal label change per i0 transition type
     (previous shape, new shape), i.e. the decoder of the quote move (-da, +db).
  2. On n_test other train sequences, decode a_t and b_t by cumulative sums of the table moves. Then compute
     F = n0 + a and F' = b - n1 per segment, and flag re-anchors: transitions whose actual label change differs from
     the table move.
  3. Report: the re-anchor rate, episode lengths in rows and transitions, F and F' relative to the quote at the
     re-anchor (a fixed offset would mean past-anchored, i.e. computable), whether a re-anchor happens when the quote
     reaches the old F or F', the F - F' gap, and example segment tables.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'src'), str(ROOT / 'train')]
import gate_paths as G  # noqa: E402
from data import TRAIN, iter_sequences, train_groups  # noqa: E402
from train_gru import STEPS  # noqa: E402


def segments(s, step):
    """-> starts (K,), shape ids (K,), labels (K, 2) int (label constant within a segment is checked), n_bad."""
    tr, typ = G.transitions(s.X[:, 0:22])
    ids = G.shape_ids(s.X[:, 0:22])
    st = np.r_[0, np.flatnonzero(tr)]
    n = np.rint(s.y.astype(np.float64) / step).astype(np.int64)
    ends = np.r_[st[1:], len(n)]
    bad = sum(int((n[a:b] != n[a]).any()) for a, b in zip(st, ends))
    return st, ids[st], n[st], typ[st], bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--classes', type=int, nargs='+', default=[3, 2])
    ap.add_argument('--n-table', type=int, default=1500)
    ap.add_argument('--n-test', type=int, default=200)
    args = ap.parse_args()
    tcls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    res = {}
    for c in args.classes:
        groups = [g for g in train_groups() if tcls[g] == c]
        rng = np.random.default_rng(c)
        groups = [int(g) for g in rng.permutation(groups)]
        tab_g, test_g = groups[:args.n_table], groups[args.n_table:args.n_table + args.n_test]
        votes = {}
        for s in iter_sequences(TRAIN, sorted(tab_g)):
            st, _, lab, typ, _ = segments(s, STEPS[c])
            d = lab[1:] - lab[:-1]
            for t, (d0, d1) in zip(typ[1:].tolist(), d.tolist()):
                votes.setdefault(t, Counter())[(d0, d1)] += 1
        table = {t: v.most_common(1)[0][0] for t, v in votes.items()}
        purity = sum(v.most_common(1)[0][1] for v in votes.values()) / sum(sum(v.values()) for v in votes.values())
        print(f'class {c}: table {len(table)} types, in-sample purity {purity:.4f}', flush=True)
        st_all = Counter()
        n_tr = n_re = n_miss = n_bad = n_seg = 0
        ep_rows, ep_trans = [], []
        anchor_F, anchor_Fp, jumpF, jumpFp, gap, hit_old = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
        moves = Counter()
        examples = []
        for k, s in enumerate(iter_sequences(TRAIN, sorted(test_g))):
            st, ids, lab, typ, bad = segments(s, STEPS[c])
            n_bad += bad
            n_seg += len(st)
            K = len(st)
            da = np.zeros(K, np.int64); db = np.zeros(K, np.int64); re = np.zeros(K, bool); known = np.ones(K, bool)
            for i in range(1, K):
                mv = table.get(int(typ[i]))
                d = (int(lab[i, 0] - lab[i - 1, 0]), int(lab[i, 1] - lab[i - 1, 1]))
                if mv is None:
                    known[i] = False
                    n_miss += 1
                    continue
                da[i], db[i] = -mv[0], mv[1]
                moves[(da[i], db[i])] += 1
                if d != tuple(mv):
                    re[i] = True
            n_tr += K - 1
            n_re += int(re.sum())
            a, b = np.cumsum(da), np.cumsum(db)
            F, Fp = lab[:, 0] + a, b - lab[:, 1]
            idx = np.flatnonzero(re)
            last = 0
            for i in idx:
                ep_rows.append(int(st[i] - st[last])); ep_trans.append(int(i - last))
                anchor_F[int(F[i] - a[i])] += 1          # new F relative to the ask at the re-anchor (= n0 there)
                anchor_Fp[int(b[i] - Fp[i])] += 1        # = n1 there
                jumpF[int(F[i] - F[i - 1])] += 1
                jumpFp[int(Fp[i] - Fp[i - 1])] += 1
                hit_old[(int(a[i] - F[i - 1]), int(b[i] - Fp[i - 1]))] += 1   # quote vs the OLD references
                last = i
            for i in range(K):
                gap[int(F[i] - Fp[i])] += 1
            if k < 4:
                examples.append([[int(st[i]), int(ids[i] % 1000), int(lab[i, 0]), int(lab[i, 1]), int(a[i]), int(b[i]),
                                  int(F[i]), int(Fp[i]), bool(re[i]), bool(known[i])] for i in range(min(K, 160))])
        top = lambda cnt, k=12: [[str(key), round(v / max(sum(cnt.values()), 1), 4)] for key, v in cnt.most_common(k)]
        r = {'table_types': len(table), 'table_purity_in_sample': purity, 'segments': n_seg,
             'segments_label_not_constant': n_bad, 'transitions': n_tr, 'unknown_type_rate': n_miss / max(n_tr, 1),
             'reanchor_rate': n_re / max(n_tr - n_miss, 1),
             'episode_rows_q': np.percentile(ep_rows, [10, 25, 50, 75, 90]).tolist() if ep_rows else [],
             'episode_transitions_q': np.percentile(ep_trans, [10, 25, 50, 75, 90]).tolist() if ep_trans else [],
             'decoded_moves(da,db)': top(moves), 'new_n0_at_reanchor': top(anchor_F), 'new_n1_at_reanchor': top(anchor_Fp),
             'F_jump_at_reanchor': top(jumpF), 'Fp_jump_at_reanchor': top(jumpFp),
             'quote_minus_old_refs_at_reanchor(a-F_old,b-Fp_old)': top(hit_old, 15), 'F_minus_Fp_all_segments': top(gap),
             'examples(start,shape%1000,n0,n1,a,b,F,Fp,reanchor,known)': examples}
        res[f'c{c}'] = r
        print(json.dumps({k: v for k, v in r.items() if not k.startswith('examples')}, default=float), flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
