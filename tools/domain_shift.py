"""How different are valid sequences from train sequences, beyond the asset-class mix? (covariate-shift check)

  python tools/domain_shift.py --name domain_shift [--workers 4]

Per sequence: mean and std of each raw column and mean |1-step diff| (336 numbers) + asset class. A HistGB classifier
valid-vs-train with 5-fold CV (folds by sequence) gives the AUC overall and within each class, and an out-of-fold
p(valid) for every train sequence. Writes results/NAME.json (AUCs, top features) and results/NAME_train_p.json
(p per train row group, for importance weights w ~ p / (1 - p)).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import TRAIN, VALID, read_sequence  # noqa: E402


def summarise(job):
    path, groups = job
    pf = pq.ParquetFile(path)
    out = []
    for g in groups:
        X = read_sequence(pf, g).X.astype(np.float64)
        out.append(np.concatenate([X.mean(0), X.std(0), np.abs(np.diff(X, axis=0)).mean(0)]))
    return out


def table(path, n, workers):
    parts = np.array_split(np.arange(n), workers * 8)
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(summarise, [(path, p.tolist()) for p in parts])
    return np.array([r for part in res for r in part])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    n_tr, n_va = pq.ParquetFile(TRAIN).num_row_groups, pq.ParquetFile(VALID).num_row_groups
    Ftr, Fva = table(TRAIN, n_tr, args.workers), table(VALID, n_va, args.workers)
    print(f'summarised {n_tr} train + {n_va} valid sequences', flush=True)
    ctr = np.array(json.loads((ROOT / 'splits' / 'train_classes.json').read_text()))
    cva = np.array(json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls'])
    F = np.vstack([Ftr, Fva])
    cls = np.concatenate([ctr, cva])
    y = np.concatenate([np.zeros(n_tr), np.ones(n_va)])
    names = [f'{s}_{c}' for s in ('mean', 'std', 'absdiff') for c in range(112)]
    res = {}
    for label, X in (('with_class', np.column_stack([F, cls])), ('features_only', F)):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y * 4 + cls):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        res[label] = {'auc': roc_auc_score(y, oof),
                      'auc_within_class': {int(c): roc_auc_score(y[cls == c], oof[cls == c]) for c in range(4)}}
        print(label, res[label], flush=True)
        if label == 'with_class':
            p_train = oof[:n_tr]
    # which summaries separate the domains (univariate AUC within class, pooled by averaging over classes)
    uni = []
    for j in range(F.shape[1]):
        a = np.mean([roc_auc_score(y[cls == c], F[cls == c, j]) for c in range(4)])
        uni.append((abs(a - 0.5), names[j], a))
    res['top_univariate'] = [(n, round(a, 3)) for _, n, a in sorted(uni, reverse=True)[:20]]
    print('top univariate (mean within-class AUC):', res['top_univariate'], flush=True)
    q = np.quantile(p_train, [0.05, 0.25, 0.5, 0.75, 0.95])
    res['p_train_quantiles'] = q.tolist()
    print('train p(valid) quantiles', q, flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(res, indent=2))
    (ROOT / 'results' / f'{args.name}_train_p.json').write_text(json.dumps(p_train.round(5).tolist()))
    np.savez(ROOT / 'results' / f'{args.name}_summaries.npz', F=F, cls=cls, y=y)
