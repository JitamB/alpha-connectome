"""Fast whole-sequence evaluation on three validation views, with bootstrap noise.

V1: valid.parquet, official mask (need_prediction & is_scored) -- comparable to the leaderboard metric
V2: valid.parquet, all need_prediction rows
V3: fixed train holdout (splits/holdout.json), all need_prediction rows

Per-sequence weighted moments are saved to results/<name>.npz so that any two runs
can be compared with a paired bootstrap over sequences.

Usage:
  python src/evaluate.py onnx PATH [PATH ...] --name NAME [--views V1 V2 V3] [--max-seqs N]   (several paths = averaged)
  python src/evaluate.py compare NAME_A NAME_B [--valid-from 937]   (V1/V2 on valid row groups >= 937 only)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from data import ROOT, VALID, TRAIN, iter_sequences, holdout_groups

RESULTS = ROOT / 'results'
CLIP = 2.0
N_BOOT = 1000
VIEWS = ('V1', 'V2', 'V3')


# ---------------------------------------------------------------- metric from moments
def seq_moments(y, p, mask):
    """(6, 2) weighted sums [W, Swy, Swp, Swyy, Swpp, Swyp] per target, metric conventions of utils.py."""
    y = np.clip(y[mask], -CLIP, CLIP).astype(np.float64)
    p = np.clip(p[mask], -CLIP, CLIP).astype(np.float64)
    w = np.abs(y)
    return np.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0),
                     (w * y * y).sum(0), (w * p * p).sum(0), (w * y * p).sum(0)])


def wp_from_moments(m):
    """m: (..., 6, 2) summed moments -> (..., 2) per-target weighted Pearson."""
    W, sy, sp, syy, spp, syp = (m[..., i, :] for i in range(6))
    with np.errstate(invalid='ignore', divide='ignore'):
        my, mp = sy / W, sp / W
        vy = np.maximum(syy / W - my * my, 0)
        vp = np.maximum(spp / W - mp * mp, 0)
        rho = (syp / W - my * mp) / np.sqrt(vy * vp)
    ok = (W >= 1e-8) & (np.sqrt(vy) > 1e-8) & (np.sqrt(vp) > 1e-8)
    return np.clip(np.where(ok, rho, 0.0), -1, 1)


def bootstrap(moments_list, seed=0):
    """Resample sequences jointly for all arrays in moments_list -> list of (N_BOOT,) scores."""
    n = len(moments_list[0])
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float64)
    return [wp_from_moments((counts @ m.reshape(n, 12)).reshape(N_BOOT, 6, 2)).mean(-1) for m in moments_list]


def summarize(moments):
    per_target = wp_from_moments(moments.sum(0))
    return {'wp': float(per_target.mean()), 't0': float(per_target[0]), 't1': float(per_target[1]),
            'sigma': float(bootstrap([moments])[0].std()), 'seqs': len(moments)}


# ---------------------------------------------------------------- predictors
class OnnxSequencePredictor:
    """Runs a stateful ONNX model (features + hidden_* inputs) over whole sequences in one call.

    Every size-1 dimension of graph inputs/outputs is made symbolic so the same file used
    row by row in a submission can take (batch, time, 112) here. Hidden state starts at zero.
    If a features.py sits next to the model, its batch_features() is applied to each sequence first.
    """

    def __init__(self, path, threads=4):
        features_py = Path(path).with_name('features.py')
        self.transform = None
        if features_py.exists():
            spec = importlib.util.spec_from_file_location(f'features_{abs(hash(str(features_py)))}', features_py)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.transform = module.batch_features
        model = onnx.load(str(path))
        for k, vi in enumerate(list(model.graph.input) + list(model.graph.output)):
            for j, d in enumerate(vi.type.tensor_type.shape.dim):
                if d.dim_value == 1:
                    d.dim_param = f'dyn_{k}_{j}'
        del model.graph.value_info[:]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model.SerializeToString(), opts, providers=['CPUExecutionProvider'])
        self.hidden = {i.name: [d.dim_value or 1 for d in i.type.tensor_type.shape.dim]
                       for i in model.graph.input if i.name != 'features'}

    def __call__(self, X):
        """X: (B, T, 112) float32 -> (B, T, 2) float32."""
        if self.transform is not None:
            X = np.stack([self.transform(x) for x in X])
        feeds = {'features': X}
        for name, shape in self.hidden.items():
            feeds[name] = np.zeros((shape[0], len(X), *shape[2:]), dtype=np.float32)
        return self.session.run(['prediction'], feeds)[0]


# ---------------------------------------------------------------- evaluation
def _batched(seqs, size):
    batch = []
    for s in seqs:
        batch.append(s)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate(predict, name, views=VIEWS, max_seqs=None, batch=8):
    """predict: (B, T, 112) -> (B, T, 2), must be causal and start each sequence from a fresh state."""
    RESULTS.mkdir(exist_ok=True)
    moments = {v: [] for v in views}
    sources = []
    if {'V1', 'V2'} & set(views):
        sources.append((VALID, None))
    if 'V3' in views:
        sources.append((TRAIN, holdout_groups()))
    t0 = time.time()
    for path, groups in sources:
        if max_seqs is not None:
            groups = (groups or range(max_seqs))[:max_seqs]
        for chunk in _batched(iter_sequences(path, groups), batch):
            preds = predict(np.stack([s.X for s in chunk]))
            for s, p in zip(chunk, preds):
                if not np.isfinite(p[s.need]).all():
                    raise ValueError(f'nonfinite prediction in seq {s.seq_ix}')
                if s.scored is not None:
                    if 'V1' in views:
                        moments['V1'].append(seq_moments(s.y, p, s.need & s.scored))
                    if 'V2' in views:
                        moments['V2'].append(seq_moments(s.y, p, s.need))
                else:
                    moments['V3'].append(seq_moments(s.y, p, s.need))
    moments = {v: np.stack(m) for v, m in moments.items()}
    summary = {v: summarize(m) for v, m in moments.items()}
    summary['eval_seconds'] = round(time.time() - t0, 1)
    np.savez(RESULTS / f'{name}.npz', **moments)
    (RESULTS / f'{name}.json').write_text(json.dumps(summary, indent=2))
    return summary


def compare(a, b, valid_from=0):
    """Paired bootstrap of score(b) - score(a) per view; valid_from restricts V1/V2 to valid row groups >= it
    (e.g. 937 = half B, when valid half A was used to fit the mask proxy)."""
    A, B = np.load(RESULTS / f'{a}.npz'), np.load(RESULTS / f'{b}.npz')
    out = {}
    for v in sorted(set(A.files) & set(B.files)):
        if len(A[v]) != len(B[v]):
            continue
        ma, mb = (A[v][valid_from:], B[v][valid_from:]) if v in ('V1', 'V2') else (A[v], B[v])
        sa, sb = bootstrap([ma, mb])
        delta = float(wp_from_moments(mb.sum(0)).mean() - wp_from_moments(ma.sum(0)).mean())
        out[v] = {'delta': delta, 'sigma': float((sb - sa).std()), 'z': delta / float((sb - sa).std())}
    return out


def print_summary(name, summary):
    print(f'== {name}')
    for v in VIEWS:
        if v in summary:
            s = summary[v]
            print(f"  {v}: WP {s['wp']:.6f}  (t0 {s['t0']:.4f}, t1 {s['t1']:.4f})  sigma {s['sigma']:.4f}  seqs {s['seqs']}")
    print(f"  eval time {summary['eval_seconds']} s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('onnx')
    o.add_argument('path', nargs='+')
    o.add_argument('--name', required=True)
    o.add_argument('--views', nargs='+', default=list(VIEWS))
    o.add_argument('--max-seqs', type=int)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--valid-from', type=int, default=0)
    args = ap.parse_args()
    if args.cmd == 'onnx':
        predictors = [OnnxSequencePredictor(p) for p in args.path]
        predict = predictors[0] if len(predictors) == 1 else (lambda X: np.mean([f(X) for f in predictors], axis=0))
        print_summary(args.name, evaluate(predict, args.name, args.views, args.max_seqs))
    else:
        for v, r in compare(args.a, args.b, args.valid_from).items():
            print(f"  {v}: {args.b} - {args.a} = {r['delta']:+.5f}  (sigma {r['sigma']:.5f}, z {r['z']:+.1f})")
