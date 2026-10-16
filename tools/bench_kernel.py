"""Time the numba kernel with K = 1..K_max models on this machine's CPU (one valid sequence, best of 3).

  python tools/bench_kernel.py submissions/ens3_all_tv --name bench_kaggle_cpu

Used to check whether a machine reproduces the scorer's super-linear slowdown at K = 3 (cache spill).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import VALID, iter_sequences  # noqa: E402
from fast_gru import WEIGHT_NAMES, FastEnsemble  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    W = dict(np.load(args.submission / 'weights.npz'))
    X = np.ascontiguousarray(next(iter(iter_sequences(VALID, [0]))).X, dtype=np.float32)
    out = {}
    try:  # CPU model and the vector extensions that matter for the kernel
        info = Path('/proc/cpuinfo').read_text().split('\n\n')[0]
        model = next(l.split(':', 1)[1].strip() for l in info.splitlines() if l.startswith('model name'))
        flags = next(l.split(':', 1)[1].split() for l in info.splitlines() if l.startswith('flags'))
        out['cpu'] = {'model': model, 'flags': sorted(f for f in flags if f.startswith(('avx', 'fma', 'amx')))}
        print(out['cpu'], flush=True)
    except (OSError, StopIteration):
        pass
    for K in range(1, W['Whh0'].shape[0] + 1):
        w = {k: (v if k in ('atoms', 'alpha') else np.ascontiguousarray(v[:K])) for k, v in W.items()}
        m = FastEnsemble(w)
        m.update(X[0])
        best = 1e9
        for _ in range(3):
            m.reset()
            t = time.perf_counter()
            for x in X:
                m.update(x)
            best = min(best, (time.perf_counter() - t) / len(X) * 1e6)
        out[K] = best
        print(f'K={K}: {best:.1f} us/row', flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(out, indent=2))
