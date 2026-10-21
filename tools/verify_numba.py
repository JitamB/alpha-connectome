"""Numba submission vs the torch models it was exported from: row-by-row outputs on the first N valid sequences.

  python tools/verify_numba.py submissions/ens2_x runs/a runs/b [--seqs 4] [--name verify_x]

Complements src/check_submission.py (which has no torch reference for numba packages). Writes results/NAME.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from check_submission import load_model_class, run_rows  # noqa: E402
from data import VALID, WARMUP, iter_sequences  # noqa: E402
from eval_torch import load  # noqa: E402
from gpu_features import torch_features  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path)
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--seqs', type=int, default=4)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    models = [load(r, 'best.pt', torch.device('cpu'))[0] for r in args.runs]
    Model = load_model_class(args.submission, 'v')
    diffs = []
    for s in iter_sequences(VALID, range(args.seqs)):
        with torch.no_grad():
            F = torch_features(torch.from_numpy(s.X)[None])
            ref = torch.stack([m(F, None)[0] for m in models]).mean(0)[0].numpy()
        got = run_rows(Model(), s)
        diffs.append(float(np.abs(got[WARMUP:] - ref[WARMUP:]).max()))
        print(f'seq {s.seq_ix}: max |numba - torch| = {diffs[-1]:.2e}', flush=True)
    out = {'max_abs_diff': max(diffs), 'per_seq': diffs, 'ok': max(diffs) <= 1e-4}
    print('PASS' if out['ok'] else 'FAIL', out, flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(out, indent=2))
