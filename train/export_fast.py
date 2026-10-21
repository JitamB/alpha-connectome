"""Export one or more trained GRU runs as a numba submission (weights.npz + fast_gru.py + solution.py).

  python train/export_fast.py runs/a [runs/b ...] --name ens_ab [--weights best.pt]

Predictions are the average of the given runs. Runs must share the architecture supported by
src/fast_gru.py (engineered features, projection, 2 GRU layers, skip head).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fast_gru import pack  # noqa: E402
sys.path.insert(0, str(ROOT / 'train'))
from train_gru import BLOCKS, make_atoms  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--name', required=True)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--bf16', action='store_true', help='store the large matrices as bfloat16 (half the weight bytes)')
    ap.add_argument('--alpha', type=float, default=0.0, help='share of the decoded distributional head in the output')
    args = ap.parse_args()
    sds = []
    for run in args.runs:
        cfg = json.loads((run / 'config.json').read_text())
        assert cfg['features'] == 'eng' and cfg['layers'] == 2 and cfg['skip'] and cfg.get('proj') and not cfg.get('conv') \
            and cfg.get('cell', 'gru') == 'gru', \
            f'{run}: unsupported architecture'
        sd = {k: v.numpy().copy() for k, v in torch.load(run / args.weights, map_location='cpu').items()}
        for b in cfg.get('drop_blocks') or ():  # trained with these inputs zeroed after standardisation
            cols = slice(BLOCKS.index(b) * 112, (BLOCKS.index(b) + 1) * 112)
            sd['proj.weight'][:, cols] = 0.0
            sd['skip_head.weight'][:, cols] = 0.0
        sds.append(sd)
    out = ROOT / 'submissions' / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'weights.npz', **pack(sds, make_atoms(), args.alpha, args.bf16))
    shutil.copy(ROOT / 'src' / 'fast_gru.py', out / 'fast_gru.py')
    shutil.copy(ROOT / 'templates' / 'solution_fast.py', out / 'solution.py')
    (out / 'MODELS.txt').write_text('\n'.join(f'{r}/{args.weights}' for r in args.runs) + '\n')
    print(f'exported average of {len(sds)} model(s) -> {out}')
