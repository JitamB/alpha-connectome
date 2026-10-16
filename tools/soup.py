"""Model soup: average the weights of several runs fine-tuned from one checkpoint into a single model.

  python tools/soup.py runs/ft1 runs/ft2 runs/ft3 --name soup_x [--no-eval]

Writes runs/NAME/best.pt (+ the first run's config.json) and, unless --no-eval, scores it with train/eval_torch.py.
Only meaningful for runs that share an initialisation (e.g. train_gru --init); inference cost is one model.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--name', required=True)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--no-eval', action='store_true')
    args = ap.parse_args()
    sds = [torch.load(r / args.weights, map_location='cpu') for r in args.runs]
    soup = {k: sum(sd[k].double() for sd in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k]
            for k in sds[0]}
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(sds[0][k].dtype) for k, v in soup.items()}, out / 'best.pt')
    shutil.copy(args.runs[0] / 'config.json', out / 'config.json')
    print(f'soup of {len(sds)} runs -> {out}', flush=True)
    if not args.no_eval:
        subprocess.run([sys.executable, '-u', str(ROOT / 'train' / 'eval_torch.py'), str(out), '--name', args.name,
                        '--batch', '32'], check=True)
