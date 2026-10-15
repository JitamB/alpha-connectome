"""Evaluate trained GRU run(s) on V1/V2/V3 with the torch model (GPU if available), via src/evaluate.py.

  python train/eval_torch.py runs/a [runs/b ...] --name NAME [--weights best.pt]

Several runs are averaged. Writes results/NAME.npz/json in the same format as the ONNX evaluator, so
`python src/evaluate.py compare` works across both. Torch-vs-ONNX prediction differences are ~1e-6.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from evaluate import evaluate, print_summary  # noqa: E402
from gpu_features import torch_features  # noqa: E402
from train_gru import TRANSFORMS, GRUModel, decode_dist, make_atoms  # noqa: E402


def load(run: Path, weights: str, device):
    cfg = json.loads((run / 'config.json').read_text())
    atoms = len(make_atoms()) if cfg.get('dist_weight') else 0
    conv = (cfg['conv'][0], cfg['conv'][1], cfg['conv'][2:]) if cfg.get('conv') else None
    model = GRUModel(TRANSFORMS[cfg['features']][0], cfg['hidden'], cfg['layers'], cfg['skip'], cfg.get('proj', 0), atoms, conv)
    model.load_state_dict(torch.load(run / weights, map_location=device))
    return model.to(device).eval(), cfg['features'] == 'eng'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--name', required=True)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-seqs', type=int, help='quick check on the first N sequences of each view')
    ap.add_argument('--output', choices=['reg', 'dist', 'blend'], default='reg',
                    help='regression head, decoded distributional head, or their average')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [load(r, args.weights, device) for r in args.runs]
    atoms = torch.tensor(make_atoms(), device=device)

    def output(m, x):
        reg, _, top = m(x, None, return_top=True)
        if args.output == 'reg':
            return reg
        dist = decode_dist(m.dist_head(top).unflatten(-1, (2, len(atoms))), atoms)
        return dist if args.output == 'dist' else (reg + dist) / 2

    @torch.no_grad()
    def predict(Xb):
        X = torch.from_numpy(Xb).to(device)
        F = torch_features(X) if any(eng for _, eng in models) else None
        preds = [output(m, F if eng else X) for m, eng in models]
        return torch.stack(preds).mean(0).float().cpu().numpy()

    print_summary(args.name, evaluate(predict, args.name, max_seqs=args.max_seqs, batch=args.batch))
