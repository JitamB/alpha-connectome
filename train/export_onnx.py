"""Export a trained GRUModel to a submission directory (model.onnx + solution.py).

  python train/export_onnx.py runs/gru_base [--weights best.pt] --name gru_base

The ONNX graph takes (features [1,1,F], hidden_i [1,1,H]) and returns (prediction, next_hidden_i),
the same stateful interface as the starterpack baseline; batch and time axes are dynamic.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
from train_gru import TRANSFORMS, GRUModel  # noqa: E402


class Stateful(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, *hidden):
        pred, new = self.model(features, list(hidden))
        return (pred, *new)


def export(run: Path, weights: str, name: str):
    cfg = json.loads((run / 'config.json').read_text())
    n_in = TRANSFORMS[cfg['features']][0]
    model = GRUModel(n_in, cfg['hidden'], cfg['layers'], cfg['skip'], cfg.get('proj', 0))
    model.load_state_dict(torch.load(run / weights, map_location='cpu'))
    model.eval()
    out = ROOT / 'submissions' / name
    out.mkdir(parents=True, exist_ok=True)
    hidden_names = [f'hidden_{i}' for i in range(cfg['layers'])]
    dummy = (torch.zeros(1, 1, n_in), *[torch.zeros(1, 1, cfg['hidden']) for _ in hidden_names])
    dynamic = {'features': {0: 'batch', 1: 'time'}, 'prediction': {0: 'batch', 1: 'time'}}
    dynamic.update({n: {1: 'batch'} for n in hidden_names})
    dynamic.update({f'next_{n}': {1: 'batch'} for n in hidden_names})
    torch.onnx.export(Stateful(model), dummy, str(out / 'model.onnx'), dynamo=False, opset_version=17,
                      input_names=['features', *hidden_names],
                      output_names=['prediction', *[f'next_{n}' for n in hidden_names]],
                      dynamic_axes=dynamic)
    shutil.copy(ROOT / 'templates' / 'solution.py', out / 'solution.py')
    if cfg['features'] == 'eng':
        shutil.copy(ROOT / 'src' / 'features.py', out / 'features.py')
    print(f'exported {run / weights} -> {out}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    export(args.run, args.weights, args.name)
