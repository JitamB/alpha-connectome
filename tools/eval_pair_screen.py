"""Kaggle-only post-train evaluation for the structured joint-pair screen."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from data import TRAIN, VALID, holdout_groups, iter_sequences  # noqa: E402
from eval_torch import load  # noqa: E402
from gpu_features import torch_features  # noqa: E402


def class_accuracy(run_names):
    import json

    train_cls = json.loads((ROOT / 'splits' / 'train_classes.json').read_text())
    valid_cls = json.loads((ROOT / 'splits' / 'valid_classes.json').read_text())['cls']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in run_names:
        model, _ = load(ROOT / 'runs' / name, 'best.pt', device)
        if model.class_head is None:
            raise ValueError(f'{name}: missing class head')
        for label, path, groups, classes in (
            ('valid', VALID, range(len(valid_cls)), valid_cls),
            ('holdout', TRAIN, holdout_groups(), train_cls),
        ):
            correct = total = 0
            chunk = []
            for seq in iter_sequences(path, groups):
                chunk.append(seq)
                if len(chunk) == 32:
                    X = torch.from_numpy(np.stack([s.X[:99] for s in chunk])).to(device)
                    F = torch_features(X)
                    with torch.no_grad():
                        _, _, top = model(F, None, return_top=True)
                        guess = model.class_head(top).mean(1).argmax(-1).cpu().numpy()
                    truth = [classes[s.group] for s in chunk]
                    correct += int((guess == truth).sum()); total += len(chunk); chunk = []
            if chunk:
                X = torch.from_numpy(np.stack([s.X[:99] for s in chunk])).to(device)
                F = torch_features(X)
                with torch.no_grad():
                    _, _, top = model(F, None, return_top=True)
                    guess = model.class_head(top).mean(1).argmax(-1).cpu().numpy()
                truth = [classes[s.group] for s in chunk]
                correct += int((guess == truth).sum()); total += len(chunk)
            print(f'{name} class detector {label}: {correct}/{total} = {correct / max(total, 1):.6f}', flush=True)


def main():
    seeds = sys.argv[1:] or ['jointpair_s0', 'jointpair_s1']
    for name in seeds:
        subprocess.run([sys.executable, 'train/eval_torch.py', f'runs/{name}', '--name', f'{name}_pair',
                        '--batch', '32', '--output', 'pair'], cwd=ROOT, check=True)
    class_accuracy(seeds)
    commands = [
        ['tools/judge.py', '--base', 'lr7e4_full', 'lr7e4_full_seed1', '--new', *seeds],
        ['tools/judge.py', '--base', *seeds, '--new', f'{seeds[0]}_pair', f'{seeds[1]}_pair'],
        ['tools/judge.py', '--base', 'lr7e4_full', 'lr7e4_full_seed1', '--new', f'{seeds[0]}_pair', f'{seeds[1]}_pair'],
    ]
    for cmd in commands:
        subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
