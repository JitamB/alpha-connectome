"""Scoring-mask proxy for training weights: a small MLP p(is_scored | current features), fitted on valid half A.

  python models/mask_model.py [--seqs 120]

Inputs are the first 336 engineered columns (raw, 1-step diff, x - EMA10), i.e. F[..., :336] of
features.batch_features / gpu_features.torch_features, so the trainer can evaluate it on the GPU.
Valid half A = row groups < 937 (used here); half B = the rest, kept for scoring the mask-weighted model.
Saves models/mask_mlp.pt: {'mean', 'std', 'W1', 'b1', 'w2', 'b2'}.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import VALID, iter_sequences  # noqa: E402
from features import batch_features  # noqa: E402

N_IN = 336
HALF_A = 937
OUT = ROOT / 'models' / 'mask_mlp.pt'


def mask_prob(F, mw):
    """F: (..., >=336) engineered features -> p(is_scored), same shape without the last axis."""
    z = (F[..., :N_IN] - mw['mean']) / mw['std']
    return torch.sigmoid(torch.relu(z @ mw['W1'].T + mw['b1']) @ mw['w2'] + mw['b2'])


def rows(groups, keep_unmasked, rng):
    """Masked rows plus a random share of unmasked rows; unmasked rows get weight 1/keep_unmasked."""
    F, y, w = [], [], []
    for s in iter_sequences(VALID, groups):
        idx = np.flatnonzero(s.need)
        m = s.scored[idx]
        sel = idx[m | (rng.random(len(idx)) < keep_unmasked)]
        F.append(batch_features(s.X)[sel, :N_IN])
        y.append(s.scored[sel])
        w.append(np.where(s.scored[sel], 1.0, 1.0 / keep_unmasked))
    return np.concatenate(F), np.concatenate(y).astype(np.float32), np.concatenate(w).astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', type=int, default=120)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    F, y, w = rows(sorted(rng.choice(HALF_A, args.seqs, replace=False).tolist()), 0.1, rng)
    Fb, yb, wb = rows(range(HALF_A, HALF_A + 40), 0.2, rng)
    mean, std = F.mean(0), F.std(0) + 1e-6
    X, Xb = torch.from_numpy((F - mean) / std), torch.from_numpy((Fb - mean) / std)
    Y, Wt = torch.from_numpy(y), torch.from_numpy(w)
    net = torch.nn.Sequential(torch.nn.Linear(N_IN, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 4096):
            j = perm[i:i + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[j]).squeeze(-1), Y[j], weight=Wt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pb = torch.sigmoid(net(Xb).squeeze(-1)).numpy()
        print(f'epoch {epoch}: half-B AUC {roc_auc_score(yb, pb):.4f}  mean p (reweighted) '
              f'{np.average(pb, weights=wb):.3f} vs scored share {np.average(yb, weights=wb):.3f}', flush=True)
    l1, l2 = net[0], net[2]
    torch.save({'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std), 'W1': l1.weight.detach(),
                'b1': l1.bias.detach(), 'w2': l2.weight.detach()[0], 'b2': l2.bias.detach()[0]}, OUT)
    with torch.no_grad():
        check = mask_prob(torch.from_numpy(Fb), torch.load(OUT)).numpy()
    print(f'saved {OUT} (reload AUC {roc_auc_score(yb, check):.4f})')
