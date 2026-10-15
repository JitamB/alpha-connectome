"""Torch (GPU) version of src/features.py batch_features, for training on machines with few CPU cores.

Within a chunk of L steps an EMA with decay d = 1 - a and incoming state s is exactly
    ema_j = d^(j+1) s + sum_{k<=j} a d^(j-k) x_k,
i.e. one (L x L) lower-triangular matrix product per span. Chunks are processed in order, carrying state.
Verified against batch_features in tests (max abs diff ~1e-5).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import EMA_SPANS, VOL_SPAN  # noqa: E402

_CACHE = {}


def _decay_mats(span, L, device):
    key = (span, L, str(device))
    if key not in _CACHE:
        a = 2.0 / (span + 1)
        d = 1.0 - a
        j = torch.arange(L, device=device, dtype=torch.float64)
        expo = (j[:, None] - j[None, :]).clamp_min(0)
        M = torch.where(j[:, None] >= j[None, :], a * d ** expo, torch.zeros((), dtype=torch.float64, device=device))
        v = d ** (j + 1)
        _CACHE[key] = (M.float(), v.float())
    return _CACHE[key]


@torch.no_grad()
def torch_features(X, chunk=1000):
    """X: (B, T, 112) float32 tensor -> (B, T, 672) float32, same column layout as features.batch_features."""
    B, T, F = X.shape
    diff = torch.cat([torch.zeros_like(X[:, :1]), X[:, 1:] - X[:, :-1]], dim=1)
    absdiff = diff.abs()
    states = {h: X[:, :1].clone() for h in EMA_SPANS}       # EMA_0 = x_0
    vol_state = torch.zeros_like(X[:, :1])                  # EMA of |diff| starts at |diff_0| = 0
    emas = {h: [] for h in EMA_SPANS}
    vols = []
    for c in range(0, T, chunk):
        xc, L = X[:, c:c + chunk], min(chunk, T - c)
        for h in EMA_SPANS:
            M, v = _decay_mats(h, L, X.device)
            e = torch.einsum('jk,bkf->bjf', M, xc) + v[None, :, None] * states[h]
            emas[h].append(e)
            states[h] = e[:, -1:]
        M, v = _decay_mats(VOL_SPAN, L, X.device)
        e = torch.einsum('jk,bkf->bjf', M, absdiff[:, c:c + chunk]) + v[None, :, None] * vol_state
        vols.append(e)
        vol_state = e[:, -1:]
    blocks = [X, diff] + [X - torch.cat(emas[h], dim=1) for h in EMA_SPANS] + [torch.cat(vols, dim=1)]
    return torch.cat(blocks, dim=-1)
