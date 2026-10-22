"""Exact i0 price-shape lookup (bundled into submissions; numba).

A shape is the exact float32 bit pattern of the 22 i0 price-like columns, hashed to uint64 exactly as
tools/gate_paths.shape_ids does (wrap-around arithmetic). The table (tools/build_shape_table.py) maps a hash to
statistics of the targets on train rows with that shape: count, sum S*step, sum D*step, row counts per S bucket
(S <= -3, -2, -1, >= 0; S = n0 + n1, D = n0 - n1 in ticks) and the majority asset class.
features() returns, in target units: hit, log1p(count), E[S*step], E[D*step], P(bucket) x 4 (shrunk to the prior).
"""
import numpy as np
from numba import njit

MULT = np.random.default_rng(12345).integers(1, 2 ** 63, 22, dtype=np.uint64) | np.uint64(1)
N_FEAT = 8
SHRINK = 20.0


def shape_ids(P):
    """numpy version, (T, 22) float32 -> (T,) uint64 (identical to tools/gate_paths.shape_ids)."""
    u = np.ascontiguousarray(P, dtype=np.float32).view(np.uint32).astype(np.uint64)
    with np.errstate(over='ignore'):
        return (u * MULT).sum(1)


@njit(cache=False)
def shape_hash(x, mult):
    """x: (>=22,) float32 row -> uint64 hash of its first 22 values."""
    u = x[:22].copy().view(np.uint32)
    h = np.uint64(0)
    for j in range(22):
        h += np.uint64(u[j]) * mult[j]
    return h


@njit(cache=False)
def lookup(h, keys, stats, cls, prior, out):
    """Fill out (N_FEAT,) for hash h. prior: (5, 6) rows = class 0..3 and global, columns = E[S*step], E[D*step],
    P(bucket) x 4; a hit shrinks to its own class prior, a miss gets the global prior. Returns the row index or -1."""
    i = np.searchsorted(keys, h)
    hit = i < keys.shape[0] and keys[i] == h
    cnt = stats[i, 0] if hit else 0.0
    pr = prior[cls[i]] if hit else prior[4]
    out[0] = 1.0 if hit else 0.0
    out[1] = np.log1p(cnt)
    den = cnt + SHRINK
    for k in range(6):
        s = stats[i, k + 1] if hit else 0.0
        out[2 + k] = (s + SHRINK * pr[k]) / den
    return i if hit else -1


def features(X, keys, stats, cls, prior):
    """Batch version for training / evaluation: (T, >=22) -> (T, N_FEAT) float32 and (T,) row index (-1 = miss)."""
    sid = shape_ids(X[:, :22])
    pos = np.clip(np.searchsorted(keys, sid), 0, len(keys) - 1)
    hit = keys[pos] == sid
    st = np.where(hit[:, None], stats[pos].astype(np.float64), 0.0)
    pr = np.where(hit[:, None], prior[np.minimum(cls[pos], 3)], prior[4])
    cnt = st[:, 0]
    den = cnt + SHRINK
    f = np.column_stack([hit.astype(np.float64), np.log1p(cnt)] +
                        [(st[:, k + 1] + SHRINK * pr[:, k]) / den for k in range(6)])
    return f.astype(np.float32), np.where(hit, pos, -1)
