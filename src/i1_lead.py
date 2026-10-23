"""i1 lead-lag lookup features (training/evaluation side; numpy).

An i1 transition is a change of the exact i1 price shape (the 22 float32 i1 price columns, hashed like
shape_lookup.shape_ids). A transition type is the ordered pair (previous shape, new shape). The table
(tools/build_i1_table.py) maps a type to the mean future change of the clipped i0 targets over the next H rows,
shrunk by SHRINK. features() returns, in target units, the causal window sums (10 / 100 / 1000 rows) of the looked-up
values at i1 transition rows: 3 windows x 2 targets = N_FEAT columns. This is the 'i1enc' group of tools/gate_paths.py
with class-free target units.
"""
import numpy as np

from shape_lookup import shape_ids

PRICE_I1 = slice(52, 74)
H = 100
SHRINK = 20.0
WINDOWS = (10, 100, 1000)
N_FEAT = 2 * len(WINDOWS)
PAIR = np.uint64(0x9E3779B97F4A7C15)
N_FOLDS = 4            # table k is built without train groups g with g % N_FOLDS == k; table N_FOLDS = all of them


def transition_types(P):
    """(T, 22) i1 prices -> (flag (T,) bool, type (T,) uint64); flag[t] is True when the shape changed at row t."""
    ids = shape_ids(P)
    tr = np.r_[False, ids[1:] != ids[:-1]]
    typ = np.zeros(len(ids), np.uint64)
    with np.errstate(over='ignore'):
        typ[1:] = ids[:-1] * PAIR + ids[1:]
    return tr, typ


def future_change(y):
    """(T, 2) targets -> change of the clipped targets over the next H rows (the last rows look at row T - 1)."""
    yc = np.clip(y, -2.0, 2.0).astype(np.float64)
    return yc[np.minimum(np.arange(len(yc)) + H, len(yc) - 1)] - yc


def window_sums(v, w):
    c = np.cumsum(v, 0)
    out = c.copy()
    out[w:] -= c[:-w]
    return out


def features(X, keys, vals):
    """(T, >=74) raw rows -> (T, N_FEAT) float32; keys sorted uint64 (n,), vals (n, 2)."""
    tr, typ = transition_types(X[:, PRICE_I1])
    pos = np.clip(np.searchsorted(keys, typ), 0, len(keys) - 1)
    hit = keys[pos] == typ
    v = np.where((hit & tr)[:, None], vals[pos].astype(np.float64), 0.0)
    return np.concatenate([window_sums(v, w) for w in WINDOWS], 1).astype(np.float32)


def load_tables(path):
    """-> list of N_FOLDS + 1 (keys, vals) tables."""
    z = np.load(path)
    return [(z[f'keys{k}'], z[f'vals{k}']) for k in range(N_FOLDS + 1)]


def fold_of(group, holdout):
    """Table index for a train row group; holdout and every validation sequence use the full table."""
    return N_FOLDS if group in holdout else group % N_FOLDS
