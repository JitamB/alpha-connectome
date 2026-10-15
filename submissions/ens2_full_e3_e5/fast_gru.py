"""numba inference for an average of K GRUModels (features -> projection -> 2 GRU layers -> head + skip head).

Bundled into submissions. One compiled call per row computes the engineered features once (same
definition as features.py: raw, diff, x - EMA10/100/1000, EMA100(|diff|)) and steps every model.
Input standardisation is folded into the projection and skip-head weights by pack().
"""
import numpy as np
from numba import njit

EMA_SPANS = (10, 100, 1000)
VOL_SPAN = 100
WEIGHT_NAMES = ('Wp', 'bp', 'Wih0', 'bih0', 'Whh0', 'bhh0', 'Wih1', 'bih1', 'Whh1', 'bhh1', 'Wo', 'bo', 'Ws', 'bs')


def pack(state_dicts):
    """List of GRUModel state_dicts (numpy float arrays) -> dict of stacked float32 arrays (leading axis K)."""
    out = {k: [] for k in WEIGHT_NAMES}
    for sd in state_dicts:
        inv = 1.0 / sd['std'].astype(np.float64)
        mean = sd['mean'].astype(np.float64)
        Wp = sd['proj.weight'].astype(np.float64) * inv
        Ws = sd['skip_head.weight'].astype(np.float64) * inv
        vals = {
            'Wp': Wp, 'bp': sd['proj.bias'] - Wp @ mean,
            'Ws': Ws, 'bs': sd['skip_head.bias'] - Ws @ mean,
            'Wo': sd['reg_head.weight'], 'bo': sd['reg_head.bias'],
        }
        for i in range(2):
            vals[f'Wih{i}'] = sd[f'blocks.{i}.gru.weight_ih_l0']
            vals[f'bih{i}'] = sd[f'blocks.{i}.gru.bias_ih_l0']
            vals[f'Whh{i}'] = sd[f'blocks.{i}.gru.weight_hh_l0']
            vals[f'bhh{i}'] = sd[f'blocks.{i}.gru.bias_hh_l0']
        for k in WEIGHT_NAMES:
            out[k].append(np.asarray(vals[k], dtype=np.float32))
    return {k: np.ascontiguousarray(np.stack(v)) for k, v in out.items()}


@njit(fastmath=True)
def _matvec(W, x, b, out):
    out[:] = np.dot(W, x) + b      # BLAS sgemv; ~1.3x faster than explicit loops on the M2


@njit(fastmath=True)
def _gru(Wih, bih, Whh, bhh, x, h, gi, gh):
    H = h.shape[0]
    _matvec(Wih, x, bih, gi)
    _matvec(Whh, h, bhh, gh)
    for i in range(H):
        r = 1.0 / (1.0 + np.exp(-(gi[i] + gh[i])))
        z = 1.0 / (1.0 + np.exp(-(gi[H + i] + gh[H + i])))
        n = np.tanh(gi[2 * H + i] + r * gh[2 * H + i])
        h[i] = n + z * (h[i] - n)


@njit(fastmath=True)
def step(x, first, prev, ema, vol, alphas, vol_alpha, f,
         Wp, bp, Wih0, bih0, Whh0, bhh0, Wih1, bih1, Whh1, bhh1, Wo, bo, Ws, bs,
         h0, h1, z, gi, gh, out):
    """Update features + all K models with row x (112,), write the averaged prediction into out (2,)."""
    n = x.shape[0]
    for j in range(n):
        xj = np.float64(x[j])
        if first:
            d = 0.0
            for s in range(ema.shape[0]):
                ema[s, j] = xj
            vol[j] = 0.0
        else:
            d = xj - prev[j]
            for s in range(ema.shape[0]):
                ema[s, j] += alphas[s] * (xj - ema[s, j])
            vol[j] += vol_alpha * (abs(d) - vol[j])
        prev[j] = xj
        f[j] = xj
        f[n + j] = d
        for s in range(ema.shape[0]):
            f[(2 + s) * n + j] = xj - ema[s, j]
        f[(2 + ema.shape[0]) * n + j] = vol[j]
    K = Wp.shape[0]
    out[0] = 0.0
    out[1] = 0.0
    for k in range(K):
        _matvec(Wp[k], f, bp[k], z)
        _gru(Wih0[k], bih0[k], Whh0[k], bhh0[k], z, h0[k], gi, gh)
        _gru(Wih1[k], bih1[k], Whh1[k], bhh1[k], h0[k], h1[k], gi, gh)
        for t in range(2):
            acc = bo[k, t] + bs[k, t]
            for i in range(h1.shape[1]):
                acc += Wo[k, t, i] * h1[k, i]
            for j in range(f.shape[0]):
                acc += Ws[k, t, j] * f[j]
            out[t] += acc
    out[0] /= K
    out[1] /= K


class FastEnsemble:
    """Row-by-row runner around step(); call reset() at the start of every sequence."""

    def __init__(self, weights):
        self.w = [weights[k] for k in WEIGHT_NAMES]
        K, H = weights['Whh0'].shape[0], weights['Whh0'].shape[2]
        P = weights['Wp'].shape[1]
        self.alphas = np.array([2.0 / (h + 1) for h in EMA_SPANS])
        self.vol_alpha = 2.0 / (VOL_SPAN + 1)
        self.prev = np.zeros(112)
        self.ema = np.zeros((len(EMA_SPANS), 112))
        self.vol = np.zeros(112)
        self.f = np.zeros(112 * (3 + len(EMA_SPANS)), dtype=np.float32)
        self.h0 = np.zeros((K, H), dtype=np.float32)
        self.h1 = np.zeros((K, H), dtype=np.float32)
        self.z = np.zeros(P, dtype=np.float32)
        self.gi = np.zeros(3 * H, dtype=np.float32)
        self.gh = np.zeros(3 * H, dtype=np.float32)
        self.out = np.zeros(2, dtype=np.float32)
        self.first = True

    def reset(self):
        self.h0[:] = 0.0
        self.h1[:] = 0.0
        self.first = True

    def update(self, x):
        step(x, self.first, self.prev, self.ema, self.vol, self.alphas, self.vol_alpha, self.f, *self.w,
             self.h0, self.h1, self.z, self.gi, self.gh, self.out)
        self.first = False
        return self.out
