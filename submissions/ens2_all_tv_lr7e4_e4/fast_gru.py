"""numba inference for an average of K GRUModels (features -> projection -> 2 GRU layers -> head + skip head).

Bundled into submissions. One compiled call per row computes the engineered features once (same
definition as features.py: raw, diff, x - EMA10/100/1000, EMA100(|diff|)) and steps every model.
Input standardisation is folded into the projection and skip-head weights by pack().
pack(..., bf16=True) stores the five large matrices (projection, GRU input / hidden weights) as bfloat16 bits
(uint16): half the weight bytes per row, widened to fp32 inside explicit loops; heads and biases stay fp32.
"""
import numpy as np
from numba import njit

EMA_SPANS = (10, 100, 1000)
VOL_SPAN = 100
BF16_NAMES = ('Wp', 'Wih0', 'Whh0', 'Wih1', 'Whh1')
WEIGHT_NAMES = ('Wp', 'bp', 'Wih0', 'bih0', 'Whh0', 'bhh0', 'Wih1', 'bih1', 'Whh1', 'bhh1', 'Wo', 'bo', 'Ws', 'bs',
                'Wd', 'bd', 'atoms', 'alpha')


def to_bf16(a):
    """float32 array -> uint16 bfloat16 bits (round to nearest even)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def pack(state_dicts, atoms=None, alpha=0.0, bf16=False):
    """List of GRUModel state_dicts (numpy float arrays) -> dict of stacked float32 arrays (leading axis K).
    alpha > 0 mixes in the decoded distributional head: output = (1 - alpha) * regression + alpha * decoded.
    bf16: store BF16_NAMES as bfloat16 bits; the projection bias is folded with the rounded projection."""
    out = {k: [] for k in WEIGHT_NAMES[:14]}
    for sd in state_dicts:
        inv = 1.0 / sd['std'].astype(np.float64)
        mean = sd['mean'].astype(np.float64)
        Wp = sd['proj.weight'].astype(np.float64) * inv
        if bf16:
            Wp = from_bf16(to_bf16(Wp)).astype(np.float64)
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
        for k in WEIGHT_NAMES[:14]:
            out[k].append(np.asarray(vals[k], dtype=np.float32))
    out = {k: np.ascontiguousarray(np.stack(v)) for k, v in out.items()}
    if bf16:
        out.update({k: to_bf16(out[k]) for k in BF16_NAMES})
    use = alpha > 0
    out['Wd'] = np.ascontiguousarray(np.stack([sd['dist_head.weight'] for sd in state_dicts]).astype(np.float32)) if use \
        else np.zeros((len(state_dicts), 0, 1), np.float32)
    out['bd'] = np.ascontiguousarray(np.stack([sd['dist_head.bias'] for sd in state_dicts]).astype(np.float32)) if use \
        else np.zeros((len(state_dicts), 0), np.float32)
    out['atoms'] = np.asarray(atoms if use else [], dtype=np.float32)
    out['alpha'] = np.asarray([alpha], dtype=np.float32)
    return out


@njit(fastmath=True)
def _matvec(W, x, b, out, buf):
    np.dot(W, x, out)              # BLAS sgemv into out (no temporaries); ~1.3x faster than explicit loops on the M2
    for i in range(out.shape[0]):
        out[i] += b[i]


@njit(fastmath=True)
def _matvec_bf16(W, x, b, out, buf):
    """W: uint16 bfloat16 bits; each row is widened into the uint32 scratch buf (viewed as fp32), then dotted."""
    n = x.shape[0]
    fv = buf[:n].view(np.float32)
    for i in range(W.shape[0]):
        for j in range(n):
            buf[j] = np.uint32(W[i, j]) << np.uint32(16)
        acc = np.float32(0.0)
        for j in range(n):
            acc += fv[j] * x[j]
        out[i] = acc + b[i]


@njit(fastmath=True)
def _gru(mv, Wih, bih, Whh, bhh, x, h, gi, gh, buf):
    H = h.shape[0]
    mv(Wih, x, bih, gi, buf)
    mv(Whh, h, bhh, gh, buf)
    for i in range(H):
        r = 1.0 / (1.0 + np.exp(-(gi[i] + gh[i])))
        z = 1.0 / (1.0 + np.exp(-(gi[H + i] + gh[H + i])))
        n = np.tanh(gi[2 * H + i] + r * gh[2 * H + i])
        h[i] = n + z * (h[i] - n)


@njit(fastmath=True)
def _decode(logits, atoms):
    """Softmax over atoms -> E[|y| y] / E[|y|]."""
    m = logits.max()
    num = 0.0
    den = 0.0
    for i in range(atoms.shape[0]):
        w = np.exp(logits[i] - m) * abs(atoms[i])
        num += w * atoms[i]
        den += w
    return num / max(den, 1e-30)


@njit(fastmath=True)
def step(mv, x, first, prev, ema, vol, alphas, vol_alpha, f,
         Wp, bp, Wih0, bih0, Whh0, bhh0, Wih1, bih1, Whh1, bhh1, Wo, bo, Ws, bs, Wd, bd, atoms, alpha,
         h0, h1, z, gi, gh, dl, buf, out):
    """Update features + all K models with row x (112,), write the averaged prediction into out (2,).
    mv: matrix-vector kernel for the large matrices (_matvec for fp32, _matvec_bf16 for bfloat16 bits)."""
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
        mv(Wp[k], f, bp[k], z, buf)
        _gru(mv, Wih0[k], bih0[k], Whh0[k], bhh0[k], z, h0[k], gi, gh, buf)
        _gru(mv, Wih1[k], bih1[k], Whh1[k], bhh1[k], h0[k], h1[k], gi, gh, buf)
        a = alpha[0]
        A = atoms.shape[0]
        if a > 0:
            _matvec(Wd[k], h1[k], bd[k], dl, buf)
        for t in range(2):
            acc = bo[k, t] + bs[k, t]
            for i in range(h1.shape[1]):
                acc += Wo[k, t, i] * h1[k, i]
            for j in range(f.shape[0]):
                acc += Ws[k, t, j] * f[j]
            if a > 0:
                acc = (1.0 - a) * acc + a * _decode(dl[t * A:(t + 1) * A], atoms)
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
        self.dl = np.zeros(max(1, weights['Wd'].shape[1]), dtype=np.float32)
        self.out = np.zeros(2, dtype=np.float32)
        self.buf = np.zeros(max(self.f.shape[0], P, H), dtype=np.uint32)
        self.mv = _matvec_bf16 if weights['Wp'].dtype == np.uint16 else _matvec
        self.first = True

    def reset(self):
        self.h0[:] = 0.0
        self.h1[:] = 0.0
        self.first = True

    def update(self, x):
        step(self.mv, x, self.first, self.prev, self.ema, self.vol, self.alphas, self.vol_alpha, self.f, *self.w,
             self.h0, self.h1, self.z, self.gi, self.gh, self.dl, self.buf, self.out)
        self.first = False
        return self.out
