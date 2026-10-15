"""Causal engineered features: one implementation for training (whole sequence) and submission (row by row).

Bundled into submissions as features.py. Blocks (112 columns each, in this order):
raw x, 1-step diff, x - EMA10, x - EMA100, x - EMA1000, EMA100(|diff|)  -> 672 columns.
EMAs start at the first row (EMA_0 = x_0); diff_0 = 0.
"""
import numpy as np

EMA_SPANS = (10, 100, 1000)
VOL_SPAN = 100
N_FEATURES = 112 * (3 + len(EMA_SPANS))


def _alpha(span):
    return 2.0 / (span + 1)


def batch_features(X):
    """X: (T, 112) float32 -> (T, 672) float32, computed over a whole sequence."""
    from scipy.signal import lfilter

    def ema(x, span):
        a = _alpha(span)
        return lfilter([a], [1, a - 1], x, axis=0, zi=(1 - a) * x[:1])[0]

    X = X.astype(np.float64)
    diff = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])
    blocks = [X, diff] + [X - ema(X, h) for h in EMA_SPANS] + [ema(np.abs(diff), VOL_SPAN)]
    return np.hstack(blocks).astype(np.float32)


class StreamingFeatures:
    """Row-by-row equivalent of batch_features; call reset() at the start of every sequence."""

    def __init__(self):
        self.alphas = np.array([_alpha(h) for h in EMA_SPANS])[:, None]
        self.vol_alpha = _alpha(VOL_SPAN)
        self.out = np.zeros(N_FEATURES, dtype=np.float32)
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, x):
        """x: (112,) float32 -> (672,) float32 view (overwritten on the next call)."""
        x = x.astype(np.float64)
        if self.prev is None:
            diff = np.zeros(112)
            self.ema = np.tile(x, (len(EMA_SPANS), 1))
            self.vol = np.zeros(112)
        else:
            diff = x - self.prev
            self.ema += self.alphas * (x - self.ema)
            self.vol += self.vol_alpha * (np.abs(diff) - self.vol)
        self.prev = x
        out = self.out
        out[:112] = x
        out[112:224] = diff
        out[224:560] = (x - self.ema).ravel()
        out[560:] = self.vol
        return out
