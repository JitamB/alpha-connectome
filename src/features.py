"""Causal engineered features: one implementation for training (whole sequence) and submission (row by row).

Bundled into submissions as features.py. Blocks (112 columns each, in this order):
raw x, 1-step diff, x - EMA10, x - EMA100, x - EMA1000, EMA100(|diff|)  -> 672 columns.
The optional clock extension appends 10 causal elapsed-time and transition-rate features.
EMAs start at the first row (EMA_0 = x_0); diff_0 = 0.
"""
import numpy as np

EMA_SPANS = (10, 100, 1000)
VOL_SPAN = 100
N_FEATURES = 112 * (3 + len(EMA_SPANS))
CLOCK_FEATURES = 10


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


def clock_features(X):
    """Causal timing representation of exact book/trade transitions plus row position.

    These make elapsed row time explicit, so a recurrent model need not preserve a precise
    counter across thousands of rows.  The four timers use i0/i1 price-book changes and the
    two trade columns used in the earlier event-clock diagnostic.  Transition rates are the
    inclusive trailing 100/1000 row means.
    """
    T = len(X)
    t = np.arange(T)

    def changed(cols):
        return np.r_[False, np.any(X[1:, cols] != X[:-1, cols], axis=1)]

    def since(flag):
        last = np.maximum.accumulate(np.where(flag, t, 0))
        return np.log1p(t - last)

    def rate(flag, width):
        cs = np.r_[0.0, np.cumsum(flag, dtype=np.float64)]
        lo = np.maximum(0, t + 1 - width)
        return (cs[t + 1] - cs[lo]) / width

    tr0, tr1 = changed(slice(0, 22)), changed(slice(52, 74))
    td0, td1 = changed([44, 48]), changed([96, 100])
    pos = t / max(T - 1, 1)
    return np.stack([since(tr0), since(tr1), since(td0), since(td1),
                     rate(tr0, 100), rate(tr1, 100), rate(tr0, 1000), rate(tr1, 1000),
                     pos, pos * pos], 1).astype(np.float32)


def batch_features_clock(X):
    """Engineered features followed by the causal clock extension."""
    return np.hstack([batch_features(X), clock_features(X)]).astype(np.float32)


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
