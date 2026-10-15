"""|y|-weighted ridge on causal features: raw, 1-step diff, x - EMA(10/100/1000), EMA100(|diff|).

Streaming equivalent of models/linear.py (feature_blocks/design); coefficients in coef.npy, shape (673, 2).
"""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

EMA_SPANS = (10, 100, 1000)
VOL_SPAN = 100


class PredictionModel:
    def __init__(self):
        self.coef = np.load(Path(__file__).resolve().parent / "coef.npy")
        self.alphas = np.array([2.0 / (h + 1) for h in EMA_SPANS])[:, None]
        self.vol_alpha = 2.0 / (VOL_SPAN + 1)
        self.features = np.ones(1 + 6 * 112)
        self.current_seq_ix = None
        self.previous_step = None

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        x = np.asarray(data_point.state, dtype=np.float32)
        if x.shape != (112,) or not np.isfinite(x).all():
            raise ValueError("expected 112 finite float32 features")
        if step == 0:
            diff = np.zeros(112, dtype=np.float32)
            self.ema = np.tile(x.astype(np.float64), (len(EMA_SPANS), 1))
            self.vol = np.zeros(112)
        else:
            diff = x - self.prev
            self.ema += self.alphas * (x - self.ema)
            self.vol += self.vol_alpha * (np.abs(diff) - self.vol)
        self.prev = x

        if not data_point.need_prediction:
            return None
        f = self.features
        f[1:113] = x
        f[113:225] = diff
        f[225:561] = (x - self.ema).ravel()
        f[561:673] = self.vol
        return (f @ self.coef).astype(np.float32)
