"""Average of K GRU models run by the bundled numba kernel (fast_gru.py) with weights from weights.npz."""
from pathlib import Path
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fast_gru import FastEnsemble  # noqa: E402


class PredictionModel:
    def __init__(self):
        with np.load(HERE / "weights.npz") as w:
            self.model = FastEnsemble({k: w[k] for k in w.files})
        self.model.update(np.zeros(112, dtype=np.float32))  # compile the kernel once up front
        self.model.reset()
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
            self.model.reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")
        out = self.model.update(state)
        if not data_point.need_prediction:
            return None
        return out.copy()
