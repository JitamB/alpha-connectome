"""Stateful ONNX submission: model.onnx with inputs (features, hidden_0..n) and outputs (prediction, next_hidden_0..n).

If features.py is bundled, each row is first turned into engineered features by its StreamingFeatures.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(Path(__file__).resolve().parent / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.hidden_names = [i.name for i in self.session.get_inputs() if i.name != "features"]
        self.hidden_shapes = [[d if isinstance(d, int) else 1 for d in i.shape]
                              for i in self.session.get_inputs() if i.name != "features"]
        here = Path(__file__).resolve().parent
        self.features = None
        if (here / "features.py").exists():
            sys.path.insert(0, str(here))
            from features import StreamingFeatures
            self.features = StreamingFeatures()
        self.current_seq_ix = None
        self.previous_step = None
        self.hidden = None

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self.hidden = [np.zeros(s, dtype=np.float32) for s in self.hidden_shapes]
            if self.features is not None:
                self.features.reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")
        if self.features is not None:
            state = self.features.update(state)
        feeds = {"features": state.reshape(1, 1, -1)}
        feeds.update(zip(self.hidden_names, self.hidden))
        prediction, *self.hidden = self.session.run(None, feeds)
        if not data_point.need_prediction:
            return None
        return np.asarray(prediction[0, 0], dtype=np.float32)
