"""Streaming access to the competition parquet files, one sequence (row group) at a time."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'wnn_connectome_starterpack'
sys.path.insert(0, str(KIT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, WARMUP  # noqa: E402

TRAIN = KIT / 'datasets' / 'train.parquet'
VALID = KIT / 'datasets' / 'valid.parquet'
HOLDOUT_FILE = ROOT / 'splits' / 'holdout.json'
N_HOLDOUT = 800
HOLDOUT_SEED = 20261007


@dataclass
class Sequence:
    group: int
    seq_ix: int
    X: np.ndarray            # (20000, 112) float32
    y: np.ndarray            # (20000, 2) float32
    need: np.ndarray         # (20000,) bool
    scored: np.ndarray | None  # (20000,) bool, valid only


def read_sequence(pf: pq.ParquetFile, group: int) -> Sequence:
    t = pf.read_row_group(group, use_threads=False)
    X = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    need = t['need_prediction'].to_numpy().astype(bool)
    scored = t['is_scored'].to_numpy().astype(bool) if 'is_scored' in t.column_names else None
    return Sequence(group, int(t['seq_ix'][0].as_py()), X, y, need, scored)


def iter_sequences(path, groups=None):
    pf = pq.ParquetFile(path)
    for g in (range(pf.num_row_groups) if groups is None else groups):
        yield read_sequence(pf, g)


def holdout_groups() -> list[int]:
    """Fixed train row groups never used for training (validation view V3)."""
    if not HOLDOUT_FILE.exists():
        pf = pq.ParquetFile(TRAIN)
        rng = np.random.default_rng(HOLDOUT_SEED)
        groups = sorted(int(g) for g in rng.choice(pf.num_row_groups, N_HOLDOUT, replace=False))
        seq_ix = [int(pf.read_row_group(g, columns=['seq_ix'])['seq_ix'][0].as_py()) for g in groups]
        HOLDOUT_FILE.parent.mkdir(exist_ok=True)
        HOLDOUT_FILE.write_text(json.dumps({'seed': HOLDOUT_SEED, 'groups': groups, 'seq_ix': seq_ix}))
    return json.loads(HOLDOUT_FILE.read_text())['groups']


def train_groups() -> list[int]:
    held = set(holdout_groups())
    return [g for g in range(pq.ParquetFile(TRAIN).num_row_groups) if g not in held]
