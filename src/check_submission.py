"""Pre-upload checks for a submission directory or ZIP.

  python src/check_submission.py submissions/<name>      # zips the directory to submissions/<name>.zip, then checks it
  python src/check_submission.py path/to/submission.zip

Checks: ZIP layout and size, imports available in the scorer, output contract, determinism,
no state leakage between sequences, causality, batch-ONNX equivalence, projected scorer runtime.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from data import KIT, VALID, WARMUP, iter_sequences
from evaluate import OnnxSequencePredictor

sys.path.insert(0, str(KIT))
from utils import DataPoint  # noqa: E402

MAX_ZIP_MB = 20
SCORER_SLOWDOWN = 1.55          # scorer s/row divided by M2 s/row; measured 1.51 (baseline), 1.63 (linear), 1.54 (gru_eng_p64_s2000)
TEST_ROWS = 39_400_000
MAX_PROJECTED_MIN = 48
SCORER_PACKAGES = {
    'numpy', 'scipy', 'pandas', 'pyarrow', 'fastparquet', 'polars', 'torch', 'torchvision', 'torchaudio',
    'lightning', 'torchmetrics', 'tensorflow', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'statsmodels',
    'numba', 'joblib', 'transformers', 'accelerate', 'datasets', 'tokenizers', 'sentencepiece', 'jax', 'flax',
    'orbax', 'tqdm', 'einops', 'rich', 'loguru', 'pydantic', 'hydra', 'omegaconf', 'yaml', 'dotenv', 'optuna',
    'mlflow', 'wandb', 'matplotlib', 'seaborn', 'onnxruntime', 'onnx', 'pytest', 'darts', 'neuralforecast',
    'rtdl_num_embeddings', 'utils',
}


def make_zip(directory: Path) -> Path:
    out = directory.with_suffix('.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(directory.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.zip':
                z.write(f, f.relative_to(directory))
    return out


def load_model_class(directory: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f'submission_{tag}', directory / 'solution.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    spec.loader.exec_module(module)
    return module.PredictionModel


def run_rows(model, seq):
    """Feed one sequence row by row like the scorer; return (20000, 2) predictions (NaN on warm-up)."""
    preds = np.full((len(seq.X), 2), np.nan, dtype=np.float32)
    for i in range(len(seq.X)):
        value = model.predict(DataPoint(seq.seq_ix, i, bool(seq.need[i]), seq.X[i]))
        if not seq.need[i]:
            if value is not None:
                raise AssertionError(f'non-None output on warm-up step {i}')
        else:
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise AssertionError(f'bad output {value!r} at step {i}')
            preds[i] = value
    return preds


def check(path: Path, n_seqs: int):
    results = []

    def report(name, ok, detail=''):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    zip_path = make_zip(path) if path.is_dir() else path
    size_mb = zip_path.stat().st_size / 2**20
    names = zipfile.ZipFile(zip_path).namelist()
    print(f'== {zip_path}')
    report('solution.py at ZIP root', 'solution.py' in names)
    report(f'ZIP size <= {MAX_ZIP_MB} MB', size_mb <= MAX_ZIP_MB, f'{size_mb:.2f} MB')

    work = Path(tempfile.mkdtemp())
    zipfile.ZipFile(zip_path).extractall(work)
    local = {p.stem for p in work.glob('*.py')} | {p.name for p in work.iterdir() if p.is_dir()}
    imports = set()
    for py in work.rglob('*.py'):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                imports |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports.add(node.module.split('.')[0])
    missing = sorted(imports - SCORER_PACKAGES - set(sys.stdlib_module_names) - local)
    report('imports available in scorer', not missing, ', '.join(missing))

    Model = load_model_class(work, 'a')
    seqs = list(iter_sequences(VALID, range(n_seqs)))

    model = Model()
    t0 = time.perf_counter()
    try:
        first = [run_rows(model, s) for s in seqs]
    except AssertionError as e:
        report('output contract (None in warm-up, finite (2,) after)', False, str(e))
        print('SOME CHECKS FAILED')
        return False
    us_per_row = (time.perf_counter() - t0) / (n_seqs * len(seqs[0].X)) * 1e6
    report('output contract (None in warm-up, finite (2,) after)', True)

    repeat = Model()
    second = [run_rows(repeat, s) for s in seqs]
    report('deterministic across instances', all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, second)))

    reverse = Model()
    rev = [run_rows(reverse, s) for s in reversed(seqs)][::-1]
    report('no state leakage between sequences (reversed order)',
           all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, rev)))

    k = 10_000
    s = seqs[0]
    perturbed = type(s)(s.group, s.seq_ix, s.X.copy(), s.y, s.need, s.scored)
    perturbed.X[k:] += np.random.default_rng(0).normal(0, 1, perturbed.X[k:].shape).astype(np.float32)
    p = run_rows(Model(), perturbed)
    report(f'causal (rows >= {k} perturbed, earlier outputs unchanged)',
           np.array_equal(p[:k], first[0][:k], equal_nan=True) and not np.array_equal(p[k:], first[0][k:]))

    onnx_files = sorted(work.glob('*.onnx'))
    if onnx_files:
        batch = OnnxSequencePredictor(onnx_files[0])(np.stack([s.X for s in seqs]))
        diff = max(np.abs(b[WARMUP:] - f[WARMUP:]).max() for b, f in zip(batch, first))
        report('row-by-row == batch evaluator (max abs diff <= 1e-5)', diff <= 1e-5, f'{diff:.2e} ({onnx_files[0].name})')

    projected = us_per_row * 1e-6 * SCORER_SLOWDOWN * TEST_ROWS / 60
    report(f'projected scorer runtime <= {MAX_PROJECTED_MIN} min', projected <= MAX_PROJECTED_MIN,
           f'{us_per_row:.1f} us/row on this machine -> ~{projected:.0f} min on scorer')
    print('ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED')
    return all(results)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=Path)
    ap.add_argument('--seqs', type=int, default=6)
    args = ap.parse_args()
    sys.exit(0 if check(args.path, args.seqs) else 1)
