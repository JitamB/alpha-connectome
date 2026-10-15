# Alpha Connectome ML Competition: Execution Summary

Condensed from `connectome_execution.md` (Quick Start, Data Overview, Submission Guide, Rules, Timeline, Prizes, FAQ, Get Help). Items marked **[my calc]** or **[my note]** are my own derivations or observations, not text from the docs.

---

## 1. The task in one paragraph

Each **sequence** is 20,000 consecutive rows. Each row is one moment in time for two trading instruments, `i0` and `i1`, described by **112 float32 features**. You predict two anonymized indicators of the future price movement of `i0`: **`t0` and `t1`**. Predictions are required at steps **99 to 19,999** of every sequence (steps 0 to 98 are warm-up). Your model is called **row by row, in order**, and must be **causal** (current and earlier rows of the same sequence only). Score = **Global Weighted Pearson (WP)**, averaged over `t0` and `t1`. Higher is better.

---

## 2. Key dates and deadlines (all at 11:59 PM UTC)

| Date | Event |
| --- | --- |
| SEP 11, 2026 | Competition starts |
| **NOV 15, 2026** | **Final submission deadline.** No new submissions after this |
| DEC 01, 2026 | Winner announcement |

- Near the end you will be asked to **choose some of your previously submitted and scored solutions** to be re-scored on the Final Dataset. Only already-scored submissions can be picked, so keep your best candidates submitted well before the deadline.
- Prize verification window: **7 calendar days** after the preliminary results are announced.
- **[my calc]** As of Oct 6, 2026 there are about 40 days to the final deadline.

---

## 3. Quick-start workflow

1. **Get the starter pack**
   ```bash
   curl -L https://files.wundernn.io/wnn_connectome_starterpack.tar.gz | tar -xz && cd wnn_connectome_starterpack
   ```
   - Manual downloads: `wnn_connectome_starterpack.zip` / `.tar.gz` at `https://files.wundernn.io/`
   - Mirrors for slow downloads:
     - `https://files-alternative.wundernn.io/wnn_connectome_starterpack.zip`
     - `https://storage.yandexcloud.net/files-alternative.wundernn.io/wnn_connectome_starterpack.zip`
2. **Set up a virtual environment** (any Python 3.10+ should work; the scorer uses **Python 3.11**) and run `pip install -r requirements.txt`.
3. **Explore the data** (Parquet).
4. **Run the baseline**
   ```bash
   python baseline/solution.py --validation datasets/valid.parquet
   ```
   It loads the validation data and the pre-trained `baseline.onnx`, runs inference, and prints the **Global Weighted Pearson** score.
5. **Submit the baseline** (sanity check of the pipeline)
   ```bash
   cd baseline
   zip -r submission.zip solution.py baseline.onnx
   ```
   Upload at **https://wundernn.io/connectome/submit**.
6. **Build your own model:** write your own `solution.py` implementing `PredictionModel`, train on `train.parquet`, validate on `valid.parquet`.

### Starter pack contents

```
wnn_connectome_starterpack/
├── baseline/        README.md, baseline.onnx, baseline_submission.zip, solution.py
├── datasets/        train.parquet, valid.parquet, valid_mask.parquet
├── docs/
├── METRIC.md        <- full scoring reference, read this
├── README.md
├── requirements.txt
└── utils.py         <- contains DataPoint
```

### Baseline facts

- Stateful vanilla **GRU**.
- Scores **0.617052 WP** on the complete validation set. This is your reference number to beat.
- ONNX Runtime session uses **one CPU compute thread**, matching the 1-core limit.
- Leaderboard as of 6th Oct: 1st - 0.6711 | 2nd - 0.6696 | 3rd - 0.6685 | ... | 7th - 0.6674 | 8th - 0.6673

---

## 4. Data

### Files

| File | Sequences | Rows (Seq × 20,000) | Columns |
| --- | --- | --- | --- |
| `train.parquet` | 10,607 | 212,140,000 | 117: `seq_ix, step_in_seq, need_prediction, <112 features>, t0, t1` |
| `valid.parquet` | 1,873 | 37,460,000 | 118: `seq_ix, step_in_seq, need_prediction, is_scored, <112 features>, t0, t1` |
| `valid_mask.parquet` | 1,873 | 37,460,000 | 3: `seq_ix, step_in_seq, is_scored` |

- `valid_mask.parquet` is identical to the `is_scored` column of `valid.parquet`. Join on (`seq_ix`, `step_in_seq`).
- **Test data** has the same format and structure as `valid.parquet` but is never distributed. The platform feeds it to your `predict` callback row by row. The test set has **1,970 sequences × 20,000 rows = 39,400,000 rows**. Test targets and the scoring mask are hidden.
- **Reading tip:** each Parquet row group is exactly one sequence (20,000 rows, a single `seq_ix`). Use `pq.ParquetFile(path).read_row_group(i)` to stream one sequence at a time.
  ```python
  import pyarrow.parquet as pq
  dataset = pq.ParquetFile('datasets/train.parquet')
  sequence = dataset.read_row_group(0)
  print(sequence.shape); print(sequence.column_names)
  ```
- **[my calc]** Full train as float32 features is roughly 95 GB (212.14M × 112 × 4 B), and valid roughly 17 GB. Plan on streaming by row group, not loading whole files into the 16 GB RAM.

### Column semantics

| Column | train | valid | valid_mask | Meaning |
| --- | --- | --- | --- | --- |
| `seq_ix` | ✓ | ✓ | ✓ | Sequence id. **Not a feature.** A larger id does not mean later in time |
| `step_in_seq` | ✓ | ✓ | ✓ | Row position, 0 to 19,999 |
| `need_prediction` | ✓ | ✓ | × | False on steps 0 to 98, true on 99 to 19,999 |
| `is_scored` | × | ✓ | ✓ | True on rows used for evaluation |
| 112 features | ✓ | ✓ | × | float32 |
| `t0`, `t1` | ✓ | ✓ | × | float32 targets for `i0` only |

### Sequence rules for your model

- Process rows **in the given order**, never drop rows, and **update state on every row**, including warm-up rows and rows where `is_scored` is false.
- **Reset state when `seq_ix` changes.** Sequences are independent. Do not join them or carry state across them.
- `seq_ix` is shuffled. **Do not use `seq_ix` as a feature or to order sequences.**
- A prediction may use **only the current row and earlier rows of the same sequence** (no lookahead).
- On every row with `need_prediction = True` you must output **two finite values**. You can't know which rows are scored at test time, so predict on all of them as if they were scored.

### The 112 features (exact order, same in files and in `state`)

Names follow `i<instrument>_<group><index>`. Groups: `p` price-like, `v` volume-like, `dp` trade price, `dv` trade volume. The data holds **L11 order books** (top-11 bid and ask levels, prices and volumes) for 2 instruments, plus a trade history, plus 8 additional features.

| Group | Columns | Count | Index in `state` |
| --- | --- | --- | --- |
| `i0` bid price-like | `i0_p0` … `i0_p10` | 11 | 0–10 |
| `i0` ask price-like | `i0_p11` … `i0_p21` | 11 | 11–21 |
| `i0` bid volume-like | `i0_v0` … `i0_v10` | 11 | 22–32 |
| `i0` ask volume-like | `i0_v11` … `i0_v21` | 11 | 33–43 |
| `i0` trade price | `i0_dp0` … `i0_dp3` | 4 | 44–47 |
| `i0` trade volume | `i0_dv0` … `i0_dv3` | 4 | 48–51 |
| `i1` bid price-like | `i1_p0` … `i1_p10` | 11 | 52–62 |
| `i1` ask price-like | `i1_p11` … `i1_p21` | 11 | 63–73 |
| `i1` bid volume-like | `i1_v0` … `i1_v10` | 11 | 74–84 |
| `i1` ask volume-like | `i1_v11` … `i1_v21` | 11 | 85–95 |
| `i1` trade price | `i1_dp0` … `i1_dp3` | 4 | 96–99 |
| `i1` trade volume | `i1_dv0` … `i1_dv3` | 4 | 100–103 |
| Additional | `a0` … `a7` | 8 | 104–111 |

**Pitfalls**
- **Do NOT sort column names alphabetically.** That puts `i0_p10` right after `i0_p1`. Use numeric order, or just use the file's column order.
- Bid vs ask is told apart **by index only** (0–10 bid, 11–21 ask).
- **The index is an identifier, not a depth level.** `p0` is not necessarily the best bid, and `p10` is not necessarily the deepest level. Don't assume any ordering by price or by distance from the best quote. The index only guarantees the same component sits in the same column on every row.
- Features are deliberately anonymized so the competition focuses on modeling rather than domain feature engineering.

### Targets

- `t0`, `t1`: float32, present only in train and valid. Their exact definition is **not disclosed**. They are two indicators of future price movement of `i0`. There are no targets for `i1`.
- Your callback returns a length-2 array in **`[t0, t1]`** order.

---

## 5. Metric: Global Weighted Pearson (WP)

**Rows scored:** `need_prediction AND is_scored`. The validation mask is public (so you can reproduce the score locally). The test mask is hidden and applied by the platform after inference.

**Computation**, separately for `t0` and `t1`, on the selected rows:

- `y = clip(t, -2, 2)`, `ŷ = clip(p, -2, 2)`, weights `w = |y|`
- `ρ_w = cov_w(y, ŷ) / sqrt(var_w(y) · var_w(ŷ))`
- Final score `S = (ρ_t0,w + ρ_t1,w) / 2`

**Important properties**
- **Global:** all selected rows are pooled across sequences. Scores are **not** averaged by sequence.
- Both targets and predictions are **clipped to [-2, 2]**, so predicting outside that range gains nothing.
- Weights are `|y|`, so large-magnitude target moments dominate the score.
- The full spec is in the starter pack's `METRIC.md`, the baseline kit and the scorer documentation. **Read `METRIC.md` before trusting any local reimplementation.**

**[my note]** A minimal local reference implementation of the formula above (verify against `METRIC.md` / `utils.py` before relying on it):

```python
import numpy as np

def wp(t, p):
    y = np.clip(t, -2, 2).astype(np.float64)
    yh = np.clip(p, -2, 2).astype(np.float64)
    w = np.abs(y)
    sw = w.sum()
    my, myh = (w * y).sum() / sw, (w * yh).sum() / sw
    cov = (w * (y - my) * (yh - myh)).sum() / sw
    vy = (w * (y - my) ** 2).sum() / sw
    vyh = (w * (yh - myh) ** 2).sum() / sw
    return cov / np.sqrt(vy * vyh)

# score = (wp(t0, p0) + wp(t1, p1)) / 2, computed on rows where need_prediction & is_scored
```

---

## 6. Submission requirements

### Package

- A single **`.zip`** with **`solution.py` at the root of the archive**.
- Include weights, helper modules, and config files; load them via **paths relative to `solution.py`**.
- Example layout:
  ```
  submission.zip
  ├── solution.py
  ├── model.onnx
  ├── your_own_helpers/lib.py
  └── MODEL.md
  ```
- Build it from inside your submission directory: `zip -r ../submission.zip .`
- Make it a **code competition**: you upload inference code, not predictions.

### Required interface

`solution.py` must define a **no-argument** class `PredictionModel` with a method `predict(self, data_point)`.

```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        self.seq_ix = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            # Reset your model's sequence state here.

        # Process data_point.state and update model state on every call.
        if not data_point.need_prediction:
            return None

        # Replace with your predictions for t0 and t1.
        return np.zeros(2, dtype=np.float32)
```

### `DataPoint` (from the provided `utils.py`)

| Attribute | Type | Meaning |
| --- | --- | --- |
| `seq_ix` | int | Current sequence id |
| `step_in_seq` | int | 0 to 19,999 |
| `need_prediction` | bool | Whether a prediction is required |
| `state` | np.ndarray | 112 float32 features, in the order above |

- It **never** receives `is_scored` or the targets.
- **Return `None`** when `need_prediction` is False. Otherwise return **two finite float32-compatible values, shape `(2,)`, `[t0, t1]`**.
- Handle internal state carefully: reset on new `seq_ix`; process every row in order, drop none.

### Resource limits (evaluation environment)

| Resource | Limit |
| --- | --- |
| CPU | **1 vCPU** |
| RAM | **16 GB** |
| Inference time | **60 minutes for the entire test set** |
| Submission size | **20 MB** for the uploaded ZIP |
| GPU | **None** (CPU inference only) |
| Internet | **Disabled** |
| Python | 3.11 |

- **[my calc]** 39.4M rows in 3,600 s is about **91 µs per row** (about 1.8 s per 20,000-row sequence) on one core, including all Python callback overhead. This is the practical architecture constraint: small, fast models (e.g. GRU/small MLP/linear/GBM-lite via ONNX), minimal per-row Python work, no per-row allocations if avoidable. Profile locally early, and remember the 20 MB ZIP limit too.
- The doc shows `lscpu` for a 48-CPU AMD EPYC 9224 host (AVX2/AVX-512 available), but your container is limited to **1 vCPU**. Don't assume multi-threading helps; set thread counts to 1 when benchmarking (the baseline's ONNX session uses 1 thread).
- **Determinism is required:** repeated runs on the same input must give identical predictions (use fixed seeds, avoid nondeterministic ops/threading).

### Scoring Docker environment

- Base image `python:3.11-slim-bookworm`; apt packages: `curl libgomp1 p7zip-full build-essential`.
- Dependencies installed from `requirements.txt` (CPU-only PyTorch wheels via `https://download.pytorch.org/whl/cpu`), plus `orbax-checkpoint`.
- Offline env vars set: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `WANDB_DISABLED=1`, `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false`.
- Cache dirs redirected into `/app`: `HOME=/app`, `XDG_CACHE_HOME=/app/.cache`, `MPLCONFIGDIR=/app/.matplotlib`, `TORCH_HOME`, `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `NUMBA_CACHE_DIR` (all under `/app/.cache/...`).
- The rest of the Dockerfile is censored in the docs. You may use the visible part for local debugging.
- **Packages missing?** Ask on Discord or by email (see section 10) to have a package added to the scorer image.
- **[my note]** Because `HOME` and caches point to `/app` and the network is off, never rely on downloading weights or tokenizers at runtime. Bundle everything in the ZIP. Numba caching goes to `/app/.cache/numba`, so JIT compile time still counts toward your 60 minutes.

### Available packages (`requirements.txt` in the scorer)

- **Core numerics/IO:** numpy>=1.26,<3; scipy>=1.13,<2; pandas>=2.1,<3; pyarrow>=15,<19; fastparquet>=2024.5.0; polars>=0.20,<1
- **PyTorch (CPU):** torch>=2.3,<3; torchvision>=0.18,<1; torchaudio>=2.3,<3; lightning>=2.4,<3; torchmetrics>=1.4,<2
- **TensorFlow:** tensorflow>=2.17,<3
- **Classical ML:** scikit-learn>=1.4,<2; xgboost>=2.0,<3; lightgbm>=4.3,<5; catboost>=1.2,<2; statsmodels>=0.14,<1; **numba==0.67.0**; joblib>=1.3
- **Transformers/seq:** transformers>=4.41,<5; accelerate>=0.30,<1; datasets>=2.19,<3; tokenizers>=0.15,<1; sentencepiece>=0.1.99
- **JAX/Flax (CPU):** jax[cpu]>=0.4.31; flax>=0.12.0; orbax-checkpoint
- **Utilities:** tqdm>=4.66; einops>=0.7; rich>=13.7; loguru>=0.7; pydantic>=2.7,<3; hydra-core>=1.3,<2; omegaconf>=2.3,<3; pyyaml>=6.0; python-dotenv>=1.0
- **Tracking/HPO:** optuna>=3.5,<4; mlflow>=2.14,<3; wandb>=0.17,<1
- **Visualization:** matplotlib>=3.8,<4; seaborn>=0.13,<1
- **Export/interop:** onnxruntime>=1.18,<2; onnx>=1.16,<2
- **Testing:** pytest>=8.0
- **Other extras:** darts; neuralforecast; rtdl-num-embeddings

**[my note]** Train with versions compatible with these ranges (especially `pyarrow<19`, `numba==0.67.0`, `onnxruntime<2`, `torch<3`, `tensorflow<3`) so saved artifacts load identically in the scorer. ONNX export is the safest route for speed and portability.

---

## 7. Rules (fair play)

- **Who:** open worldwide; individuals or teams. (The interface currently has no way to team up.)
- **Code competition:** upload inference code as a `.zip`.
- **No external data.** Solutions must use only the provided data. Per the FAQ, that means only `train.parquet` and `valid.parquet`.
- **Pre-trained models are allowed** if they are **publicly available, open-source, and contain no external market data**.
- **Isolated container, no internet.**
- **Sharing:** do **not** publicly share your code, models, or detailed strategies before the competition ends. General idea discussion is fine. **Private sharing between participants is also not allowed.**
- **Submissions: max 5 per day.**
- **One account per person/team.**
- **Don't attack the infrastructure.** Cracking the scoring system or leaking data leads to disqualification.
- **Violations lead to disqualification.**

### Leaderboards and ranking

- **Public Leaderboard:** feedback on a *sample* of the test data.
- **Private Leaderboard:** hidden separate dataset, decides final winners.
- Final position is **solely** the Private Leaderboard score, on solutions you select for re-scoring on the Final Dataset.
- **Tie-break:** identical score, so the **earlier submission ranks higher**.

---

## 8. Prizes and verification

**Pool: $13,600 USDT for the top 8 on the Private Leaderboard**

| Place | Prize (USDT) |
| --- | --- |
| 1st | $5,000 |
| 2nd | $2,500 |
| 3rd | $1,700 |
| 4th | $1,300 |
| 5th | $1,000 |
| 6th | $800 |
| 7th | $700 |
| 8th | $600 |
| **Total** | **$13,600** |

### Verification (top 8 only, 7 calendar days after preliminary results)

You must provide:
1. **Inference code**: the exact code that produced the winning submission, including model weights.
2. **Training code**: complete, with **fixed random seeds**.
3. **Technical report**: 2 to 8 pages covering approach, architecture, validation strategy.
4. **Code review call**: 30 to 60 minute video call walking through your code.
5. **License**: a **non-exclusive license** to use your winning solution **internally for research purposes**. Your work will not be made public.

They check **reproducibility** (same results from your code), **compliance** (all rules followed), and **clarity** (of the report).

**Payout:** USDT to your wallet **within 14 days** of verification. If verification fails, you'll be asked for corrections. If unresolved within **5 days**, the prize is offered to the next-highest-ranking participant.

**[my note]** Because training code must be reproducible with fixed seeds and the report is mandatory, keep clean experiment logs, seeds, configs and a short running design diary from day one.

---

## 9. Validation strategy guidance (from the FAQ)

- The supplied `valid.parquet` + mask is **one useful validation setup** based on a selected subset of moments. You **may define your own validation splits and masks** from the labeled data. This doesn't change the platform's fixed test mask or the Global WP metric.
- The test mask is hidden. **Aim for reliable predictions at all required moments**, not just those resembling the visible validation subset. Return predictions whenever `need_prediction` is true.
- A strong score on the supplied subset **does not guarantee** strong test performance. Check additional held-out splits and moments to reduce overfitting to the visible validation mask and the risk of weaker private-final results.
- Since you can use both `train` and `valid` for training, decide deliberately when to hold out validation vs. fold it into final training. Keep a held-out slice (e.g. grouped by `seq_ix`) to detect overfitting.
- Don't tune endlessly on the public leaderboard: it's only a sample of the test data, and the final is a separate hidden set.

---

## 10. Support

- **Discord (official):** https://discord.gg/Xgs4smap9b (WUNDERNN discord)
- **Email:** ops@wunderfund.io (private inquiries or issues)
- For technical issues include: **Python/package versions, starterpack version, the exact command you ran, and the complete error message.**
- Use these channels to request additional packages for the scorer image.

---

## 11. Pre-submission checklist

- [ ] `solution.py` is at the **ZIP root**; all weights/helpers/configs included and loaded via paths **relative to `solution.py`**
- [ ] `PredictionModel` takes **no constructor args**; `predict(self, data_point)` exists
- [ ] Returns **`None`** when `need_prediction` is False; otherwise a **finite `(2,)` float32** array in `[t0, t1]` order
- [ ] **State is updated on every row** (warm-up rows included) and **reset when `seq_ix` changes**
- [ ] No use of `seq_ix` as a feature; no lookahead; no dropped rows
- [ ] **Deterministic**: two repeated runs give identical outputs
- [ ] Features consumed in the **file order** (not alphabetical)
- [ ] Runtime comfortably under **60 min** for 39.4M rows on **1 core** (leave margin; scorer hardware may differ from yours)
- [ ] RAM comfortably under **16 GB**, ZIP under **20 MB**
- [ ] Only packages in the scorer `requirements.txt`; no network access, no runtime downloads
- [ ] Local WP on validation matches expectations (baseline reference: **0.617052**)
- [ ] Remaining daily submissions (max **5/day**) planned

---

## 12. Suggested execution plan

1. **Day 0:** download the starter pack, read `METRIC.md`, `baseline/README.md`, `utils.py`; reproduce the **0.617052** baseline locally; submit the baseline zip once to validate the pipeline end to end.
2. **Build a fast local harness** that streams row groups, feeds `DataPoint`s to your `PredictionModel`, and computes WP with the validation mask. Add a timing report in µs/row.
3. **Establish a validation protocol** beyond the supplied mask (extra held-out sequences and moments, compare score on the mask vs. on all `need_prediction` rows).
4. **Iterate within the compute budget** (about 91 µs/row **[my calc]**). Export to ONNX (single thread) when possible.
5. **Control public-leaderboard feedback**: 5 submissions/day is a budget, so batch experiments locally first.
6. **Before the deadline (Nov 15, 23:59 UTC):** select your final candidates for Private re-scoring, and make sure each was already submitted and scored.
7. **If you place top 8:** have training code (fixed seeds), inference code with weights, the 2 to 8 page report, and availability for a 30 to 60 minute call ready within the 7-day window.

---

## 13. Open points to confirm

- The docs don't give the **exact public/private leaderboard split** or the exact **selection count** for final re-scoring. Watch the submit page or ask on Discord.
- The "No external data" rule says "training data we provide"; the FAQ clarifies this includes both `train.parquet` and `valid.parquet`.
- Exact `t0`/`t1` definitions are **intentionally undisclosed**.
- **Team formation** isn't supported in the interface yet, so ask organizers if you plan to compete as a team.
- Quick Start packages with `zip -r submission.zip solution.py baseline.onnx`, while the Submission Guide uses `zip -r ../submission.zip .` from the submission directory. Either works as long as `solution.py` ends up at the archive root. Check with `unzip -l submission.zip` and make sure no extra top-level folder wraps your files.
