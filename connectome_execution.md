
## Quick start guide

This guide will help you get started with the **Alpha Connectome ML Competition**.

## 1. Get the starter pack

```bash
curl -L https://files.wundernn.io/wnn_connectome_starterpack.tar.gz | tar -xz && cd wnn_connectome_starterpack
```

or download the archive manually:

**STARTERPACK**

- [wnn_connectome_starterpack.zip](https://files.wundernn.io/wnn_connectome_starterpack.zip)
- [wnn_connectome_starterpack.tar.gz](https://files.wundernn.io/wnn_connectome_starterpack.tar.gz)
- *slow download? VPN enjoyer?*
  try [mirror-1](https://files-alternative.wundernn.io/wnn_connectome_starterpack.zip)/[mirror-2](https://storage.yandexcloud.net/files-alternative.wundernn.io/wnn_connectome_starterpack.zip)

### What's inside

```
wnn_connectome_starterpack/
├── baseline
│   ├── README.md
│   ├── baseline.onnx
│   ├── baseline_submission.zip
│   └── solution.py
├── datasets
│   ├── train.parquet
│   ├── valid.parquet
│   └── valid_mask.parquet
├── docs
├── METRIC.md
├── README.md
├── requirements.txt
└── utils.py
```

## 2. Set up your environment and packages

We strongly recommend using a virtual environment to keep your project dependencies tidy and avoid conflicts. Probably any Python 3.10+ will do. Submissions will be evaluated in a Docker container based on `python:3.11-slim-bookworm` image.

Set up the env and then install the required packages:

`pip install -r requirements.txt`

## 3. Explore the data

The dataset is provided in Parquet format.

*   `datasets/train.parquet`: 10,607 sequences.
*   `datasets/valid.parquet`: 1,873 sequences.

Each sequence contains 20,000 rows. Steps 0–98 are warm-up; predictions are required at steps 99–19,999. Inputs contain 112 features.

```
import pyarrow.parquet as pq

dataset = pq.ParquetFile('datasets/train.parquet')
sequence = dataset.read_row_group(0)
print(sequence.shape)
print(sequence.column_names)
```

The details are in [Data overview](#data-overview).

## 4. Run the example solution

We provided a baseline solution in `baseline` folder.

`python baseline/solution.py --validation datasets/valid.parquet`

This script will load the validation data and the pre-trained `baseline.onnx` model, run inference, and calculate the **Global Weighted Pearson** score.

FUN FACTS

> *   This baseline is a stateful vanilla GRU.
> *   It scores **0.617052 WP** on the complete validation set.
> *   The ONNX Runtime session uses one CPU compute thread.

## 5. Submit

Zip your `solution.py` and model artifacts:

```
cd baseline
zip -r submission.zip solution.py baseline.onnx
```

Go to the [submit](https://wundernn.io/connectome/submit) page and send your solution for scoring.

> 🎉 ta-daa, you're awesome

* * *

## 6. Create your own model

1.   Create your own `solution.py` implementing the `PredictionModel` interface. Use [Submission guide](#how-to-make-a-submission) for reference.
2.   Train your model using `datasets/train.parquet`.
3.   Validate using `datasets/valid.parquet`.

The data and metric is described in details in [Data overview](#data-overview). The starterpack's `METRIC.md` contains the full scoring reference.

Good luck with the challenge! We're excited to see what you build.

## Data overview

Each row of the data is one moment in time for two trading instruments: `i0` and `i1`.   
 There are 112 input features: orderbook- and trades-related features for both instruments plus eight additional features.

The task is to predict `t0` and `t1`, two anon indicators of the future price movement of instument `i0`. Submissions are scored with **Global Weighted Pearson** correlation (WP), computed over all rows selected by `need_prediction AND is_scored` in the evaluated dataset, separately for each target and then averaged across the two targets.

## Data files

The participant dataset contains:

*   `datasets/train.parquet` — training sequences with targets `t0`, `t1`.
*   `datasets/valid.parquet` — validation sequences with targets and an additional scoring mask column `is_scored`.
*   `datasets/valid_mask.parquet` — the very same validation scoring mask as a standalone three-column table (`seq_ix`, `step_in_seq`, `is_scored`).

| File | Sequences | Rows (= Seq × 20,000) |
| --- | --- | --- |
| `train.parquet` | 10,607 | 212,140,000 |
| `valid.parquet` | 1,873 | 37,460,000 |
| `valid_mask.parquet` | 1,873 | 37,460,000 |

TIP

> **Reading data:** Each Parquet row group is exactly one sequence: 20,000 rows with a single `seq_ix`. `ParquetFile.read_row_group(i)` therefore returns one complete sequence, which is the simplest way to iterate over the data without loading a whole file (see [Quick Start](#quick-start-guide))

Test data has the same format and sequence structure as `valid.parquet`, but is never distributed to participants: the platform feeds it to your solution's callback row by row during the evalutation.

## Sequences

### Structure

*   A sequence is 20,000 consecutive rows with one `seq_ix` and `step_in_seq` = 0 … 19,999.
*   Steps 0–98 are warm-up: `need_prediction` is false and no output is expected. They give your model 99 rows of context before the first prediction.
*   Steps 99–19,999: `need_prediction` is true; two finite predictions are required on every one of these steps.
*   Only some of the required steps are scored (`is_scored`; see Evaluation).

### Rules for your model

*   Process rows in the given order and update your state on every row — including warm-up rows and rows where `is_scored` is false.
*   Reset the state when `seq_ix` changes. Sequences are independent: do not join them or carry state across them.
*   `seq_ix` values are shuffled: consecutive ids are not adjacent in time. Do not use `seq_ix` as a feature or to order sequences.
*   A prediction may use only the current row and earlier rows of the same sequence.

## Row columns

| Column | train | valid | valid_mask | Meaning |
| --- | --- | --- | --- | --- |
| `seq_ix` | ✓ | ✓ | ✓ | Sequence id. Not a feature; a larger id does not mean later in time |
| `step_in_seq` | ✓ | ✓ | ✓ | Row position in the sequence, 0 … 19,999 |
| `need_prediction` | ✓ | ✓ | × | False on steps 0–98, true on steps 99–19,999 |
| `is_scored` | × | ✓ | ✓ | True on rows used for evaluation |
| 112 feature columns | ✓ | ✓ | × | float32, in the order of the table below |
| `t0`, `t1` | ✓ | ✓ | × | float32 targets for `i0` |

Column layout of each file:

```
train.parquet       seq_ix, step_in_seq, need_prediction,            <112 features>, t0, t1   (117 columns)
valid.parquet       seq_ix, step_in_seq, need_prediction, is_scored, <112 features>, t0, t1   (118 columns)
valid_mask.parquet  seq_ix, step_in_seq, is_scored
```

`valid_mask.parquet` carries the same mask as the `is_scored` column of `valid.parquet`; join the two on (`seq_ix`, `step_in_seq`).

At test time your `predict` callback receives a `DataPoint` with `seq_ix`, `step_in_seq`, `need_prediction` and `state` — the 112 features as a float32 array in the order below. It never receives `is_scored` or the targets. See [Submission guide](#how-to-make-a-submission).

## Target columns

`t0` and `t1` are two indicators of the future price movement of instument `i0`; their exact definition and nature is not disclosed. There are no targets for instument `i1`.

*   Both are float32 columns present in `train.parquet` and `valid.parquet` only.
*   Your callback returns them as a length-2 array in `[t0, t1]` order.
*   The metric clips targets and predictions to `[-2, 2]` during scoring.

## Feature columns

Features describe two instruments: `i0`, whose future price movement you predict, and `i1`, a second instrument provided as input only (it has no targets).

Column names follow `i<instrument>_<group><index>` naming, where the group is `p` (price-like), `v` (volume-like), `dp` (trade price) or `dv` (trade volume).

In other words, the data contains **L11** order books (top-11 bid and ask prices+volumes) for 2 instruments, plus history of trades, plus 8 additional features.

The table below is the exact order of the 112 features in the files and in `state`. Within each range indices increase numerically (`p0, p1, …, p10`). Do not sort column names alphabetically: that would place `i0_p10` right after `i0_p1`, before `i0_p2`. Note that bid and ask are told apart by index only (0–10 bid, 11–21 ask).

| Group | Columns | Count | Position in `state` |
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

> **The index is an identifier, not a depth level.**`p0` is not necessarily the best bid and `p10` is not necessarily the deepest level; the same holds for volume-like features. An index only guarantees that the same component of the book representation appears in the same column on every row. Do not assume any ordering by price or by distance from the best quote.

## Evaluation

### Which rows are scored

Only rows with `is_scored = true` participate in the scoring; formally `need_prediction AND is_scored` (the mask never selects warm-up rows, so the `AND` here is a safeguard).

For validation the mask is given (`is_scored` / `valid_mask.parquet`) so that you can reproduce the score locally. For test it is hidden and applied by the platform after inference. Your model therefore cannot know which rows count: return a prediction on every required row as if it were scored.

### The metric: Global Weighted Pearson correlation

The metric is **Global Weighted Pearson correlation (WP)**. Higher is better.

Calculate WP separately for `t0` and `t1` over the entire evaluated dataset, using all rows where `need_prediction AND is_scored` is true. On those rows, both target and prediction values are clipped to `[-2.0, 2.0]`:

$$y_i = \text{clip}(t_i, -2, 2), \quad \hat{y}_i = \text{clip}(p_i, -2, 2), \quad w_i = \vert{}y_i\vert{}.$$


For each target, the weighted correlation is

$$\rho_w = \frac{\text{cov}_w(y, \hat{y})}{\sqrt{\text{var}_w(y) \text{var}_w(\hat{y})}}.$$


The final score is the mean of the two global target correlations:

$$S = (\rho_{t0,w} + \rho_{t1,w}) / 2$$


All selected moments are pooled across sequences; scores are not averaged by sequence. See the baseline kit and scorer documentation for the full evaluation specification.
## How to make a submission

This page covers the technical requirements for your submission, including the code format, how to package your files, and the resource limits.

## What to submit

This is a code competition. You'll submit a `.zip` file containing all the code and artifacts needed to generate predictions.

The key requirements are:

*   The zip file must contain a `solution.py` file at its root.
*   Your `solution.py` must define a class named `PredictionModel`.
*   This class must have a `predict(self, data_point)` method.

## The `PredictionModel` class

Your `solution.py` must define a no-argument `PredictionModel` class with a `predict(self, data_point)` method. This example shows the interface; replace the placeholder output with your model's predictions.

```
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

## Inputs, `DataPoint` and outputs

Your `predict` method will receive a `DataPoint` object on each call. Your code should return `None` if `need_prediction` is `False`,otherwise return two finite float32-compatible values with shape `(2,)`, in `[t0, t1]` order.

The `DataPoint` object comes from the provided `utils.py` file and has the following attributes:

*   `seq_ix: int`: The ID for the current sequence
*   `step_in_seq: int`: The step number within the sequence: 0 to 19,999
*   `need_prediction: bool`: Whether a prediction is required for this point
*   `state: np.ndarray`: 112 float32 features in the order listed in [Data overview](#data-overview).

NOTE

> Remember to handle the model's internal state. When you encounter a new sequence (a new `seq_ix`), you must reset the state.

Process every row in the provided order, not dropping any.

## How to package your solution

You need to package all your files into a single `.zip` archive with `solution.py` in the root of the archive.

Include your weights, helper modules and configuration files. Load them using paths relative to `solution.py`.

```
submission.zip
├── solution.py
├── model.onnx
├── your_own_helpers
├   └── lib.py
└── MODEL.md
```

From a directory containing your submission files:

`zip -r ../submission.zip .`

## Evaluation environment and limits

Your code will run in an isolated Linux environment with Python 3.11 with the following constraints:

| Resource | Limit |
| --- | --- |
| CPU | 1 vCPU |
| RAM | 16 GB |
| Inference time | 60 minutes for the entire test set |
| Submission size | 20 MB for the uploaded ZIP |
| GPU | Not available; CPU inference only |
| Internet access | Disabled |

### CPU

Here's the exact CPU spec:

`scoring@server:~# lscpu`

```
scoring@server:~# lscpu
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          52 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   48
  On-line CPU(s) list:    0-47
Vendor ID:                AuthenticAMD
  Model name:             AMD EPYC 9224 24-Core Processor
    CPU family:           25
    Model:                17
    Thread(s) per core:   2
    Core(s) per socket:   24
    Socket(s):            1
    Stepping:             1
    Frequency boost:      enabled
    CPU(s) scaling MHz:   88%
    CPU max MHz:          2500.0000
    CPU min MHz:          1500.0000
    BogoMIPS:             4992.59
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constan
                          t_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsa
                          ve avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb
                          bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid
                          cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_
                          llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc amd_ibpb_ret arat npt lbrv svm_lock nrip_sav
                          e tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx5
                          12_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
Virtualization features:
  Virtualization:         AMD-V
Caches (sum of all):
  L1d:                    768 KiB (24 instances)
  L1i:                    768 KiB (24 instances)
  L2:                     24 MiB (24 instances)
  L3:                     64 MiB (4 instances)
NUMA:
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-47
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Mitigation; Safe RET
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
root@222725652df5:~# LC_ALL=C lscpu
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          52 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   48
  On-line CPU(s) list:    0-47
Vendor ID:                AuthenticAMD
  Model name:             AMD EPYC 9224 24-Core Processor
    CPU family:           25
    Model:                17
    Thread(s) per core:   2
    Core(s) per socket:   24
    Socket(s):            1
    Stepping:             1
    Frequency boost:      enabled
    CPU(s) scaling MHz:   92%
    CPU max MHz:          2500.0000
    CPU min MHz:          1500.0000
    BogoMIPS:             4992.59
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constan
                          t_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsa
                          ve avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb
                          bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid
                          cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_
                          llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc amd_ibpb_ret arat npt lbrv svm_lock nrip_sav
                          e tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx5
                          12_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
Virtualization features:
  Virtualization:         AMD-V
Caches (sum of all):
  L1d:                    768 KiB (24 instances)
  L1i:                    768 KiB (24 instances)
  L2:                     24 MiB (24 instances)
  L3:                     64 MiB (4 instances)
NUMA:
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-47
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Mitigation; Safe RET
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

## How submissions are scored

### Docker

When you submit a solution, a scoring Docker container is deployed. The intricacies of the scoring process is not important here, but some details might be useful for you.

*   The container image is based on `python:3.11-slim-bookworm`
*   The environment variables used by some ML libraries are configured to prevent them from attempting to access the network

Below is the part of Dockerfile. You might want to use it locally for debug purposes.

```
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

# Keep heavy libs strictly offline at runtime 
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
# Redirect all caches into /app
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba

### REST IS SENSORED ###
```

### Libs and packages

You may notice a `requirements.txt` is mentioned in Dockerfile above. We tried to install some reasonable set of popular libs often used in ML.

NOTE

> If you need any package added to the scorer docker image —please drop us a line in Discord or email: [get help](https://wundernn.io/connectome/docs/get_help)

Here's the full `requirements.txt`:

```
### requirements.txt ###

# === Core numerics & IO ===
numpy>=1.26,<3
scipy>=1.13,<2
pandas>=2.1,<3
pyarrow>=15,<19
fastparquet>=2024.5.0
polars>=0.20,<1

# === Deep Learning (PyTorch stack, CPU-only) ===
torch>=2.3,<3
torchvision>=0.18,<1
torchaudio>=2.3,<3
lightning>=2.4,<3           # (ex. pytorch-lightning)
torchmetrics>=1.4,<2

# === TensorFlow ===
tensorflow>=2.17,<3         # provides tensorflow.keras

# === Classical ML ===
scikit-learn>=1.4,<2
xgboost>=2.0,<3
lightgbm>=4.3,<5
catboost>=1.2,<2
statsmodels>=0.14,<1
numba==0.67.0
joblib>=1.3

# === Transformers / seq modeling ===
transformers>=4.41,<5
accelerate>=0.30,<1
datasets>=2.19,<3
tokenizers>=0.15,<1
sentencepiece>=0.1.99

# === JAX & Flax (CPU) ===
jax[cpu]>=0.4.31
flax>=0.12.0
orbax-checkpoint

# === Utilities ===
tqdm>=4.66
einops>=0.7
rich>=13.7
loguru>=0.7
pydantic>=2.7,<3
hydra-core>=1.3,<2
omegaconf>=2.3,<3
pyyaml>=6.0
python-dotenv>=1.0

# === Experiment tracking & HPO ===
optuna>=3.5,<4
mlflow>=2.14,<3
wandb>=0.17,<1

# === Visualization ===
matplotlib>=3.8,<4
seaborn>=0.13,<1

# === Export / interop ===
onnxruntime>=1.18,<2
onnx>=1.16,<2

# === Testing ===
pytest>=8.0

# === Other extras ===
darts
neuralforecast
rtdl-num-embeddings
```

## Before uploading

*   Check that `solution.py` is at the ZIP root and required files are included.
*   Check output shape, warm-up handling and sequence-state resets.
*   Confirm identical predictions across repeated runs.
*   Check that the solution meets the execution and submission-size limits above.

The [Rules](#competition-rules) are also worth reading.


## Competition rules

Here are the rules to ensure a fair and fun competition for everyone.

## The basics

*   **Who can join:** The competition is open to everyone, worldwide. You can participate as an individual or as part of a team.
*   **How you're scored:** We use two leaderboards. The Public Leaderboard provides feedback on a sample of the test data. The Private Leaderboard, used at the end, determines the final winners on a separate, hidden dataset.
*   **What you submit:** This is a code competition. You'll upload your inference code as a `.zip` file.

## Fair play

*   **No external data:** Your solution must only use the training data we provide.
*   **Pre-trained models are allowed:** You may use open-source pre-trained models, provided they are _publicly available and do not contain external market data_.
*   **Isolated environment:** Your code will run in an isolated Linux container with no internet access.
*   **Sharing:** Please don't share your code, models, or detailed strategies publicly before the competition is over. Discussing general ideas is great, but sharing complete solutions isn't fair to others. Private sharing between participants is also not allowed.
*   **Submissions:** You can make up to 5 submissions per day.
*   **Single accounts:** One account per team/person. We won't see the #1 participant to take all the winning ranks.
*   **No messing with the infra:** We respect out-of-the box thinking and the hacker way very much. If you crack our scoring system, leak the data or such — you'll have our regards, but you'll also be disqualified.

> Violation of the rules will lead to disqualification.

## Leaderboard and tie-breaking

*   **Final ranking:** Your final position will be determined solely by your score on the Private Leaderboard. Close to the end of the competition you will be asked to choose some of your _previously submitted and scored solutions_ to be re-scored on the Final Dataset.
*   **Tie-breaking:** If two participants have the exact same score, the participant who submitted their solution earlier will be ranked higher.


## Competition timeline

Here are the key dates for the Alpha Connectome ML Competition. All deadlines are at 11:59 PM UTC
*   SEP 11, 2026 — competition starts
*   NOV 15, 2026 — final submission deadline, new submissions are no longer accepted
*   DEC 01, 2026 — winner announcement

We're excited to see your submissions!


## Prizes and verification

We have a total prize pool of over $13,000 USDT for the top 8 participants on the Private Leaderboard.

## Prize pool

The prizes will be distributed as follows:

+--------+-------------+ | Place | Prize, USDT | +--------+-------------+ | 1st | $5,000 | | 2nd | $2,500 | | 3rd | $1,700 | | 4th | $1,300 | | 5th | $1,000 | | 6th | $800 | | 7th | $700 | | 8th | $600 | +--------+-------------+ | Total | $13,600 | +--------+-------------+

## Prize eligibility and verification

To be eligible for a prize, the top 8 participants must pass a solution verification process. After the preliminary results are announced, you will have 7 calendar days to submit your materials.

### What you'll need to provide

1.   **Inference code:** The exact code that produced your winning submission, including any model weights.
2.   **Training code:** The complete code used to train your model, with fixed random seeds.
3.   **Technical report:** A short (2–8 page) document describing your approach, architecture, and validation strategy.
4.   **Code review call:** A 30-60 minute video call with us to walk through your code.
5.   **License:** You'll grant us a non-exclusive license to use your winning solution internally for research purposes. Your work will not be made public.

### How we'll verify your solution

We will check for:

*   **Reproducibility:** We can run your code and get the same results.
*   **Compliance:** Your solution follows all competition rules.
*   **Clarity:** Your technical report clearly explains your method.

### Prize payout

Once your solution is verified, we'll transfer the prize in USDT to your wallet within 14 days.

If verification fails, we'll ask for corrections. If the issues can't be resolved within 5 days, the prize will be offered to the next-highest-ranking participant.


## FAQ

Here are answers to some common questions. This list will be updated during the competition.

## Competition mechanics

**What is the goal of this competition?**  
 Your goal is to predict `t0` and `t1`, two indicators of future price movements of a trading instrument.

**What is the evaluation metric?**  
 Global Weighted Pearson correlation (WP), calculated over all selected rows of the evaluated dataset separately for each target, then averaged across the two targets. On the scoring mask, targets and predictions are clipped to `[-2.0, 2.0]` before WP. See [Data overview](#data-overview) for the formula. A higher score is better.

**Can I work in a team?**  
 Yes, you can participate as an individual or as part of a team. Currently there's no way to team-up in the interface.

**How many submissions can I make per day?**  
 You can make up to 5 submissions per day.

## Data questions

**What data is provided?**  
`datasets/train.parquet` and `datasets/valid.parquet`. Dataset sizes, columns and sequence format are described in [Data overview](#data-overview).

**How large is the test set?**  
 The current test set contains **1,970 sequences**, each with **20,000 rows**: **39,400,000 rows** in total. Test targets and the scoring mask are hidden.

**Must I use the supplied validation setup for model selection?**  
 The supplied validation set and mask provide one useful validation setup based on a selected subset of moments. You may define your own validation splits and masks using the supplied labeled data. This does not change the platform's fixed test mask or the Global WP metric.

**Is the scoring mask available at test time?**  
 The validation mask is public, but the test mask is hidden. Aim for reliable predictions at all required moments, not only those resembling the visible validation subset. Return predictions whenever `need_prediction` is true.

A strong score on the supplied subset alone does not guarantee strong test performance. Check additional held-out splits and moments to reduce overfitting to the visible validation mask and the risk of weaker results on the private final test.

**Can I use external data?**  
 No. All solutions must be trained using only the provided `train.parquet` and `valid.parquet` datasets.

**Can I use pre-trained models?**  
 Yes, you may use open-source pre-trained models, provided they are publicly available and do not contain external market data.

**Why are the features anonymized?**  
 The features are anonymized to focus the competition on the modeling task itself, rather than on domain-specific feature engineering.

## Technical questions

**When should I reset model state?**  
 Reset when `seq_ix` changes. Process every row, including warm-up rows.

**What are the execution limits?**  
 Your code will run in a Linux container with:

*   1 CPU core
*   16 GB of RAM
*   No GPU — CPU inference only
*   A **60-minute** time limit for the entire test set

**Must my solution be deterministic?**  
 Yes. Repeated runs on the same input must produce identical predictions.

**Can my submission contain multiple files?**  
 Yes. Include the required code and model files in one ZIP, with `solution.py` at the root. See [Submission guide](#how-to-make-a-submission).


## Get help

Have questions? We're here to help.

The best way to get in touch with the organizers and other participants is through our official Discord server.

*   **Join the Discord:**[WUNDERNN discord](https://discord.gg/Xgs4smap9b)

For any private inquiries or issues, you can also reach us by email.

*   **Email us:**[ops@wunderfund.io](mailto:ops@wunderfund.io)

We'll do our best to get back to you as quickly as possible. Good luck!

> For technical issues, pls include your Python/package versions, starterpack version, the command you ran and the complete error message.