#!/bin/bash
# Usage: [EXTRA='files'] kaggle/push.sh JOB_SLUG 'RUN1|args...' ['RUN2|args...']   (RUN 'eval:NAME' = eval_torch args only)
# Uploads the current code as private dataset jitamb/wunder-code (new version) and pushes a private GPU notebook
# jitamb/<JOB_SLUG> running kaggle/run_job.py with the given runs (one per GPU). GPU=false: CPU-only notebook (no GPU quota).
set -euo pipefail
cd "$(dirname "$0")/.."
K=~/.local/bin/kaggle; USER=jitamb; SLUG=$1; shift
# The CLI's stored OAuth login stops working for API calls after a while; a freshly printed token still works.
export KAGGLE_API_TOKEN=$($K auth print-access-token 2>/dev/null | tail -1)
B=$(mktemp -d)
mkdir -p $B/data $B/kernel
COPYFILE_DISABLE=1 tar czf $B/data/code.tgz src/data.py src/evaluate.py src/features.py train/train_gru.py train/gpu_features.py \
    train/eval_torch.py tools/fetch_data.py splits/holdout.json splits/train_classes.json wnn_connectome_starterpack/utils.py \
    models/mask_model.py models/mask_mlp.pt tools/probe_hidden.py splits/valid_classes.json ${EXTRA:-}   # EXTRA: more files, e.g. checkpoints for eval-only jobs
cat > $B/data/dataset-metadata.json <<META
{"title": "wunder-code", "id": "$USER/wunder-code", "licenses": [{"name": "other"}]}
META
if $K datasets status $USER/wunder-code >/dev/null 2>&1; then
  $K datasets version -p $B/data -m "code for $SLUG" -r skip
else
  $K datasets create -p $B/data -r skip   # private by default
fi
JOBS=$(python3 - "$@" <<'PY'
import sys, shlex
print(repr([(j.split('|', 1)[0], shlex.split(j.split('|', 1)[1])) for j in sys.argv[1:]]))
PY
)
JOBS="$JOBS" python3 -c "import os,sys; print(open('kaggle/run_job.py').read().replace('JOBS = []', 'JOBS = ' + os.environ['JOBS'], 1), end='')" > $B/kernel/run_job.py
cat > $B/kernel/kernel-metadata.json <<META
{"id": "$USER/$SLUG", "title": "$SLUG", "code_file": "run_job.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": ${GPU:-true}, "enable_internet": true,
 "dataset_sources": ["$USER/wunder-code"], "competition_sources": [], "kernel_sources": []}
META
grep '^JOBS' $B/kernel/run_job.py
until $K datasets status $USER/wunder-code 2>/dev/null | grep -qi ready; do sleep 10; done   # notebook must mount the new version
$K kernels push -p $B/kernel
