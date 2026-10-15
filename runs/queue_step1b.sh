#!/bin/bash
# Follow-up to runs/queue_capacity.sh (its h384 run died in the data fetch: OOM, fixed in tools/fetch_data.py).
# Waits for that queue, then: realised-past-target head, capacity ceiling h384, auxiliary future-change targets,
# mask-proxy weighting (all with weight EMA, evaluated as <run>_ema).
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
while pgrep -f queue_capacity.sh >/dev/null; do sleep 60; done
COMMON="--max-train-seqs 4000 --epochs 3 --batch 16"
run() { name=$1; shift; keep=$1; shift
  rm -rf "runs/$name"
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- $COMMON "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_lag_s4000         --keep-session --seed 0 --lag-targets 100 300 --lag-weight 0.5 --ema-decay 0.999
run x_h384_p192_s4000   --keep-session --seed 0 --hidden 384 --proj 192 --ema-decay 0.999
run x_aux_s4000         --keep-session --seed 0 --aux-weight 0.2 --ema-decay 0.999
run x_maskw_s4000       ""             --seed 0 --mask-model models/mask_mlp.pt --mask-eps 0.1 --ema-decay 0.999
