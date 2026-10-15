#!/bin/bash
# x_lag_s4000 (realised-past-target head) gave V1 +0.0032 / V2 +0.006 / V3 +0.003 vs the 2-seed reference;
# weight EMA gave nothing. Next: replicate the lag head, a stronger lag variant, then the remaining levers; h384 last.
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
while pgrep -f "colab_train.py x_aux_s4000" >/dev/null; do sleep 60; done
COMMON="--max-train-seqs 4000 --epochs 3 --batch 16"
run() { name=$1; shift; keep=$1; shift
  rm -rf "runs/$name"
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- $COMMON "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_lag_s4000_seed1   --keep-session --seed 1 --lag-targets 100 300 --lag-weight 0.5
run x_lag5_s4000        --keep-session --seed 0 --lag-targets 20 50 100 200 300 --lag-weight 1.0
run x_maskw_s4000       --keep-session --seed 0 --mask-model models/mask_mlp.pt --mask-eps 0.1
run x_h384_p192_s4000   ""             --seed 0 --hidden 384 --proj 192
