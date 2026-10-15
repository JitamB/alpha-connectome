#!/bin/bash
# After x_lag_s4000 (started by runs/queue_step1b.sh): zero-inference-cost levers first, capacity ceiling last
# (h192/p96 gave no gain, so h384 is now the lowest-priority run). All runs keep weight EMA (<run>_ema).
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
while pgrep -f "colab_train.py x_lag_s4000" >/dev/null; do sleep 60; done
COMMON="--max-train-seqs 4000 --epochs 3 --batch 16"
run() { name=$1; shift; keep=$1; shift
  rm -rf "runs/$name"
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- $COMMON "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_aux_s4000         --keep-session --seed 0 --aux-weight 0.2 --ema-decay 0.999
run x_maskw_s4000       --keep-session --seed 0 --mask-model models/mask_mlp.pt --mask-eps 0.1 --ema-decay 0.999
run x_h384_p192_s4000   ""             --seed 0 --hidden 384 --proj 192 --ema-decay 0.999
