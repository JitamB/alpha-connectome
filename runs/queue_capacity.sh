#!/bin/bash
# Step 1 of the breakthrough plan: capacity ceiling test (s4000, 3 epochs). Logs: runs/<name>.orchestrator.log
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
COMMON="--max-train-seqs 4000 --epochs 3 --batch 16"
run() { name=$1; shift; keep=$1; shift
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- $COMMON "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_h384_p192_s4000   --keep-session --seed 0 --hidden 384 --proj 192
run x_h192_p96_s4000    ""             --seed 0 --hidden 192 --proj 96
