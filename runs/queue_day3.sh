#!/bin/bash
# 2026-10-15: bundled run A (s4000: mask weighting + lag & future-change heads + distributional head), then the same
# recipe on all training data (B). Each run is scored with the regression, decoded-distribution and blended outputs.
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
RECIPE="--epochs 3 --batch 16 --seed 0 --mask-model models/mask_mlp.pt --mask-eps 0.1 --lag-targets 100 300 --lag-weight 0.5 --aux-weight 0.2 --dist-weight 0.5"
run() { name=$1; shift; keep=$1; shift
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_all_s4000  --keep-session --max-train-seqs 4000 $RECIPE
run gru_all_full ""             $RECIPE
