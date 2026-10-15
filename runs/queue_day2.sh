#!/bin/bash
# 2026-10-10 Colab queue (Mac only orchestrates): replicate + combine the aux heads, stronger lag variant, mask weighting.
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
COMMON="--max-train-seqs 4000 --epochs 3 --batch 16"
run() { name=$1; shift; keep=$1; shift
  rm -rf "runs/$name"
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- $COMMON "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_lagaux_s4000_seed1 --keep-session --seed 1 --lag-targets 100 300 --lag-weight 0.5 --aux-weight 0.2
run x_lag5_s4000         --keep-session --seed 0 --lag-targets 20 50 100 200 300 --lag-weight 1.0
run x_maskw_s4000        ""             --seed 0 --mask-model models/mask_mlp.pt --mask-eps 0.1
