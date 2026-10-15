#!/bin/bash
# Sequential Colab experiment queue (one session, data fetched once). Logs: runs/<name>.orchestrator.log
cd "$(dirname "$0")/.."
COMMON="--max-train-seqs 4000 --epochs 3 --batch 16"
run() { name=$1; shift; keep=$1; shift
  .venv/bin/python -u tools/colab_train.py "$name" $keep -- $COMMON "$@" > "runs/$name.orchestrator.log" 2>&1
  echo "$(date +%H:%M) finished $name (exit $?)"; }
run x_base_s4000        --keep-session --seed 0
run x_pearson_s4000     --keep-session --seed 0 --loss pearson
run x_base_s4000_seed1  --keep-session --seed 1
run x_h160_s4000        --keep-session --seed 0 --hidden 160
run x_p128_s4000        ""             --seed 0 --proj 128
