#!/bin/bash
# Run back-to-back PLR variance runs (distinct seeds) until the machine is killed.
# Each run = 10000 plr steps (5000 DR-equivalent). Sequential (one GPU, ~33GB/run).
cd "$(dirname "$0")"
for seed in 2 3 4 5 6 7 8 9; do
  echo "### seed $seed START $(date -u +%H:%M:%S)" >> var_loop.log
  python -u var_run.py "$seed" 5000 > "run_seed${seed}.log" 2>&1
  echo "### seed $seed END   $(date -u +%H:%M:%S) (exit $?)" >> var_loop.log
done
echo "### ALL DONE $(date -u +%H:%M:%S)" >> var_loop.log
