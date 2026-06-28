#!/usr/bin/env bash
# 5k-gradient-step convergence study: DR, PLR(no norm), ACCEL(no norm).
# wandb project "4x4 convergence 5k". ACCEL last (solver-blowup risk under non-norm).
set -u
cd /root/arena8-capstone

for M in dr plr50 accel_walk_smooth_nonorm; do
  echo "=== START $M $(date -Is) ==="
  MPLBACKEND=Agg python run.py "$M" 5000 > "conv_${M}_5000.log" 2>&1
  echo "=== END $M rc=$? $(date -Is) ==="
done
echo "=== CONVERGENCE SUITE COMPLETE $(date -Is) ==="
