#!/usr/bin/env bash
# 4x4 grid training suite: DR, PLR(no norm), PLR(norm), ACCEL — 1000 grad steps each.
# step cost 0.05 (rewards.py default), world_size 4 (run.py default), wandb project "4x4 grid runs".
set -u
cd /root/arena8-capstone

for M in dr plr50 plr50_norm accel_walk_smooth; do
  echo "=== START $M $(date -Is) ==="
  MPLBACKEND=Agg python run.py "$M" 1000 > "suite_${M}_1000.log" 2>&1
  echo "=== END $M rc=$? $(date -Is) ==="
done
echo "=== SUITE COMPLETE $(date -Is) ==="
