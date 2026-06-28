#!/usr/bin/env bash
# Periodically commit+push training checkpoints to qi/5k-convergence so results
# survive losing the machine (not just a process kill). Stops at the GPU cutoff.
set -u
cd /root/arena8-capstone
BRANCH=qi/5k-convergence
DEADLINE=$(date -d '2026-06-29 09:05' +%s)

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 1800   # 30 min
  git add -A 2>/dev/null
  git add -f agent_*_ckpt.pt agent_*.pt 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "wip checkpoint $(date -Is)" 2>/dev/null \
      && git push -q origin "$BRANCH" 2>/dev/null \
      && echo "pushed checkpoint $(date -Is)" \
      || echo "push FAILED $(date -Is)"
  else
    echo "no changes $(date -Is)"
  fi
done
# final push after cutoff
git add -A 2>/dev/null; git add -f agent_*_ckpt.pt agent_*.pt 2>/dev/null
git commit -q -m "final checkpoint $(date -Is)" 2>/dev/null && git push -q origin "$BRANCH" 2>/dev/null
echo "watcher done $(date -Is)"
