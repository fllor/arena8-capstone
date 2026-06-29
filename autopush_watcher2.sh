#!/usr/bin/env bash
# Resumed auto-push: GPU outlived the 9am estimate and ACCEL is still training.
# Push checkpoints every 20 min while the suite runs; stop when no run.py remains
# (suite done) or at a far safety deadline, then do a final push.
set -u
cd /root/arena8-capstone
BRANCH=qi/5k-convergence
DEADLINE=$(date -d '2026-06-29 18:00' +%s)

commit_push () {
  git add -A 2>/dev/null
  git add -f agent_*_ckpt.pt agent_*.pt 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "$1 $(date -Is)" 2>/dev/null \
      && git push -q origin "$BRANCH" 2>/dev/null \
      && echo "pushed ($1) $(date -Is)" || echo "push FAILED $(date -Is)"
  else
    echo "no changes $(date -Is)"
  fi
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 1200   # 20 min
  commit_push wip
  pgrep -f "run.py" >/dev/null || { echo "suite finished $(date -Is)"; break; }
done
commit_push final
echo "watcher2 done $(date -Is)"
