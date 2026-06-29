# Session handoff — 5x5 GMG + regret-UED experiments

**Status as of:** 2026-06-29. **Branch:** `experiments-5x5-ued` (commit history under it; not pushed). **Update this file as runs complete / state changes.**

Goal: show goal-misgeneralisation (GMG) in the pottery-shop gridworld and mitigate it
with regret-based UED. Headline story confirmed: **DR (GMG fails) < PLR (partial fix)
< ACCEL (best)**, with the ACCEL buffer demonstrably *building* urn-walls.

---

## Fixed configuration (all runs unless noted)
- **Reward** (`rewards.py`): `BREAK_PENALTY=3.0`, `SHAPING_COEFF=0.5`, `BIN_REWARD=1.0`,
  `STEP_COST=0.02`, `WASTE_PENALTY=0.0`, `DISCOUNT_RATE=0.995`.
- **Env**: `world_size=5`, `shard_mean=1.7`, `urn_mean=1.3` (geometric, floored at 1).
- **PPO/UED** (`run5.py` / `var_run.py`): `num_envs=4096`, `num_env_steps=64`,
  `num_epochs=1`, `num_minibatches=32`, `buffer_capacity=32768`, `lr=0.003`,
  `entropy_coeff=0.01`, `seed=1` (local runs).
- **Solver OOM cap** (critical): `generate.py` clamps every level to
  `2^shards * 3^urns < 1e6` (reduce urns first); `editor.py` `_cap_state` guards
  toggle-mode urn ratchet. Without this the exact oracle DP OOMs on dense tail levels
  (a single >40GB tensor). Walk-mode edits conserve urns and need no cap.
- **Step budget convention**: CLI arg = DR-equivalent *gradient updates*. DR
  `replay_prob=0` -> steps = arg. PLR/ACCEL `replay_prob=0.5` -> steps = 2*arg.
  So `plr50 5000` and `accel_walk 5000` run 10000 steps = 5000 grad updates.

## Methods (curricula in `run5.py`)
- **DR**: `replay_prob=0`, train every step. (`dr`)
- **PLR (-bot)**: `replay_prob=0.5`, `train_on_generate=False`, raw regret. (`plr50`)
- **ACCEL**: PLR + walk editor `edit_prob=0.3, num_edits=1, edit_mode="walk"`, raw
  regret. (`accel_walk`) — **normalisation OFF** (see findings).

---

## Completed runs + headline (all artifacts in `data5/`)

Three-way, final-window means (DR at 4000-5000; PLR/ACCEL at 8000-10000 = equal grad budget):

| metric | DR | PLR | ACCEL |
|---|---|---|---|
| wall regret (stoch) | 2.91 | 1.60 | **0.95** |
| wall regret (greedy) | 2.92 | 2.80 | **2.09** |
| wall break rate | 0.04 | 0.59 | **0.86** |
| random regret (in-dist) | 0.010 | 0.010 | 0.026 |
| buffer mean urns | — | 2.54 | **5.48** |

`data5/` run names: `dr_5000`, `plr_p50_5000`, `accel_walk_5000_nonorm` (+ each has
agent/buffer/history `.pt`, `history_eval_*.csv`, `run_*.log`).

## Key findings
1. **Opposite temporal dynamics (the real story).** PLR *peaks early then forgets*:
   wall regret 0.48 / break 1.00 at step ~2500, degrading to 1.83 / 0.50 by 10000
   (buffer keeps enriching: buf_urns 1.3->2.56). ACCEL *learns slowly then converges*:
   regret ~2.4 through step 8000, dropping to 0.72 / break 0.94 by step 10000, and
   **still improving at the cutoff** (buf_urns ->5.7). => final-window numbers undersell
   both. Use windowed (~300-500 step) eval, never single points (they swing 0.2->2.0).
2. **Normalisation hurts — killed.** `plr_p50_5000_norm` killed at step 1294.
   `normalise_score` divides regret by `(optimal - worst_return)`, which is ~3x larger
   for a 5-6 urn wall than a sparse level, so it *deprioritises the urn-walls* in the
   buffer. Confirmed: buf_urns lagged (1.90 vs non-norm 2.17 at matched step), 4x worse
   wall regret. **Preferred setting: normalisation OFF** (carried into ACCEL).
3. **Entropy is load-bearing.** `plr_p50_5000_ent0`: fine-tuning the step-10000 PLR
   agent with `entropy_coeff=0` for 1000 steps *regressed* wall break 0.78->0.22 (policy
   sharpened toward the majority "walk-around" mode). Keep `entropy_coeff=0.01`.
4. **CUDA non-determinism => PLR is high-variance.** A fresh seed-1 PLR
   (`plr_p50_peak2500`) did NOT reproduce the original trajectory (break ~0.67 vs 1.0
   at step 2500). The **buffer reproduces** (CPU-deterministic sampling + exact oracle),
   the **policy does not** (non-deterministic GPU kernels, amplified by RL chaos).
   Divergence visible by ~step 1000 (regret metric by ~300-500). => the single-seed
   "PLR forgets" claim needs multiple seeds (in progress). `peak2500` agent is NOT a
   peak; capturing a true peak needs save-best-eval-checkpoint logic (not yet built).

---

## PLR variance runs (COMPLETE) — `data5/remote_var{,_luna}/`
Two boxes (octavia seeds 2-9, luna seeds 10-17) ran fresh non-norm PLR, 10000 steps,
net-init+train seed = SEED, logging to the user's W&B. **Loops stopped after one seed
each ("no time"); no further seeds.** Final eval (step 10000), with the original
seed-1 PLR for comparison:

| run | wall regret (stoch) | wall break | buf_urns | W&B |
|---|---|---|---|---|
| seed 1 (orig `plr_p50_5000`) | 1.60 | 0.59 | 2.54 | — |
| seed 2 (octavia) | **2.26** | **0.41** | 2.54 | `v6deksg7` |
| seed 10 (luna) | **1.25** | **0.81** | 2.53 | `0hvf47gw` |

=> **Large run-to-run variance confirmed** (break 0.41 vs 0.81 vs 0.59 at the same
budget/config). PLR's outcome is a high-variance draw, not a stable result — the
single-seed "peak-then-forget" story is one realisation among a wide spread. Buffer
enrichment is consistent (~2.5 urns) across seeds; the *policy* is what varies.
Artifacts (agent/buffer/history/log) committed under `data5/remote_var/` (seed2) and
`data5/remote_var_luna/` (seed10).

## ACCEL continuation (`accel_walk_5000_nonorm_cont`) — RUNNING (~done), partial committed
- Warm-started from `accel_walk_5000_nonorm`, `step_offset=10000`, +10000 steps
  (logged 10000->20000). W&B `jbjavv6h`. Log `run_accel_cont.log`. Artifacts in
  `data5/` (`*_accel_walk_5000_nonorm_cont.*`) at the latest 100-step checkpoint.
- **Finding: oscillating plateau, NO clean convergence.** Windowed wall regret/break
  over the continuation:

  | logged step | wall regret | break |
  |---|---|---|
  | 10-11k | 1.74 | 0.54 |
  | 12-13k | **0.63** | **0.98** (best, ~solved) |
  | 16-17k | 0.62 | 0.84 |
  | 18-19k | 1.98 | 0.45 (worst) |
  | 19-20k | 0.95 | 0.92 |

  More ACCEL steps do **not** drive it to fully-solved (regret->0); it swings between
  ~0.6 (near-solved) and ~2.0 (degraded) — the same buffer-churn instability as PLR.
  (`buf_urns` ~6.6-8; the editor over-densifies.) NB: an earlier "monotonic regression"
  read was a single noisy eval point — always window (~1000 steps), never trust one
  eval. Will refresh artifacts + this section if the run finishes in time.

---

## Infrastructure
- **Remote box**: `ssh arena8-octavia` (same GPU/timeout as local). Deps `jaxtyping`,
  `einops` installed. Code + `solver_cache/` (88M, warms solves) + `sprites.png` in
  `~/capstone-florian`. W&B: machine's own netrc is account `dquarel`; we inject the
  user's key from `~/capstone-florian/.wandb_key` (0600) via `WANDB_API_KEY` for the
  loop only (dquarel netrc untouched).
  - **Relaunch the loop**: `cd ~/capstone-florian && export WANDB_API_KEY=$(cat .wandb_key)
    && nohup bash var_loop.sh > var_loop.out 2>&1 < /dev/null &`
- **Local->local sync of remote results**: background loop, every 20 min, rsyncs
  `history_/agent_/buffer_plr_p50_seed*.pt` + `run_seed*.log` to `data5/remote_var/`.
- **Checkpointing**: every 100 steps each run rewrites agent + buffer + history +
  shared `solver_cache/`. A crashed/continued run resumes via `WARM_START`.

## How to resume after the machine dies
1. All completed artifacts are committed under `data5/` on branch `experiments-5x5-ued`.
2. To **continue any run**: in `run5.py` set `WARM_START=True`, `WARM_START_FROM=<run name>`,
   and a `STEP_OFFSET` = the run's last logged step (for a seamless W&B curve); launch the
   same method/budget. Buffer+cache load automatically.
3. `run5.py` is currently in the **ACCEL-continuation** config block (warm-start from
   `accel_walk_5000_nonorm`). Revert the marked block for a fresh run.
4. Remote variance agents land two levels deep (`data5/remote_var/agent_*.pt`) where
   `.gitignore`'s `!*/*.pt` doesn't reach — `git add -f` them when committing.

## Open questions / next steps
- Does ACCEL converge to regret->0 / break->1 with more steps? (continuation running)
- Is PLR's "peak-then-forget" robust across seeds, or just high variance? (variance runs)
- *Why* does PLR forget — buffer/curriculum drift, staleness, LR? (not yet diagnosed)
- Multi-seed error bars on the DR/PLR/ACCEL headline (single-seed so far).
- Build save-best-eval-checkpoint to capture a true PLR peak agent for forensics.
- The greedy-vs-stochastic eval gap (PLR's greedy advantage over DR is small) — eval set
  is only 27 walls; consider a larger held-out wall set + averaged rollouts for less noise.
