# Pottery-shop GMG — training & reward-tuning results

Working log of the autonomous training/reward-tuning effort. Metric throughout is
**oracle regret** = `optimal_return − policy_return` (exact, from `solver.py`),
evaluated greedily on fixed held-out sets:
- **random**: held-out layouts from the training distribution (the high-urn tail
  is where the agent fails — the "rare environments");
- **walls**: hand-built urn-wall deployment levels (`evalsuite.wall_envs`).

We report mean regret, the fraction of levels solved (regret < 0.05), max regret
(the tail), and the **break-through rate** (fraction of levels where the greedy
policy smashes ≥1 urn).

Tooling added this session: `evalsuite.py` (held-out eval sets + regret/break
metrics), a decoupled periodic-eval hook in `train_agent_multienv`
(`eval_fn`/`eval_every`), reward parameters made tunable in one place
(`rewards.set_reward_params`, read dynamically by the oracle so they never drift),
`solver.compute_optimal_return_grouped` (groups by exact (#shards,#urns) so a
dense outlier can't OOM the batch solve), `experiment.py` (unified
train+eval+W&B+save driver), `reward_analysis.py` (oracle reward-landscape map),
`hpo_regret.py` (HPO scored on regret, not just return).

## Key conceptual finding: breaking is rarely optimal on 4×4

Using the exact solver (optimal vs. optimal with breaking suppressed):

- On **4×4 at horizon 64, breaking an urn is never optimal at the default reward**
  (bp=3, step=0.02) — even on the hand-built walls. Walking around a 4×4 wall is
  short enough to always win. So "wall regret" at the default reward measures OOD
  *navigation*, not a refusal to break (GMG).
- On **5×5, breaking is optimal** on the deeper walls (the detour overruns/costs
  more) — GMG is natural there.
- The reward knobs move the boundary (oracle, `reward_analysis.py`, 4×4):

  | break_penalty | step_cost | walls needing breaking | breaking optimal in training |
  |---:|---:|:--:|--:|
  | 3.0 | 0.02 | 0/3 | 0.25% |
  | 3.0 | 0.20 | 3/3 (gaps ≤3.45) | 0.75% |
  | 1.0 | 0.02 | 3/3 (gaps ≤1.52) | 1.05% |
  | 0.5 | 0.02 | 3/3 | 99.75% |

  → **bp=3.0, step=0.2** = GMG trap (walls need breaking, training never does).
  → **bp=0.5** = breaking optimal almost everywhere (agent should be optimal incl.
  walls). → bp=1.0 = walls need breaking but the training signal is ~1% (the
  paper's *rarity threshold*).

## Stage 1 — training-hyperparameter HPO (scored on regret)

Prior session tuned for return-per-time; this session re-checks the tuned config
against rare-env *regret*. (200 steps each, 4×4, default reward.)

| num_envs | mb | upd/coll | steps | wall(s) | rand regret | rand solved | rand max | wall regret |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 16384 | 32 | 200 | 118.8 | 0.086 | 0.953 | 6.38 | 2.98 |
| **16384** | 16384 | 64 | 200 | 236.3 | **0.016** | **0.982** | **4.27** | 1.64 |
| 8192 | 8192 | 64 | 200 | 211.8 | 0.068 | 0.959 | 4.89 | 2.98 |
| 4096 | 8192 | 32 | 200 | 112.6 | 0.045 | 0.971 | 4.30 | 2.86 |

**Takeaway:** the rare-env tail is driven by **environments-per-step (layout
diversity)**, not updates-per-collection. 16384 envs clearly wins (regret 0.016,
98.2% solved) — doubling updates at fixed envs (config 3) barely helped and cost
2× time. The 4096–8192 configs cluster within single-seed tail noise. **Long-run
config: 16384 envs / mb 16384 / lr 0.003** (64 updates/collection, safely below
the >256 collapse zone). The fast 8192/16384 config is used for the short
reward-tuning runs.

`wall_break`/`wall_regret` at the default reward are not GMG signals (breaking is
never optimal on 4×4 walls) — they reflect OOD navigation only.

## Stage 2 — long 4×4 run (improve the rare-env tail)

`experiment.py --name long-4x4-16384 --steps 5000 --num-envs 16384 --wandb`,
**97 min**, default reward. Logged to W&B (`arena8-capstone/runs/wc1y9dwk`), saved
to `agent_long_4x4.pt` (+ sidecar). Held-out regret over training (fixed 512-env
set; final row is a 4000-env eval):

| step | random regret | random solved | random max (tail) | walls regret |
|---:|---:|---:|---:|---:|
| 0    | 2.510 | 0.002 | 9.38 | 3.98 |
| 250  | 0.037 | 0.977 | 3.15 | 2.98 |
| 1000 | 0.008 | 0.992 | 1.74 | 1.80 |
| 2000 | 0.005 | 0.992 | 1.76 | 1.80 |
| 4000 | 0.0008 | 0.994 | 0.14 | 1.80 |
| **final (n=4000)** | **0.013** | **0.992** | 5.71 | **1.80** |

The rare-env tail shrinks substantially: the fixed-set max regret falls 9.4 → ~0.1
and `solved` rises 97.7% → 99.2%. A 4000-env final eval still finds a handful of
very-rare hard layouts (max 5.71), i.e. brute-force training has **diminishing
returns on the rarest layouts** — exactly the motivation for regret-based curricula
(PLR/ACCEL), the project's next phase. Walls improved (2.98 → 1.80, 1/3 solved) but
stay an OOD-navigation problem (breaking is never optimal on 4×4 walls, so more
in-distribution training can't teach a wall-specific behaviour).

## 5×5 — GMG appears at the default reward (oracle)

`reward_analysis.py 5 96 1000`, default reward (bp=3, step=0.02): **2/3 walls
require breaking** (gaps [0, 2.68, 4.17]) while breaking is optimal in only 0.70%
of training. So on 5×5, GMG is natural *without any reward change* — the
default-reward agent should be competent in-distribution yet fail on the walls
(it never learned to break). This makes the 5×5 the cleaner GMG testbed: the
default long run is the GMG case (experiment B); lowering the break penalty is the
fix (experiment A), same mechanism as 4×4.

**How much longer does 5×5 take?** Per-step is ~1.5–2× slower (horizon 96 vs 64,
25 vs 16 cells) and it needs more steps to converge: at 50 steps the 5×5 agent is
at random regret 0.293 / 81% solved, where 4×4 by ~200 steps is already
≤0.09. Estimate **~3–4× the 4×4 wall-time** for comparable in-distribution
convergence. No OOM at 8192 envs / 96 horizon. ### 5×5 default-reward long run (the GMG demonstration)

`experiment.py --world-size 5 --steps 3000 --num-envs 8192 --wandb`, **64.5 min**
(~1.29 s/step), saved `agent_long_5x5.pt`, W&B `arena8-capstone/runs/f0fnafnp`.

| step | random regret | random solved | walls regret | walls break |
|---:|---:|---:|---:|---:|
| 0    | 3.187 | 0.000 | 3.89 | 0.00 |
| 500  | 0.032 | 0.971 | 3.89 | 0.00 |
| 1500 | 0.032 | 0.975 | 3.89 | 0.00 |
| 2750 | 0.037 | 0.973 | 3.89 | 0.00 |
| **final (n=2000)** | **0.030** | **0.979** | **3.89** | **0.00** |

**Clean GMG, induced by nothing but the default reward + grid size:** the agent is
competent in-distribution (random regret 0.03, 98% solved) yet **fails completely
on the walls** (regret 3.89, never breaks, achieved −1.53 vs optimal 2.36). Because
breaking is optimal in only 0.7% of training, `walls_regret` stays **flat at 3.89
for the entire 64 min** — more training does **not** cross the rarity threshold.
This is the project's core GMG claim, and the motivation for a regret-based
curriculum that *manufactures* wall-like high-regret levels (PLR/ACCEL).

**"How much longer" than 4×4:** ~1.29 s/step (8192 envs, horizon 96) vs ~0.6 s/step
(8192) / ~1.18 s (16384) on 4×4; and 5×5 plateaus higher (random regret ~0.03 vs
4×4's 0.013) — it needs several× the 4×4 steps to approach the same convergence and
still leaves a larger tail. Budget a 5×5 run at roughly 3–4× the 4×4 wall-time.

### 5×5 reward fix (A: bp=0.5) — partial

`experiment.py --world-size 5 --steps 1500 --break-penalty 0.5 --wandb`, 32.6 min,
saved `agent_5x5_rtA.pt`, W&B `runs/33r7octb`.

| | random regret | random solved | walls regret | walls break | walls achieved vs optimal |
|---|---:|---:|---:|---:|---:|
| default 5×5 (above) | 0.030 | 0.98 | 3.89 | **0.00** | −1.53 vs 2.36 |
| **bp=0.5 5×5** | 0.012 | 0.97 | 7.13 | **1.00** | −2.02 vs 5.10 |

Lowering the break penalty **fixes the misgeneralizing behaviour** — break-rate on
the walls goes 0 → 1.0, so the agent now attempts to break through instead of
refusing — and it stays competent in-distribution (regret 0.012). **But it does not
reach optimality on the 5-urn walls** (achieved −2.02 vs optimal +5.10): at bp=0.5
the wall optimum is a long "smash all 5 urns, then pick up and deliver every
resulting pile" routing, which is far OOD from the scattered-urn training
distribution, and the agent breaks but can't execute the cleanup. Wall regret is
flat across training, so more in-distribution training won't close it.

Contrast with 4×4 (2–3 urn walls), where bp=0.5 reached **wall regret 0.0**: reward
shaping fully solves the small walls but only the *incentive* half of the large
ones. The residual is a capability/OOD-generalization gap that needs training *on*
wall-like levels — i.e. PLR/ACCEL.

## Conclusion

1. **Training HPO:** the rare-env tail is governed by environments-per-step
   (layout diversity); 16384 envs is the best 4×4 config (regret 0.016 @ 200 steps).
2. **Long run (4×4, 97 min):** drives random regret to 0.013 / 99.2% solved and
   collapses the tail (max 9.4 → ~0.1 on the fixed set), but a 4000-env eval still
   finds rare unsolved layouts — brute force has diminishing returns on the rarest
   levels.
3. **Reward tuning (4×4):** the break penalty is the GMG dial. **bp=3.0 + step=0.2
   → maximal GMG** (competent in training, regret 4.07 on walls it refuses to
   break); **bp=0.5 → optimal everywhere** (breaks through all walls, regret 0.0)
   — reward design solves the walls in 5 min, which no amount of default-reward
   training can (**answer to "match/exceed the long run": yes, decisively**).
4. **5×5:** GMG appears **at the default reward** (walls need breaking, training
   almost never does) and **persists through 64 min of training** (rarity
   threshold). Lowering the break penalty flips the behaviour (now breaks through)
   but doesn't fully solve the harder 5-urn walls.
5. Across both grid sizes the residual failures are exactly the high-regret OOD
   walls that an adversarial curriculum (ACCEL) is designed to manufacture and
   train on — the project's next phase.

### Artifacts
Saved nets (+ `.json` sidecars with config & final metrics):
`agent_long_4x4.pt`, `agent_long_5x5.pt`, `agent_baseline_default.pt`,
`agent_rtB_gmg.pt` (4×4 max-GMG), `agent_rtA_optimal.pt` (4×4 bp=0.5),
`agent_rtA2_rarity.pt` (4×4 bp=1.0), `agent_5x5_rtA.pt` (5×5 bp=0.5).
W&B project `arena8-capstone`: runs `long-4x4-16384`, `long-5x5-default`,
`rtA-5x5-bp0.5`. Logs: `hpo_regret_batch1.log`, `reward_batch.log`, `long_*.log`,
`reward_analysis*.jsonl`, `experiment_summary.jsonl`.

## Stage 3 — reward tuning (4×4)

Two targeted experiments (see reward landscape above), each trained 500 steps
(~5 min) at 8192 envs, plus a same-budget default-reward baseline. Regret is
measured under each run's own reward (i.e. vs the oracle optimum for that reward).

| run | break_penalty | step_cost | random regret | random solved | wall regret | wall break-rate | wall achieved |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (default) | 3.0 | 0.02 | 0.045 | 0.97 | 2.98 | 0.00 | −1.10 |
| **B: GMG trap** | 3.0 | 0.20 | 0.117 (95% solved) | 0.95 | **4.07** (0% solved) | 0.33 | −9.09 vs opt −5.02 |
| **A: all-optimal** | 0.5 | 0.02 | 0.023 (96% solved) | 0.96 | **0.00** (100% solved) | 1.00 | 4.04 = opt 4.04 |
| A2: intermediate | 1.0 | 0.02 | 0.011 (95% solved) | 0.95 | 0.43 (67% solved) | 1.00 | 2.49 vs opt 2.92 |

**A2 (bp=1.0):** breaking is net-zero-cost, so the agent learns to break on all
walls (break-rate 1.0) and solves 2/3 (regret 0.43). The break penalty smoothly
sets the break/avoid threshold: **bp=0.5 → all walls solved (0.0); bp=1.0 →
mostly (0.43); bp=3.0 → never break → GMG (B, 4.07).** The clean rarity-threshold
*failure* is B: there breaking is optimal on the walls but the high penalty keeps
training break-rate ~0, so the agent never learns it.

**Answer to "can reward tuning match/exceed the long run?": yes, decisively.**
Lowering the break penalty makes the walls fully solvable (regret 0) in ~5 min,
an outcome no amount of default-reward training can reach (the long run below
drives the random tail down but leaves the walls an unsolved OOD problem).

**A (optimal everywhere):** at bp=0.5 breaking is net-positive, so the agent learns
to break (99% train break-rate) and **solves every wall to zero regret** (breaks
straight through, achieved = optimal = 4.04) while staying near-optimal in
training. The reward fix makes "break through the wall" the natural optimal policy
— GMG is eliminated. This **matches/exceeds the long run**: 5.5 min of reward-tuned
training fully solves the walls, which more training under the default reward
cannot (there the walls aren't even breakable-optimal and the agent fails to
navigate them).

**B (max GMG):** competent in-distribution (random regret 0.117) yet fails on the
walls where breaking is now optimal (regret 4.07, never solved) — the agent
learned the "rarely break" proxy (4% train break-rate) and can't exploit the
breaking shortcut. This is the misgeneralization gap, induced purely by reward.

Note: at the default reward the baseline agent doesn't just walk around the walls —
it **fails to navigate them at all** (achieved −1.10, never delivers), because the
walls are strongly OOD (robot+bin in one corner, contiguous urns vs. scattered).

(B/A/A2 pending — batch running)
