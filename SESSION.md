# 5x5 GMG + regret-UED — results summary

Goal: demonstrate **goal misgeneralisation (GMG)** in the pottery-shop gridworld and
mitigate it with **regret-based UED**. Result: **DR (GMG) < PLR < ACCEL** on wall
behaviour, with ACCEL's adversarial editor learning to build the urn-walls.

## Configuration
- Env: 5x5, fixed bin; `shard_mean=1.7`, `urn_mean=1.3` (geometric, floored at 1).
- Reward: `BREAK_PENALTY=3.0`, `STEP_COST=0.02`, `SHAPING_COEFF=0.5`, `BIN_REWARD=1.0`,
  `WASTE_PENALTY=0.0`, `DISCOUNT_RATE=0.995`.
- PPO: `num_envs=4096`, `num_env_steps=64`, `num_epochs=1`, `num_minibatches=32`,
  `lr=0.003`, `entropy_coeff=0.01`, `buffer_capacity=32768`.
- Oracle DP-size cap `2^shards * 3^urns < 1e6` (in `generate.py` + the walk editor)
  keeps the exact regret solver tractable.
- Methods: **DR** `replay_prob=0`; **PLR** `replay_prob=0.5, train_on_generate=False`,
  raw regret; **ACCEL** = PLR + walk editor (`edit_prob=0.3, num_edits=1`), raw regret.
  Budget = 5000 gradient updates each.

## Headline results (equal gradient budget, final-window means)

| metric | DR | PLR | ACCEL |
|---|---|---|---|
| wall regret (stochastic) | 2.91 | 1.60 | **0.95** |
| wall regret (greedy) | 2.92 | 2.80 | **2.09** |
| wall break rate | 0.04 | 0.59 | **0.86** |
| in-distribution (random) regret | 0.010 | 0.010 | 0.026 |
| buffer mean urns | — | 2.5 | **5.5** |

- **DR misgeneralises**: competent in-distribution (regret 0.01) but never breaks the
  deployment urn-walls (break 0.04, regret 2.9) — it competently takes the long way around.
- **PLR partially fixes** it (break 0.59).
- **ACCEL is strongest**: lowest wall regret, highest break, and its editor *builds*
  walls — buffer mean urns 5.5 vs PLR's 2.5.

## PLR run-to-run variance (3 independent seeds, identical config)

| seed | wall regret | break |
|---|---|---|
| 1 | 1.60 | 0.59 |
| 2 | 2.26 | 0.41 |
| 10 | 1.25 | 0.81 |

PLR's outcome is **high-variance** (break 0.41–0.81 at identical config). Buffer
enrichment is consistent (~2.5 urns) across seeds — the **policy** is what varies.

## Extended ACCEL (5000 -> 10000 gradient updates)
Final: stochastic wall regret **0.66 / break 0.85**, greedy wall regret **0.56**, random
regret 0.02. The trajectory **oscillates** (~0.6 near-solved ↔ ~2.0 degraded) and does
**not** converge to a stable fully-solved policy; extra compute raises the achievable
best but does not stabilise it.

## Key findings
1. **GMG reproduced and mitigated**: DR < PLR < ACCEL on both wall regret and break rate;
   ACCEL's adversarial editor builds the rare urn-walls random generation almost never
   produces (buffer 5.5 vs PLR 2.5 mean urns).
2. **Regret-UED is unstable late** on this task: PLR's wall performance peaks early then
   degrades, and ACCEL oscillates. Report windowed/best performance, not a single final
   eval (single eval points swing 0.2–2.0).
3. **PLR is high-variance** run-to-run; multiple seeds are needed for any quantitative claim.
4. **Score normalisation hurts**: ranking buffer levels by `regret/(optimal-worst)`
   deprioritises urn-walls (their achievable range is ~3x larger than a sparse level's),
   slowing wall learning. Use **raw** regret.
5. **The entropy bonus is load-bearing**: removing it collapses the agent back toward the
   non-breaking proxy (break 0.78 -> 0.22).

## Artifacts (`data5/`)
Per run: agent + buffer + training-history + eval-CSV + log.
- `dr_5000`, `plr_p50_5000`, `accel_walk_5000_nonorm`, `accel_walk_5000_nonorm_cont`
- `plr_p50_5000_ent0` (entropy-0), `plr_p50_5000_norm` (normalised), `plr_p50_peak2500`
- PLR variance seeds 2 and 10: `data5/remote_var/`, `data5/remote_var_luna/`
