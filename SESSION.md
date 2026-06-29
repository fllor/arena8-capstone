# 5x5 GMG + regret-UED — results summary

Goal: demonstrate **goal misgeneralisation (GMG)** in the pottery-shop gridworld and
mitigate it with **regret-based UED**. Result: **DR (GMG) < PLR < ACCEL** on wall
behaviour, with ACCEL's adversarial editor building the urn-walls.

## How to read these numbers (metric conventions)
- All eval numbers are **stochastic-policy, windowed means** over the run's final 2000
  steps (8000–10000 unless noted). Windowing averages out per-eval noise.
- **Never** quote a single eval point or single-step endpoint: per-point eval std is ~0.5
  (regret) — comparable to the gaps between methods, so a single point can flip a comparison.
- **Never** use greedy/deterministic eval: it is one rollout per level over a 27-wall set
  and is even noisier.
- Treat windowed differences smaller than the cross-seed std (~0.2 regret) as not meaningful.

## Configuration
- Env: 5x5, fixed bin; `shard_mean=1.7`, `urn_mean=1.3` (geometric, floored at 1).
- Reward: `BREAK_PENALTY=3.0`, `STEP_COST=0.02`, `SHAPING_COEFF=0.5`, `BIN_REWARD=1.0`,
  `WASTE_PENALTY=0.0`, `DISCOUNT_RATE=0.995`.
- PPO: `num_envs=4096`, `num_env_steps=64`, `num_epochs=1`, `num_minibatches=32`,
  `lr=0.003`, `entropy_coeff=0.01`, `buffer_capacity=32768`.
- Oracle DP-size cap `2^shards * 3^urns < 1e6` (in `generate.py` + the walk editor) keeps
  the exact regret solver tractable.
- Methods (budget = 5000 gradient updates each): **DR** `replay_prob=0`; **PLR**
  `replay_prob=0.5, train_on_generate=False`, raw regret; **ACCEL** = PLR + walk editor
  (`edit_prob=0.3, num_edits=1`), raw regret.

## Headline (windowed stochastic finals)

| metric | DR (n=1) | PLR (n=3) | ACCEL (n=1) |
|---|---|---|---|
| wall regret | 2.91 | 1.74 ± 0.18 | 0.95 |
| wall break rate | 0.04 | 0.68 ± 0.07 | 0.86 |
| random (in-distribution) regret | 0.01 | 0.01 | 0.03 |
| buffer mean urns | — | 2.50 ± 0.03 | ~5.5 |

(± is cross-seed std over PLR seeds 1/2/10; DR and ACCEL are single runs.)

- **DR misgeneralises**: optimal in-distribution (regret 0.01) yet essentially never breaks
  the deployment urn-walls (break 0.04) — it competently walks around them.
- **PLR partially fixes** it (break 0.68).
- **ACCEL is strongest**: lowest wall regret, highest break, and its editor *builds* walls
  (buffer ~5.5 vs PLR ~2.5 urns).
- The method gaps (~1.2 and ~0.8) are several times the per-method spread (~0.2), so the
  **ordering DR > PLR > ACCEL is robust** even allowing for the single-run uncertainty on
  DR and ACCEL.

## PLR run-to-run variance (3 independent seeds, windowed finals)
Seeds 1/2/10 — wall regret 1.60 / 2.00 / 1.63; break 0.59 / 0.70 / 0.76; buffer urns
2.54 / 2.47 / 2.48. Two distinct noise scales:
- **within-run** (single eval point): std ~0.5 — large, and the dominant noise source.
- **cross-seed** (windowed final): std **0.18** regret / **0.07** break — moderate.
- **buffer composition**: cross-seed std **0.03** — essentially reproducible (sampling +
  oracle are deterministic; only the *policy* varies between runs).

So PLR's *windowed* outcome is fairly consistent across seeds; the wild swings people see
are within-run eval noise, not run-to-run irreproducibility. A single run is trustworthy
*only* as a windowed mean, never as an endpoint.

## Extended ACCEL (5000 → 10000 gradient updates), n=1
The continuation **oscillates** (windowed wall regret swings 0.62 ↔ 1.98) rather than
converging: it touches ~0.62 in good windows but swings back to ~2.0, and its overall mean
(~1.1) is **no better than the original ACCEL final (0.95)**. More compute neither
stabilises nor improves it — the editor over-densifies the buffer (~7 urns) and the
resulting churn destabilises the policy. **This run's endpoint must not be read as a result.**

## Robust findings
1. **GMG reproduced and mitigated**: DR > PLR > ACCEL on wall regret and break (ordering
   robust: gaps ≫ cross-seed spread). ACCEL's editor builds the rare urn-walls random
   generation almost never produces.
2. **ACCEL builds denser walls** (buffer ~5.5 vs PLR ~2.5 urns) — the most trustworthy
   result, since buffer composition is near-reproducible (PLR cross-seed std 0.03), so the
   ACCEL-vs-PLR buffer gap is reliable signal rather than noise.
3. **Regret-UED is unstable late** on this task: all 3 PLR seeds peak early then degrade
   (windowed peak regret 0.55 / 1.06 / 1.20 → final 1.60 / 2.00 / 1.63), and extended ACCEL
   oscillates. Performance does not settle to a solved policy.
4. **Score normalisation deprioritises urn-walls**: ranking buffer levels by
   `regret/(optimal − worst)` divides by a ~3× larger denominator for walls than for sparse
   levels, so the buffer enriches walls more slowly (reproducible signal: 1.90 vs 2.17 mean
   urns at matched step). Use **raw** regret. (Run stopped early — no final-performance
   comparison; the claim rests on the mechanism + the reproducible buffer lag.)

## Not established / open
- **Entropy bonus**: a single 1000-step `entropy_coeff=0` fine-tune showed only a mild,
  noisy downward drift in wall break (~0.58 → 0.40 windowed) — suggestive that entropy helps
  sustain breaking, but a single short run at this noise level is **inconclusive**.
- **DR and ACCEL have no seed replication** (n=1); their exact values carry unquantified
  cross-seed uncertainty (likely ~0.2 by analogy with PLR). The ordering survives this; the
  precise values do not.
- Cheapest way to cut the dominant (within-run) noise: a **larger held-out wall set +
  averaged rollouts**, not more seeds.

## Artifacts (`data5/`)
Per run: agent + buffer + training-history + eval-CSV/log.
- `dr_5000`, `plr_p50_5000` (seed 1), `accel_walk_5000_nonorm`, `accel_walk_5000_nonorm_cont`
- PLR variance seeds 2, 10: `data5/remote_var/`, `data5/remote_var_luna/`
- Exploratory, not used for results: `plr_p50_5000_ent0`, `plr_p50_5000_norm`,
  `plr_p50_peak2500`
