# Goal Misgeneralisation + Adversarial UED — Project Guide

A 5-day, 2-person RL project. Goal: get **goal misgeneralisation (GMG)** to appear
in the "pottery shop" gridworld, then **mitigate it with regret-based UED**
(curriculum / adversarial environment design), in the simplest possible form.

## The story we are demonstrating

- The robot navigates a gridworld. Stepping onto an **urn** smashes it into
  shards. There is a **break penalty** (`reward_no_break = -2 * reward_break`).
- In **random training** layouts, urns are scattered, so walking *around* them is
  almost always optimal and breaking is almost never worth it.
- At **deployment**, urns can form a **wall** between robot and bin, where
  breaking through is the optimal shortcut. An agent trained only on random
  layouts learns the proxy "never break urns" and **misgeneralises** — it
  competently takes the long way around instead of breaking through.
- **Fix:** train harder on the rare high-regret layouts (the urn-walls). Random
  replay (PLR) can only amplify walls it happens to sample; an **adversarial
  editor (ACCEL)** can *build* walls, so it is the more robust approach.

## Critical conceptual note (do not get this wrong)

The right objective is **minimax regret, NOT minimax return**.
`regret = optimal_achievable_return − policy_return`. Selecting environments by
*low return* is a trap: it favours unsolvable/degenerate layouts (e.g. bin walled
off completely) and training collapses. High *regret* = "the agent did much worse
than it could have", which is exactly the solvable urn-walls we care about.

The pottery shop is small, so the **optimal return is exactly computable**
(shortest path via Dijkstra, where stepping onto an urn cell costs the break
penalty). This lets us use an **oracle regret estimator** and skip the literature's
hardest, most fragile part (biased sample-based "max-latest" estimation).
Always keep generated/edited deployment levels **solvable**.

## Project plan (in order)

1. **Get GMG to appear.** Tune break penalty / discount / per-step cost so
   breaking is *occasionally* optimal in training but *rare*, and clearly optimal
   on hand-built urn-wall deployment levels. Confirm a random-trained agent
   misgeneralises (competent, but walks around the wall). This is the make-or-break
   gate.
2. **PLR⊥ (prioritised level replay).** A level buffer keyed by oracle regret;
   replay high-regret levels for PPO updates. Expect limited gains when walls are
   rare in random gen (the "rarity threshold") — that is the motivation for step 3.
   If PLR looks weak, the fix is to **seed the base generator with occasional
   wall-like structures** so the buffer has something to find — do *not* burn slack
   concluding the method is broken.
3. **ACCEL (adversarial editor).** Edit high-regret buffer levels (toggle cells
   to/from `URN`, **unrestricted** so it can create walls), re-score regret,
   re-admit. This is the strongest, most robust result. The "money shot" is the
   buffer learning to build walls.

### 5-day milestones (3-day technical core + 2 slack days)
The codebase already works, so the technical work fits in 3 days with 2 days of
slack/writeup. **Parallelise across the 2 people** — Day 1's two tracks are
independent, which removes the main spill risk. Day 1 is the highest-risk day and
the most likely place the slack gets spent.

- **Day 1 (GMG + solver, in parallel):**
  - *Person A:* confirm the baseline trains (run `run.py` cell-by-cell), then tune
    rewards/discount until a random-trained agent demonstrably **misgeneralises**
    (competent, but walks around a hand-built urn-wall). This is the make-or-break
    GATE — lots of waiting on training runs.
  - *Person B:* write the **oracle Dijkstra regret solver** (optimal return =
    shortest path with urn cells costing the break penalty) + build solvable
    urn-wall deployment levels. No trained agent needed, fully parallel. (Pulled
    into Day 1 so it does not lurk inside Day 2.)
- **Day 2 — PLR⊥:** level buffer keyed on oracle regret, prioritised replay,
  plugged into the `gen` seam; compare vs DR. Build the eval/plotting harness.
- **Day 3 — ACCEL:** edit operator (toggle cells to `URN`, unrestricted) +
  solvability filter on edits; three-way DR vs PLR vs ACCEL comparison + buffer viz.
- **Day 4–5 — slack + writeup:** absorb Day-1 spill, then 2–3 seeds, freeze
  headline figures, write-up.

If behind, cut in order: ACCEL → multi-seed → PLR. Minimum story = GMG gate +
one of {PLR, ACCEL}. ACCEL alone (skipping PLR) is a legitimate fallback.

## Files in this directory

Self-contained except library imports (`torch`, `jaxtyping`, `tqdm`). Imports are
flat (`from potteryshop import ...`); **run scripts from this directory**.

| File | Contents |
|------|----------|
| `potteryshop.py` | Env dynamics, `State`/`Observation`/`Environment`, batched `step`/`observe`/`reset`, rollout collection. |
| `agent.py` | `ActorCriticNetwork` (residual CNN + MLP, actor/critic heads). |
| `ppo.py` | Multi-env PPO step + GAE + clipped-surrogate loss. |
| `evaluation.py` | `RewardFunction`, `compute_return`, `evaluate_behaviour`. |
| `rewards.py` | `reward2` (intended reward) + chain, `DISCOUNT_RATE`. |
| `generate.py` | `generate()` — FIXED-bin random layout distribution. |
| `train.py` | `train_agent_multienv()`, `default_device()` — reusable library, no driver code. |
| `run.py` | Interactive `# %%` driver: build → train → plot → evaluate. Imports from the rest. |

Smoke-tested on CUDA: trains across 32 parallel envs, return climbs.

### The key extension seam
`train_agent_multienv(gen, ...)` calls `gen(num_envs=, generator=)` each step to
get a fresh batch of `Environment`s. **Swap `generate` for a buffer/curriculum/
adversarial generator with the same signature** — that is the entire insertion
point for PLR and ACCEL. No changes to PPO needed.

### Notes on the extraction
Rendering/sprites were removed (no PIL/einops/sprite assets). Training is
headless: `train_agent_multienv` returns `(net, history)` (list of per-step
metric dicts) and prints progress every `log_every` steps. Device is a parameter
(`default_device()`: cuda→mps→cpu). The sampling `Generator` stays on CPU for
device-independent reproducibility. Only the fixed-bin `generate` is included.

## Environment cheat-sheet (`potteryshop.py`)
- `Item`: EMPTY=0, SHARDS=1, URN=2. `Action`: WAIT, UP, LEFT, DOWN, RIGHT, PICKUP, PUTDOWN.
- Batched: every `State`/`Environment` field has a leading batch dim `B`; `step`
  advances all envs at once. No walls — grid edges clamp movement; **stepping onto
  an urn smashes it to shards** (free mechanically; the penalty is in the reward).
- Obs: bool grid `[B, ws, ws, 4]` (robot/bin/shards/urns) + inventory vec `[B, 2]`.
- Rewards are external `RewardFunction(state, action, next_state) -> float[B]`,
  never baked into `step` — fully pluggable.

## Background
The approach follows Abdel Sadek, Farrugia-Roberts et al., *Mitigating Goal
Misgeneralization via Minimax Regret* (RLC 2025, arXiv:2507.03068) — full text at
`doc/2507.03068.txt`. Not a new
algorithm — it applies existing regret-based UED (**PLR⊥** and **ACCEL**) as a
defence against GMG. Student = PPO; adversary = level-buffer curation (no learned
adversary network). Each method has a rarity threshold below which it fails. There
is no released code; we reimplement here. This `code/` package was extracted from
an ARENA exercise (the GMG narrative and a worked corner/bin example live in that
exercise's notebook/solutions, one level up).

## Conventions
- Measure the **behavioural** outcome (does it break the wall?) with
  `evaluate_behaviour` + the `reward_break` probe, not just return — GMG is a
  behaviour claim.
- Per-run compute is tiny (small grids, small nets); confirm a GPU is available
  and a UED run finishes in minutes. If slow, shrink grid/net before scaling.
- After any reward/discount change, re-confirm the GMG gate (step 1) still holds.
