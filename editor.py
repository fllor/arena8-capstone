"""
ACCEL edit operator for the pottery shop.

ACCEL (Adversarially Compounding Complexity by Editing Levels; Parker-Holder et
al. 2022, arXiv:2203.01302) extends PLR-bot with a level *editor*: it mutates the
just-replayed high-regret levels, re-scores the children by oracle regret, and
offers them back to the buffer. Over generations this concentrates the buffer on
the rare urn-walls that random generation almost never *arranges* -- crossing the
"rarity threshold" where plain PLR stalls (the project's `CLAUDE.md` story).

`edit_levels` is the elementary edit distribution. Each edit picks two random
cells in one level and **swaps their contents**. This is a *count-conserving*
("move") edit distribution: it relocates shards/urns but never creates or
destroys them, so the per-item-type counts of a level are invariant under
editing. We pair it with the exact oracle regret estimator
(`solver.compute_optimal_return`), the paper's strongest, most robust
configuration ("oracle-latest").

Why move, not create? An *unrestricted create/destroy* editor (set a cell to any
item) ratchets urn density upward: early in training the agent is near-random, so
every extra urn it blunders into adds break-penalty regret, the buffer latches
onto denser levels, and editing compounds them toward an all-urn degenerate fill
(which also blows up the solver's `3^(#urns)` DP factor). The papers don't hit
this because their reward is bounded/sparse (regret doesn't scale with obstacle
count) and their editors are class/density-anchored (GMG paper App. J,
arXiv:2507.03068). Conserving counts removes the density axis entirely:

* **Density is owned by the generator, arrangement by the editor.** `generate`'s
  `urn_mean` sets how many urns a seed has (bounded, tunable, no feedback loop);
  the editor can only *rearrange* a seed's urns -- e.g. slide 4 scattered urns
  into a line to form a wall. Wall *width* is therefore capped by the seed's urn
  count, so raise `urn_mean` if walls need to be wider than `generate` supplies.
  generate+edit still reaches any level "in principle" (generate supplies the
  count, edit the arrangement) and finds wall arrangements far faster than
  generate alone.
* **No density explosion, so no solver blow-up and no urn cap needed.** Edited
  levels are exactly as dense as their parents.

Two further pottery-shop specifics:

* **No solvability filter.** Urns are *passable* (stepping smashes them to
  shards), so the grid graph is always fully connected and the oracle always
  returns a finite optimum -- every edit stays solvable by construction.
* **The robot is not on the map.** Robot position is a separate `Environment`
  field (`init_robot_pos`), so swaps can't move it; we also exclude the robot
  *and* bin cells from the swap so no item ever lands under the spawn or bin
  (`step` only clears an urn the robot moves *onto*, never one under the spawn).
  Robot-position diversity comes from `generate`, not edits.
"""

from __future__ import annotations

import torch

from potteryshop import Environment


def edit_levels(
    envs: Environment,
    *,
    num_edits: int,
    generator: torch.Generator,
) -> Environment:
    """
    Return a batch of edited copies of `envs` (the ACCEL mutation operator).

    For each level, applies `num_edits` elementary edits in sequence. Each edit
    picks two uniformly-random eligible cells -- any cell except the bin and the
    robot spawn -- and *swaps their contents*. This conserves the per-item-type
    counts of every level (a "move" edit), so editing changes arrangement but
    never density. If the two cells coincide (or hold the same item) the swap is
    a harmless no-op. The robot and bin positions are left untouched.

    Vectorised over the batch and grid-size agnostic. Randomness is drawn from
    `generator` (on its own device) and applied on the levels' device, so the
    caller's CUDA/CPU training generator works unchanged.
    """
    items = envs.init_items_map.clone()  # [B, ws, ws], long
    B, ws, _ = items.shape
    device = items.device
    gdev = generator.device
    cells = ws * ws
    flat = items.view(B, cells)  # shares storage with `items`
    batch = torch.arange(B, device=device)

    # Eligible = every cell except the bin and robot-start cells. Robot/bin are
    # separate Environment fields (never on the map), but an item must not land
    # on them, so mask their flattened indices out.
    robot_idx = (envs.init_robot_pos[:, 0] * ws + envs.init_robot_pos[:, 1]).long()
    bin_idx = (envs.bin_pos[:, 0] * ws + envs.bin_pos[:, 1]).long()
    eligible = torch.ones(B, cells, dtype=torch.bool, device=device)
    eligible.scatter_(1, robot_idx[:, None], False)
    eligible.scatter_(1, bin_idx[:, None], False)

    def pick() -> torch.Tensor:
        # One eligible cell per level via noise-argmax (a multinomial-free,
        # device-robust uniform draw over the eligible cells).
        noise = torch.rand(B, cells, generator=generator, device=gdev).to(device)
        return noise.masked_fill(~eligible, -1.0).argmax(dim=1)  # [B]

    for _ in range(num_edits):
        a, b = pick(), pick()  # [B], [B] -- two independent eligible cells
        va = flat[batch, a].clone()
        vb = flat[batch, b].clone()
        flat[batch, a] = vb
        flat[batch, b] = va

    return envs.replace(init_items_map=items)
