"""
ACCEL edit operator for the pottery shop.

ACCEL (Adversarially Compounding Complexity by Editing Levels; Parker-Holder et
al. 2022, arXiv:2203.01302) extends PLR-bot with a level *editor*: it mutates the
just-replayed high-regret levels, re-scores the children by oracle regret, and
offers them back to the buffer. Over generations this *builds* the rare urn-walls
that random generation almost never produces -- crossing the "rarity threshold"
where plain PLR stalls (the project's `CLAUDE.md` story).

`edit_levels` is the elementary edit distribution. Following the paper's
"sequence of n elementary level modifications", each edit picks one random cell
and sets it to a different item. This is the fully *unrestricted* edit
distribution from the GMG paper's Appendix J (arXiv:2507.03068): an edit can
freely turn a sparse level into a distinguishing urn-wall. We pair it with the
exact oracle regret estimator (`solver.compute_optimal_return`), the paper's
strongest, most robust configuration ("oracle-latest").

Two pottery-shop specifics make this simpler than the jaxued maze editor:

* **No solvability filter.** Urns are *passable* (stepping smashes them to
  shards), so the grid graph is always fully connected and the oracle always
  returns a finite optimum -- every edit stays solvable by construction. We only
  forbid placing an item on the bin or robot-start cell.
* **No urn cap.** The oracle's `3^(#urns)` DP factor is the only bad scaler, but
  on the small grid urn/shard counts are hard-bounded by the cells, and the
  solver already groups by exact `(#shards, #urns)` and chunks accordingly, so a
  dense edited level is merely slower, never an OOM.
"""

from __future__ import annotations

import torch

from potteryshop import Environment, Item

_NUM_ITEMS = len(Item)  # EMPTY, SHARDS, URN


def edit_levels(
    envs: Environment,
    *,
    num_edits: int,
    generator: torch.Generator,
) -> Environment:
    """
    Return a batch of edited copies of `envs` (the ACCEL mutation operator).

    For each level, applies `num_edits` elementary edits in sequence. Each edit
    picks one uniformly-random cell -- any cell except the bin and the robot
    spawn -- and sets it to a uniformly-random *different* item in
    {EMPTY, SHARDS, URN}, so every edit is a real change. The robot and bin
    positions are left untouched.

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

    # Eligible = every cell except the bin and robot-start cells. Robot/bin are
    # separate Environment fields (never on the map), but an item must not land
    # on them, so mask their flattened indices out.
    robot_idx = (envs.init_robot_pos[:, 0] * ws + envs.init_robot_pos[:, 1]).long()
    bin_idx = (envs.bin_pos[:, 0] * ws + envs.bin_pos[:, 1]).long()
    eligible = torch.ones(B, cells, dtype=torch.bool, device=device)
    eligible.scatter_(1, robot_idx[:, None], False)
    eligible.scatter_(1, bin_idx[:, None], False)

    for _ in range(num_edits):
        # Pick one eligible cell per level via noise-argmax (a multinomial-free,
        # device-robust uniform draw over the eligible cells).
        noise = torch.rand(B, cells, generator=generator, device=gdev).to(device)
        cell = noise.masked_fill(~eligible, -1.0).argmax(dim=1)  # [B]

        cur = flat.gather(1, cell[:, None]).squeeze(1)  # [B], current item
        # New item != current: add a random offset in {1, ..., n-1} (mod n).
        offset = torch.randint(
            1, _NUM_ITEMS, (B,), generator=generator, device=gdev
        ).to(device)
        new = (cur + offset) % _NUM_ITEMS
        flat.scatter_(1, cell[:, None], new[:, None])

    return envs.replace(init_items_map=items)
