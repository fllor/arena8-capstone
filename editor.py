"""
ACCEL edit operator for the pottery shop.

ACCEL (Adversarially Compounding Complexity by Editing Levels; Parker-Holder et
al. 2022, arXiv:2203.01302) extends PLR-bot with a level *editor*: it mutates the
just-replayed high-regret levels, re-scores the children by oracle regret, and
offers them back to the buffer. Over generations this *builds* the rare urn-walls
that random generation almost never produces -- crossing the "rarity threshold"
where plain PLR stalls (the project's `CLAUDE.md` story).

`edit_levels` is the elementary edit distribution, selected by `edit_mode`:

* **`"toggle"`** (default, the paper's setup): each edit picks one random cell
  and sets it to a different item. This is the fully *unrestricted* edit
  distribution from the GMG paper's Appendix J (arXiv:2507.03068): an edit can
  freely turn a sparse level into a distinguishing urn-wall. It is *count-changing*
  -- nothing bounds the urn total, so under regret-keyed re-editing the urn count
  ratchets up faster than the student can learn (the "urn explosion").
* **`"walk"`** (count-conserving): each edit moves one existing URN to a random
  adjacent EMPTY cell. The urn multiset is fixed by the seed level, so the editor
  can only raise regret by improving *arrangement* (assembling scattered urns into
  walls) -- difficulty is bounded by the seed urn budget, which removes the
  explosion while keeping ACCEL's wall-building. Every edit moves a urn, so the
  regret gradient stays dense even when urns are sparse.

Both pair with the exact oracle regret estimator (`solver.compute_optimal_return`),
the paper's strongest, most robust configuration ("oracle-latest").

Two pottery-shop specifics make this simpler than the jaxued maze editor:

* **No solvability filter.** Urns are *passable* (stepping smashes them to
  shards), so the grid graph is always fully connected and the oracle always
  returns a finite optimum -- every edit stays solvable by construction. We only
  forbid placing an item on the bin or robot-start cell.
* **Urn cap (toggle only).** The oracle's `3^(#urns)` DP factor is the only bad
  scaler: under regret-keyed re-editing the count-changing `toggle` edit ratchets
  the urn total up until a single dense level OOMs the solver (a >40GB tensor).
  So `toggle` children are passed through `_cap_state`, which removes urns (then
  shards) until `2^(#shards) * 3^(#urns) < _STATE_LIMIT` (~5GB solver peak,
  mirroring the generator's cap). The bound is generous (~11 urns on 5x5), so
  ACCEL still builds full walls. `walk` conserves the urn multiset, so its
  children never exceed the already-capped seed and it skips the guard.
"""

from __future__ import annotations

import math

import torch

from potteryshop import Environment, Item

# Cap on the oracle DP state factor 2^(#shards) * 3^(#urns) (mirrors the
# generator). Keeps the exact solver's peak allocation to ~5GB.
_STATE_LIMIT = 1_000_000

_NUM_ITEMS = len(Item)  # EMPTY, SHARDS, URN


def edit_levels(
    envs: Environment,
    *,
    num_edits: int,
    generator: torch.Generator,
    edit_mode: str = "toggle",
) -> Environment:
    """
    Return a batch of edited copies of `envs` (the ACCEL mutation operator).

    For each level, applies `num_edits` elementary edits in sequence. The edit
    distribution is `edit_mode`:

    * ``"toggle"`` -- pick one uniformly-random cell (any except the bin and robot
      spawn) and set it to a uniformly-random *different* item in
      {EMPTY, SHARDS, URN}. Count-changing (urn total can grow each edit).
    * ``"walk"`` -- pick one existing URN and move it to a uniformly-random
      adjacent EMPTY cell. Count-conserving: the urn/shard/empty multiset is
      preserved, so the editor rearranges urns into walls without inflating the
      urn total. Levels with no urn, or whose chosen urn has no empty neighbour,
      are left unchanged by that edit.

    Either way the robot and bin positions are left untouched. Vectorised over the
    batch and grid-size agnostic. Randomness is drawn from `generator` (on its own
    device) and applied on the levels' device, so the caller's CUDA/CPU training
    generator works unchanged.
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

    if edit_mode == "toggle":
        _toggle_edits(flat, eligible, num_edits=num_edits, generator=generator)
        # Guard: toggle ratchets the urn count up, so keep every child within the
        # oracle DP's affordable size (walk conserves counts and needs no cap).
        _cap_state(flat)
    elif edit_mode == "walk":
        _walk_edits(flat, eligible, ws=ws, num_edits=num_edits, generator=generator)
    else:
        raise ValueError(f"unknown edit_mode {edit_mode!r} (expected 'toggle' or 'walk')")

    return envs.replace(init_items_map=items)


def _cap_state(flat, *, limit=_STATE_LIMIT):
    """Remove urns (then shards) per level, in place, until each satisfies
    ``2^(#shards) * 3^(#urns) < limit`` -- the oracle DP's affordable size.

    Mirrors the generator's cap on the post-edit item map: shards are bounded so
    at least one urn always fits, then urns (which carry the larger ``3^U``
    factor, and whose removal also offsets a large ``2^S``) are shaved by cell
    order. Only removes items, so the robot/bin cells are never touched.
    """
    is_shard = flat == Item.SHARDS
    is_urn = flat == Item.URN
    S, U = is_shard.sum(1), is_urn.sum(1)  # [B]
    s_keep = S.clamp(max=18)  # 2^18 * 3 < 1e6, so >= 1 urn always fits
    excess = s_keep * math.log(2) + U * math.log(3) - math.log(limit)
    u_keep = (U - (excess / math.log(3)).ceil().clamp(min=0).long()).clamp(min=0)
    # drop urns/shards ranked beyond the keep counts (rank = 1-indexed cell order)
    urn_rank = is_urn.long().cumsum(1)
    flat[is_urn & (urn_rank > u_keep[:, None])] = int(Item.EMPTY)
    shard_rank = is_shard.long().cumsum(1)
    flat[is_shard & (shard_rank > s_keep[:, None])] = int(Item.EMPTY)


def _toggle_edits(flat, eligible, *, num_edits, generator):
    """Count-changing edit: set a random eligible cell to a different item (in place)."""
    B, cells = flat.shape
    device, gdev = flat.device, generator.device
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


# Flat-index offsets for the 4 grid neighbours (up, down, left, right) as (drow, dcol).
_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _walk_edits(flat, eligible, *, ws, num_edits, generator):
    """Count-conserving edit: move a random URN to a random adjacent EMPTY cell (in place)."""
    B, cells = flat.shape
    device, gdev = flat.device, generator.device
    dr = torch.tensor([o[0] for o in _NEIGHBOUR_OFFSETS], device=device)  # [4]
    dc = torch.tensor([o[1] for o in _NEIGHBOUR_OFFSETS], device=device)  # [4]

    for _ in range(num_edits):
        # Pick one URN per level via noise-argmax over the urn mask.
        urn_mask = flat == Item.URN  # [B, cells]
        has_urn = urn_mask.any(dim=1)  # [B]
        noise = torch.rand(B, cells, generator=generator, device=gdev).to(device)
        src = noise.masked_fill(~urn_mask, -1.0).argmax(dim=1)  # [B]

        # Candidate destinations: the 4 neighbours of the chosen urn cell.
        src_r, src_c = src // ws, src % ws  # [B]
        nr = src_r[:, None] + dr[None, :]  # [B, 4]
        nc = src_c[:, None] + dc[None, :]
        in_bounds = (nr >= 0) & (nr < ws) & (nc >= 0) & (nc < ws)  # [B, 4]
        # Clamp out-of-bounds indices to a safe cell so gather never reads OOB;
        # in_bounds masks them out below.
        nidx = (nr.clamp(0, ws - 1) * ws + nc.clamp(0, ws - 1))  # [B, 4]
        dest_empty = flat.gather(1, nidx) == Item.EMPTY  # [B, 4]
        dest_elig = eligible.gather(1, nidx)  # [B, 4]
        valid = in_bounds & dest_empty & dest_elig  # [B, 4]
        has_dest = valid.any(dim=1)  # [B]

        # Pick one valid neighbour per level via noise-argmax.
        dnoise = torch.rand(B, 4, generator=generator, device=gdev).to(device)
        which = dnoise.masked_fill(~valid, -1.0).argmax(dim=1)  # [B]
        dest = nidx.gather(1, which[:, None]).squeeze(1)  # [B]

        # Apply src->EMPTY, dest->URN only where a real move exists; elsewhere
        # write back the current value so the row is untouched (src/dest indices
        # for no-move rows are arbitrary but harmless).
        move = (has_urn & has_dest)[:, None]  # [B, 1]
        cur_src = flat.gather(1, src[:, None])
        flat.scatter_(1, src[:, None],
                      torch.where(move, torch.full_like(cur_src, int(Item.EMPTY)), cur_src))
        cur_dest = flat.gather(1, dest[:, None])
        flat.scatter_(1, dest[:, None],
                      torch.where(move, torch.full_like(cur_dest, int(Item.URN)), cur_dest))
