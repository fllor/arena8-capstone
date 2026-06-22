"""
Procedural generation of pottery shop environments, in batched PyTorch.

`generate` produces a batch of random layouts with the bin pinned to the
top-left corner (0, 0) -- the "fixed bin location" distribution. The robot and
items are placed at distinct random cells.

The *number* of shards and urns is randomised per environment from truncated
geometric distributions (one mean each), rather than fixed. A geometric is
monotonically decreasing, so most layouts stay sparse (the common case the agent
should treat as "walk around urns"), while every count keeps a non-zero
probability up to what the grid can physically hold -- including the dense
urn-wall layouts that matter at deployment. Both counts are floored at 1: every
level has a shard to deliver and at least one urn to navigate around. There is
no artificial upper cap; the only bound is placeability, and the falloff is
steep enough that dense layouts are vanishingly rare (e.g. 11 urns has
probability ~1e-7 at the default mean), so they essentially never appear even in
an 8k-env batch yet remain possible.
"""

from __future__ import annotations

import torch

from potteryshop import Environment, Item


def _sample_counts(
    mean: float,
    num_envs: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """
    Sample `num_envs` counts as ``1 + Geometric`` with the given `mean` (>= 1).

    The geometric is sampled by inverse-CDF, so it has **unbounded** support: any
    count has non-zero probability and there is no cap. Geometric
    ``P(k) prop. q**k`` (k = 0, 1, ...) has mean ``q / (1 - q)``, so a count mean
    of `mean` (floored at 1) uses ``q = (mean - 1) / mean``; ``k = floor(log U /
    log q)`` for ``U ~ Uniform(0, 1)`` inverts the CDF. Returned on the
    generator's device (CPU by default).
    """
    if mean == 1.0:  # degenerate geometric -> always exactly 1
        device = generator.device if generator is not None else torch.device("cpu")
        return torch.ones(num_envs, dtype=torch.long, device=device)
    ratio = (mean - 1.0) / mean
    u = torch.rand(num_envs, generator=generator, device=_gen_device(generator))
    u = u.clamp_min(torch.finfo(u.dtype).tiny)  # avoid log(0) -> -inf
    k = torch.floor(u.log() / torch.tensor(ratio).log())
    return 1 + k.long()


def _gen_device(generator: torch.Generator | None) -> torch.device:
    return generator.device if generator is not None else torch.device("cpu")


def generate(
    world_size: int,
    shard_mean: float,
    urn_mean: float,
    num_envs: int,
    generator: torch.Generator | None = None,
) -> Environment:
    """
    Sample `num_envs` random layouts with the bin fixed at (0, 0).

    Per environment, the shard and urn counts are drawn as ``1 + Geometric`` with
    means `shard_mean` and `urn_mean` respectively (both therefore >= 1). The
    geometric is uncapped; a draw is only ever clamped down if it would not
    physically fit on the grid (shards + urns <= capacity), which the steep
    falloff makes vanishingly rare. The robot and items are then placed at
    distinct cells sampled without replacement from the non-bin cells (cell 0 is
    the bin).

    Both means must be >= 1 (each is the mean of a count floored at 1). Returned
    tensors live on the generator's device (CPU by default), so passing a CUDA
    generator builds layouts on-GPU directly; otherwise move the batch to the
    training device with `.to(device)`.
    """
    assert shard_mean >= 1.0, f"shard_mean must be >= 1 (got {shard_mean})"
    assert urn_mean >= 1.0, f"urn_mean must be >= 1 (got {urn_mean})"

    num_cells = world_size**2
    # cells available for shards + urns after reserving the bin (cell 0) and one
    # cell for the robot.
    capacity = num_cells - 2
    assert capacity >= 2, (
        f"{world_size}x{world_size} grid has no room for a robot, shard and urn"
    )
    device = generator.device if generator is not None else torch.device("cpu")

    # --- sample per-env counts from (unbounded) geometric distributions -------
    # Each count is floored at 1 but otherwise uncapped: the geometric has full
    # support, so any number of urns/shards has non-zero probability. Sampled in a
    # single shot. The ONLY bound is placeability -- a layout needs
    # shards + urns <= capacity to fit on the grid -- so we truncate a draw down
    # to the physical maximum if (and only if) it would not fit. With the steep
    # falloff this clamp fires with negligible probability, so the tiny mass it
    # piles at the boundary does not matter; nothing is capped below the grid
    # limit.
    num_shards = _sample_counts(shard_mean, num_envs, generator)
    num_urns = _sample_counts(urn_mean, num_envs, generator)
    num_shards = num_shards.clamp(max=capacity - 1)  # leave room for >= 1 urn
    num_urns = torch.minimum(num_urns, capacity - num_shards)

    num_shards = num_shards.to(device)
    num_urns = num_urns.to(device)
    counts = num_shards + num_urns  # items per env (excludes robot)
    max_slots = int(counts.max().item())

    # place the bin in the top left corner of the world
    bin_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)

    # sample robot and item positions without replacement from the remaining
    # cells (cell 0 is the bin), by taking the first few cells of a random
    # permutation. We only need the robot (1) plus `max_slots` item cells.
    perm = (
        torch.rand(num_envs, num_cells - 1, generator=generator, device=device)
        .argsort(dim=1)
        + 1
    )
    positions = perm[:, : 1 + max_slots]
    rows, cols = positions // world_size, positions % world_size
    robot_pos = torch.stack((rows[:, 0], cols[:, 0]), dim=-1)

    # assign an item to each of the `max_slots` candidate cells per env: the
    # first `num_shards` slots are SHARDS, the next `num_urns` are URN, any
    # trailing slots stay EMPTY. Writing EMPTY (0) to those trailing cells is a
    # no-op since the map starts at zero and every cell index is distinct.
    slot_idx = torch.arange(max_slots, device=device)[None, :]  # [1, max_slots]
    is_shard = slot_idx < num_shards[:, None]
    is_urn = (slot_idx >= num_shards[:, None]) & (slot_idx < counts[:, None])
    slot_item = torch.where(
        is_shard,
        torch.full_like(slot_idx, Item.SHARDS),
        torch.where(
            is_urn,
            torch.full_like(slot_idx, Item.URN),
            torch.full_like(slot_idx, Item.EMPTY),
        ),
    )

    items_map = torch.zeros(
        (num_envs, world_size, world_size), dtype=torch.long, device=device
    )
    batch = torch.arange(num_envs, device=device)[:, None]
    items_map[batch, rows[:, 1:], cols[:, 1:]] = slot_item

    return Environment(
        init_robot_pos=robot_pos,
        init_items_map=items_map,
        bin_pos=bin_pos,
    )
