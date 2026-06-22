"""
Procedural generation of pottery shop environments, in batched PyTorch.

`generate` produces a batch of random layouts with the bin pinned to the
top-left corner (0, 0) -- the "fixed bin location" distribution. The robot and
items are placed at distinct random cells.
"""

from __future__ import annotations

import torch

from potteryshop import Environment, Item


def generate(
    world_size: int,
    num_shards: int,
    num_urns: int,
    num_envs: int,
    generator: torch.Generator | None = None,
) -> Environment:
    """
    Sample `num_envs` random layouts with the bin fixed at (0, 0).

    The robot, `num_shards` shards, and `num_urns` urns are placed at distinct
    cells sampled without replacement from the remaining cells (cell 0 is the
    bin). Returned tensors live on the generator's device (CPU by default);
    move the batch to the training device with `.to(device)`.
    """
    num_cells = world_size**2

    # place the bin in the top left corner of the world
    bin_pos = torch.zeros((num_envs, 2), dtype=torch.long)

    # sample robot and item positions without replacement from the remaining
    # cells (cell 0 is the bin), by taking the first few cells of a random
    # permutation
    num_positions = 1 + num_shards + num_urns
    perm = torch.rand(num_envs, num_cells - 1, generator=generator).argsort(dim=1) + 1
    positions = perm[:, :num_positions]
    rows, cols = positions // world_size, positions % world_size
    robot_pos = torch.stack((rows[:, 0], cols[:, 0]), dim=-1)

    # create item map
    items_map = torch.zeros((num_envs, world_size, world_size), dtype=torch.long)
    batch = torch.arange(num_envs)[:, None]
    items_map[
        batch,
        rows[:, 1 : 1 + num_shards],
        cols[:, 1 : 1 + num_shards],
    ] = Item.SHARDS
    items_map[
        batch,
        rows[:, 1 + num_shards :],
        cols[:, 1 + num_shards :],
    ] = Item.URN

    return Environment(
        init_robot_pos=robot_pos,
        init_items_map=items_map,
        bin_pos=bin_pos,
    )
