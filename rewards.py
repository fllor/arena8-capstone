"""
Reward functions for the pottery shop environment, in batched PyTorch.

The "intended" training reward is `reward2`, the sum of:

* `reward_shaped` -- a potential-based shaping reward that rewards depositing
  shards in the bin, with a discounted shaping term for picking shards up; and
* `reward_no_break` -- a penalty for smashing urns.

`reward_break` (a probe for urn-smashing) is included because
`reward_no_break` is defined in terms of it.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

from potteryshop import Action, Item, State

# Discount rate, shared by training, GAE, and the shaping term below.
DISCOUNT_RATE = 0.995


def reward_break(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """Probe: +1 on any transition that turns an urn into shards (a smash)."""
    batch = torch.arange(state.inventory.shape[0], device=state.inventory.device)
    item_below_robot_after_transition = next_state.items_map[
        batch,
        next_state.robot_pos[:, 0],
        next_state.robot_pos[:, 1],
    ]
    item_there_before_transition = state.items_map[
        batch,
        next_state.robot_pos[:, 0],
        next_state.robot_pos[:, 1],
    ]
    return (
        (item_below_robot_after_transition == Item.SHARDS)
        & (item_there_before_transition == Item.URN)
    ).float()


def reward_no_break(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """Penalty of -2 per urn smashed."""
    return -2.0 * reward_break(state, action, next_state)


def inventory_potential(state: State) -> Float[Tensor, "B"]:
    """Shaping potential: 1 while the robot is holding shards, else 0."""
    return (state.inventory == Item.SHARDS).float()


def reward_bin(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """+1 for a PUTDOWN of shards while standing on the bin."""
    return (
        (state.bin_pos[:, 0] == state.robot_pos[:, 0])
        & (state.bin_pos[:, 1] == state.robot_pos[:, 1])
        & (state.inventory == Item.SHARDS)
        & (action == Action.PUTDOWN)
    ).float()


def reward_shaped(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """Bin reward plus a (potential-based, discounted) pickup shaping term."""
    pickup_shaping_term = DISCOUNT_RATE * inventory_potential(next_state) - inventory_potential(state)
    bin_reward_term = reward_bin(state, action, next_state)
    return bin_reward_term + pickup_shaping_term / 2


def reward2(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """The intended training reward: shaped binning reward minus urn-break penalty."""
    shaped = reward_shaped(state, action, next_state)
    nobreak = reward_no_break(state, action, next_state)
    return shaped + nobreak
