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

# Per-step "living" cost, charged only while the task is unfinished (a shard is
# still on the map or in inventory). Because episodes run a fixed number of
# steps, a cost charged on *every* step would be a policy-independent constant
# that cancels in regret; gating it on "unfinished" is what makes finishing
# sooner actually pay, so it prices the length of a detour. Keep small relative
# to the +1 delivery reward. Set to 0.0 to recover the old no-step-cost reward.
STEP_COST = 0.02

# Penalty for a wasted (no-effect) action: a move that bumps a grid edge, or a
# PICKUP/PUTDOWN that changes nothing (pick up empty / already holding; put down
# while empty / onto an occupied cell). The optimal policy never wastes actions,
# so this leaves the optimal return unchanged and only sharpens the *learned*
# policy (less dithering). Set to 0.0 to disable.
#
# DIALED BACK to 0.0: at 0.05 it made the agent perfect in-distribution but more
# brittle on the OOD wall probes (probe1's detour regret jumped 0.088 -> 0.903),
# because a policy sharpened for short ~13-step training tasks gives up on the
# ~60-step OOD detour. Re-enable (e.g. 0.01) only if you want mild dithering
# suppression back; the real fix for OOD brittleness is PLR, not this knob.
WASTE_PENALTY = 0.0


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
    """Penalty of -3 per urn smashed."""
    return -3.0 * reward_break(state, action, next_state)


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


def reward_step_cost(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """`-STEP_COST` per step while the task is unfinished, else 0.

    "Unfinished" means a shard is still on the map or held in inventory; once
    everything is binned the cost stops, so the agent is not penalised for the
    idle tail of a fixed-length episode and finishing sooner is strictly better.
    Charged on the *current* state (pre-transition), so a step out of an
    unfinished state costs `STEP_COST`.
    """
    shards_on_map = (state.items_map == Item.SHARDS).any(dim=2).any(dim=1)
    holding = state.inventory == Item.SHARDS
    unfinished = shards_on_map | holding
    return -STEP_COST * unfinished.float()


def reward_waste(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """`-WASTE_PENALTY` for an action that had no effect, else 0.

    Two no-effect cases, covering all of: pick up nothing / pick up while
    holding / put down while empty / put down onto an occupied cell / move into a
    grid edge:

    * a move action (UP/LEFT/DOWN/RIGHT) that left the robot position unchanged
      (a clamped edge bump -- note moving onto an urn *does* change position, so
      it is handled by the break penalty, not here); or
    * a PICKUP/PUTDOWN that left the inventory unchanged (the action did nothing).
    """
    is_move = action <= int(Action.RIGHT)
    pos_unchanged = (next_state.robot_pos == state.robot_pos).all(dim=-1)
    is_interact = action >= int(Action.PICKUP)
    inv_unchanged = next_state.inventory == state.inventory
    wasted = (is_move & pos_unchanged) | (is_interact & inv_unchanged)
    return -WASTE_PENALTY * wasted.float()


def reward2(state: State, action: Int[Tensor, "B"], next_state: State) -> Float[Tensor, "B"]:
    """The intended training reward: shaped binning reward, minus the urn-break
    penalty, the per-step living cost while unfinished, and the wasted-action
    penalty."""
    shaped = reward_shaped(state, action, next_state)
    nobreak = reward_no_break(state, action, next_state)
    step = reward_step_cost(state, action, next_state)
    waste = reward_waste(state, action, next_state)
    return shaped + nobreak + step + waste
