"""
Evaluating agent behaviour with reward functions, in batched PyTorch.

A `RewardFunction` maps a *batch* of transitions to a batch of scalar
rewards: it takes a batched `State` (fields with leading dimension B), a
batched action (int[B]), and a batched next state, and returns float[B].
(PyTorch port of Matthew Farrugia-Roberts' JAX original,
https://github.com/matomatical/reward-lab.)
"""

from __future__ import annotations

from typing import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from agent import ActorCriticNetwork
from potteryshop import Environment, State, collect_rollout, tree_map
from rewards import DISCOUNT_RATE

RewardFunction = Callable[
    [State, Int[Tensor, "B"], State],
    Float[Tensor, "B"],
]


def compute_return(
    rewards: Float[Tensor, "... num_steps"],
    discount_rate: float,
) -> Float[Tensor, "..."]:
    """
    The discounted cumulative sum of rewards from the start of the
    trajectory, $\\sum_t \\gamma^t r_t$ (computed along the final axis).
    """
    num_steps = rewards.shape[-1]
    discounting = discount_rate ** torch.arange(
        num_steps,
        dtype=rewards.dtype,
        device=rewards.device,
    )
    return (rewards * discounting).sum(dim=-1)


@torch.no_grad()
def evaluate_behaviour(
    env: Environment,
    net: ActorCriticNetwork,
    reward_fn: RewardFunction,
    num_steps: int = 64,
    num_rollouts: int = 1000,
    discount_rate: float = DISCOUNT_RATE,
    generator: torch.Generator | None = None,
    deterministic: bool = True,
) -> Float[Tensor, "num_rollouts"]:
    """
    Sample `num_rollouts` trajectories from the policy and score each one
    with `reward_fn`, returning the per-trajectory discounted returns.

    The environment is moved onto the network's device automatically, so a CPU
    layout can be evaluated against a network trained on the GPU.

    By default actions are taken greedily (`deterministic=True`): this measures
    the policy's actual learned behaviour without exploration noise, and matches
    the deterministic oracle optimum used for regret. Note that greedy rollouts
    from the same layout are identical, so `num_rollouts` can be set to 1 unless
    the layouts themselves differ. Pass `deterministic=False` to recover the
    old stochastic-sampling behaviour.
    """
    device = next(net.parameters()).device
    rollouts = collect_rollout(
        env=env,
        policy_fn=net.policy,
        num_steps=num_steps,
        num_rollouts=num_rollouts,
        generator=generator,
        device=device,
        deterministic=deterministic,
    )
    # apply the reward function to all B*T transitions at once
    transitions = tree_map(
        lambda x: x.flatten(start_dim=0, end_dim=1),
        rollouts.transitions,
    )
    rewards = reward_fn(
        transitions.state,
        transitions.action,
        transitions.next_state,
    ).view(num_rollouts, num_steps)
    return compute_return(rewards, discount_rate)
