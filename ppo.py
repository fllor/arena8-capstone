"""
Reinforcement learning with (simplified) proximal policy optimisation and
generalised advantage estimation in the pottery shop environment, in batched
PyTorch.

Each train step collects a batch of rollouts with the current policy,
estimates advantages with GAE, and performs a single clipped-surrogate
gradient update (no minibatch epochs). This extraction keeps only the
multi-environment training step (`ppo_train_step_multienv`). (PyTorch port of
Matthew Farrugia-Roberts' JAX original,
https://github.com/matomatical/reward-lab.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from agent import ActorCriticNetwork
from evaluation import RewardFunction, compute_return
from potteryshop import (
    AnnotatedTransition,
    Environment,
    collect_annotated_rollout,
    tree_map,
)
from rewards import DISCOUNT_RATE


def ppo_train_step_multienv(
    net: ActorCriticNetwork,
    envs: Environment,  # a batch of environments, one per rollout
    reward_fn: RewardFunction,
    optimiser: torch.optim.Optimizer,
    num_env_steps: int = 64,
    discount_rate: float = DISCOUNT_RATE,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    minibatch_size: int = 4096,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """
    One PPO training step across a batch of environments: collect one rollout
    in each environment, then update `net` in place with `num_epochs` passes of
    minibatch SGD over the collected experience. Each pass shuffles the
    `N = num_envs * num_env_steps` transitions and splits them into minibatches
    of `minibatch_size` (so the number of gradient updates grows with the
    batch -- a fixed minibatch *count* would instead give ever-larger, ever-
    fewer updates as `num_envs` grows, and the policy barely moves). Returns
    training metrics (losses/diagnostics averaged over all minibatch updates).
    """
    assert envs.num_envs is not None, (
        "got a single environment; add a batch dimension to the environment "
        "fields (one layout per parallel rollout)"
    )
    # collect experience with current policy...
    rollouts = collect_annotated_rollout(
        env=envs,
        policy_value_fn=net.policy_value,
        num_steps=num_env_steps,
        num_rollouts=None,  # one rollout per environment in the batch
        generator=generator,
    )
    # compute rewards (flatten the batch and time dimensions, apply the
    # reward function to all B*T transitions at once, then reshape)
    B, T = rollouts.transitions.action.shape
    flat_transitions = tree_map(
        lambda x: x.flatten(start_dim=0, end_dim=1),
        rollouts.transitions,
    )
    with torch.no_grad():
        rewards = reward_fn(
            flat_transitions.state,
            flat_transitions.action,
            flat_transitions.next_state,
        ).view(B, T)
    # estimate advantages on the collected experience...
    advantages = generalised_advantage_estimation(
        rewards=rewards,
        values=rollouts.transitions.value_pred,
        final_values=rollouts.final_value_pred,
        eligibility_rate=eligibility_rate,
        discount_rate=discount_rate,
    )
    # update the policy with several epochs of minibatch SGD over the collected
    # experience. `flat_transitions` already carries the collection-time
    # ("old") action logits and value predictions, which stay fixed across
    # epochs while `net` is updated -- the clipped surrogate keeps each update
    # proximal to that fixed reference policy.
    flat_advantages = advantages.flatten()
    N = flat_advantages.shape[0]
    mb_size = max(1, min(minibatch_size, N))
    device = flat_advantages.device

    loss_sum = 0.0
    aux_sum: dict[str, float] = {}
    num_updates = 0
    for _epoch in range(num_epochs):
        # shuffle once per epoch; build the permutation on the generator's own
        # device (so randperm and the generator agree), then move it onto the
        # data device -- this keeps a CPU generator able to drive a CUDA update
        gen_device = generator.device if generator is not None else device
        perm = torch.randperm(N, generator=generator, device=gen_device)
        if perm.device != device:
            perm = perm.to(device)
        for start in range(0, N, mb_size):
            idx = perm[start : start + mb_size]
            mb_transitions = tree_map(lambda x: x[idx], flat_transitions)
            mb_advantages = flat_advantages[idx]
            loss, aux = ppo_loss_fn(
                net=net,
                transitions=mb_transitions,
                advantages=mb_advantages,
                proximity_eps=proximity_eps,
                critic_coeff=critic_coeff,
                entropy_coeff=entropy_coeff,
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimiser.step()
            loss_sum += loss.item()
            for k, v in aux.items():
                aux_sum[k] = aux_sum.get(k, 0.0) + v
            num_updates += 1

    # metrics (losses/diagnostics averaged over all minibatch updates; return
    # is measured once on the freshly collected experience)
    train_metrics = {
        "loss": loss_sum / num_updates,
        "return": compute_return(rewards, discount_rate).mean().item(),
        **{k: v / num_updates for k, v in aux_sum.items()},
    }
    return train_metrics


# # #
# PPO loss function


def ppo_loss_fn(
    net: ActorCriticNetwork,
    transitions: AnnotatedTransition,  # single leading dim (minibatch)
    advantages: Float[Tensor, "minibatch"],
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    # `transitions`/`advantages` already have a single (flat) batch dimension;
    # the caller flattens the (B, num_steps) rollout and slices minibatches.
    batch_size = advantages.shape[0]
    batch = torch.arange(batch_size, device=advantages.device)

    # run network to get latest predictions
    new_action_logits, new_value_preds = net.policy_value(transitions.obs)
    # -> float[batch_size, num_actions], float[batch_size]

    # actor loss
    new_action_logprobs = F.log_softmax(new_action_logits, dim=1)
    new_chosen_logprobs = new_action_logprobs[batch, transitions.action]
    old_action_logprobs = F.log_softmax(transitions.action_logits, dim=1)
    old_chosen_logprobs = old_action_logprobs[batch, transitions.action]
    action_log_ratios = new_chosen_logprobs - old_chosen_logprobs
    action_prob_ratios = torch.exp(action_log_ratios)
    action_prob_ratios_clipped = torch.clamp(
        action_prob_ratios,
        1 - proximity_eps,
        1 + proximity_eps,
    )
    std_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = -torch.minimum(
        std_advantages * action_prob_ratios,
        std_advantages * action_prob_ratios_clipped,
    ).mean()

    # critic loss
    value_diffs = new_value_preds - transitions.value_pred
    value_diffs_clipped = torch.clamp(
        value_diffs,
        -proximity_eps,
        proximity_eps,
    )
    new_value_preds_proximal = transitions.value_pred + value_diffs_clipped
    targets = transitions.value_pred + advantages
    critic_loss = (
        torch.maximum(
            torch.square(new_value_preds - targets),
            torch.square(new_value_preds_proximal - targets),
        ).mean()
        / 2
    )

    # entropy regularisation term
    per_step_entropy = -torch.sum(
        torch.exp(new_action_logprobs) * new_action_logprobs,
        dim=1,
    )
    average_entropy = per_step_entropy.mean()

    # diagnostics
    with torch.no_grad():
        actor_clipfrac = (action_prob_ratios_clipped != action_prob_ratios).float().mean()
        actor_approxkl1 = (-action_log_ratios).mean()
        actor_approxkl3 = ((action_prob_ratios - 1) - action_log_ratios).mean()
        critic_clipfrac = (value_diffs != value_diffs_clipped).float().mean()

    # total loss
    total_loss = (
        actor_loss + critic_coeff * critic_loss - entropy_coeff * average_entropy
    )
    return (
        total_loss,
        {
            "loss-actor": actor_loss.item(),
            "loss-critic": critic_loss.item(),
            "entropy": average_entropy.item(),
            "actor-clip": actor_clipfrac.item(),
            "critic-clip": critic_clipfrac.item(),
            "actor-kl1": actor_approxkl1.item(),
            "actor-kl3": actor_approxkl3.item(),
        },
    )


# # #
# Generalised advantage estimation


def generalised_advantage_estimation(
    rewards: Float[Tensor, "B num_steps"],
    values: Float[Tensor, "B num_steps"],
    final_values: Float[Tensor, "B"],
    eligibility_rate: float,
    discount_rate: float,
) -> Float[Tensor, "B num_steps"]:
    """
    Compute GAE advantages for a batch of rollouts with a reverse scan
    through the time axis.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    next_values = final_values
    for t in reversed(range(T)):
        gae = (
            rewards[:, t]
            - values[:, t]
            + discount_rate * (next_values + eligibility_rate * gae)
        )
        advantages[:, t] = gae
        next_values = values[:, t]
    return advantages
