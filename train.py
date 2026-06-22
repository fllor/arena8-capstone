"""
Train one agent across many environments at once, on GPU.

Each training step samples a fresh batch of random layouts from `gen`, moves
them onto the training device, and performs one multi-environment PPO update.
The notebook's live plotting is replaced with plain headless metric logging so
this runs cleanly as a script on a GPU box.

Run the bundled demo (fixed-bin generator, intended reward) with:

    python train.py
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from tqdm import tqdm

from agent import ActorCriticNetwork
from evaluation import RewardFunction
from generate import generate
from potteryshop import Action, Environment
from ppo import ppo_train_step_multienv
from rewards import DISCOUNT_RATE, reward2


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_agent_multienv(
    gen: Callable[..., Environment],
    net: ActorCriticNetwork,
    reward_fn: RewardFunction,
    num_train_steps: int = 4096,
    num_envs: int = 32,
    num_env_steps: int = 64,
    discount_rate: float = DISCOUNT_RATE,
    entropy_coeff: float = 0.01,  # needs more exploration than single-env
    lr: float = 0.001,
    device: torch.device | str | None = None,
    seed: int = 1,
    log_every: int = 128,
    progress: bool = True,
) -> tuple[ActorCriticNetwork, list[dict[str, float]]]:
    """
    Train `net` in place, resampling `num_envs` random layouts from `gen` every
    step. Returns the (in-place updated) network and the list of per-step
    metric dicts.

    `gen` must accept `num_envs=...` and `generator=...` keyword arguments and
    return a batch of that many environments (e.g. `generate` partially applied
    with `world_size`/`num_shards`/`num_urns`).
    """
    device = torch.device(device) if device is not None else default_device()
    net = net.to(device)

    # sampling generator stays on CPU so layout/action draws are reproducible
    # regardless of the training device
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    steps = tqdm(range(num_train_steps)) if progress else range(num_train_steps)
    for step in steps:
        envs = gen(num_envs=num_envs, generator=generator).to(device)
        metrics = ppo_train_step_multienv(
            net=net,
            envs=envs,
            reward_fn=reward_fn,
            optimiser=optimiser,
            num_env_steps=num_env_steps,
            discount_rate=discount_rate,
            eligibility_rate=0.95,
            proximity_eps=0.1,
            critic_coeff=0.5,
            entropy_coeff=entropy_coeff,
            max_grad_norm=0.5,
            generator=generator,
        )
        history.append(metrics)
        if log_every and (step + 1) % log_every == 0:
            print(
                f"step {step + 1:>5}/{num_train_steps}"
                f"  return={metrics['return']:+.3f}"
                f"  loss={metrics['loss']:+.3f}"
                f"  entropy={metrics['entropy']:.3f}"
            )

    return net, history


if __name__ == "__main__":
    # Demo: train on the fixed-bin distribution with the intended reward.
    # (Mirrors the notebook's `net3` configuration.)
    world_size = 4
    device = default_device()
    print(f"training on {device}")

    net = ActorCriticNetwork.init(
        obs_height=world_size,
        obs_width=world_size,
        net_channels=16,
        net_width=64,
        num_conv_layers=5,
        num_dense_layers=2,
        num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )

    net, history = train_agent_multienv(
        gen=functools.partial(
            generate,
            world_size=world_size,
            num_shards=4,
            num_urns=2,
        ),
        net=net,
        reward_fn=reward2,
        num_train_steps=4096,
        device=device,
        seed=1,
    )

    final = sum(m["return"] for m in history[-32:]) / min(32, len(history))
    print(f"done. mean return over last 32 steps: {final:+.3f}")
