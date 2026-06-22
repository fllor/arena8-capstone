"""
Train one agent across many environments at once, on GPU.

Each training step samples a fresh batch of random layouts from `gen`, moves
them onto the training device, and performs one multi-environment PPO update.
The notebook's live plotting is replaced with plain headless metric logging so
this runs cleanly on a GPU box.

This module is a library of reusable functions (`train_agent_multienv`,
`default_device`); see `run.py` for an interactive driver with `# %%` cells.
"""

from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from agent import ActorCriticNetwork
from evaluation import RewardFunction
from potteryshop import Environment
from ppo import ppo_train_step_multienv
from rewards import DISCOUNT_RATE


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
    num_envs: int = 256,
    num_env_steps: int = 64,
    num_epochs: int = 4,
    minibatch_size: int = 4096,
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

    # Put the sampling generator on the training device when that device is
    # CUDA. This lets `_sample_actions` (and layout generation in `gen`) run
    # `multinomial`/`rand` directly on-GPU, eliminating the GPU<->CPU round
    # trip that otherwise happens on *every* rollout timestep and serialises
    # the GPU. Draws are still reproducible for a fixed (seed, device), but no
    # longer bit-identical across devices. CPU/MPS keep a CPU generator (MPS
    # has patchy generator support); for those, `_sample_actions` falls back to
    # its cross-device sampling path.
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device).manual_seed(seed)
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
            num_epochs=num_epochs,
            minibatch_size=minibatch_size,
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
