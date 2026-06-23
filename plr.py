"""
Robust PLR (PLR-bot) training driver for the pottery shop.

This is the regret-curriculum counterpart to `train.train_agent_multienv` (which
stays pure domain randomisation). The loop curates a `LevelSampler` buffer keyed
by *oracle regret* and, each step, takes one of two branches:

* **generate** (probability `1 - replay_prob`, and always until the buffer is
  filled past `minimum_fill_ratio`): sample a fresh batch from `gen`, roll the
  policy out on it *without a gradient update* (the PLR-bot stop-gradient), score
  it by oracle regret, and offer it to the buffer.
* **replay**: sample a high-regret batch from the buffer, take a real PPO update
  on it, and refresh those levels' regret scores.

Only replay steps move the policy -- that is what makes the buffer (a
regret-maximising adversary) the thing the student is trained against, so at
equilibrium the policy is minimax-regret (Jiang et al. 2021a). The oracle optimum
in the regret is exact (`solver.compute_optimal_return`) and cached per
level, so the expensive solve runs only on generate steps, never on replay.

The `gen` seam, network, reward function, PPO step, and oracle solver are all
reused unchanged from the DR path; only the loop differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from agent import ActorCriticNetwork
from evaluation import RewardFunction
from level_sampler import LevelSampler
from potteryshop import Environment, Item
from ppo import ppo_train_step_multienv
from rewards import DISCOUNT_RATE
from solver import compute_optimal_return
from train import default_device


def _buffer_stats(sampler: LevelSampler) -> dict[str, float]:
    """Buffer-composition diagnostics (the 'is it learning to build walls?' view)."""
    n = sampler.size
    if n == 0:
        return {"buffer/size": 0}
    items = sampler.levels.init_items_map[:n]
    urns = (items == Item.URN).flatten(1).sum(1).float()
    scores = sampler.scores[:n]
    return {
        "buffer/size": float(n),
        "buffer/mean_score": scores.mean().item(),
        "buffer/max_score": scores.max().item(),
        "buffer/mean_urns": urns.mean().item(),
        "buffer/max_urns": urns.max().item(),
    }


@dataclass
class PLRConfig:
    """All arguments for `train_agent_plr`, with sensible defaults.

    `gen`, `net`, and `reward_fn` are required (they have no meaningful default);
    everything else is a PPO, PLR-buffer, or logging hyperparameter. The PPO
    defaults are tuned for PLR (see `train_agent_plr`'s docstring). Build one and
    hand it to `train_agent_plr(config)`.
    """

    gen: Callable[..., Environment]
    net: ActorCriticNetwork
    reward_fn: RewardFunction
    num_train_steps: int = 4096
    num_envs: int = 256
    num_env_steps: int = 64
    num_epochs: int = 1
    minibatch_size: int = 512
    discount_rate: float = DISCOUNT_RATE
    entropy_coeff: float = 0.01
    lr: float = 0.003
    device: torch.device | str | None = None
    seed: int = 1
    progress: bool = True
    # --- PLR knobs ---
    buffer_capacity: int = 4096
    replay_prob: float = 0.5
    staleness_coeff: float = 0.1
    temperature: float = 0.1
    minimum_fill_ratio: float = 0.5
    duplicate_check: bool = True
    log_every: int = 1
    eval_fn: Callable[[ActorCriticNetwork, int], dict[str, float]] | None = None
    eval_every: int = 0
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def train_agent_plr(
    config: PLRConfig,
) -> tuple[ActorCriticNetwork, list[dict[str, float]], LevelSampler]:
    """
    Train `net` in place with robust PLR. Returns `(net, history, sampler)`; the
    sampler is returned so its buffer can be inspected/visualised afterwards.

    `gen` has the same `(num_envs=, generator=)` signature as for the DR driver,
    so the existing fixed-bin generator is reused as the base distribution.

    The PPO defaults are tuned for PLR: with `num_envs=256`, `num_env_steps=64`
    (collection size N=16384), `num_epochs=1` and `minibatch_size=512` give the
    ~32 gradient updates per collection that the DR HPO found to be the learning
    sweet spot (fewer undertrains, far more drifts off-policy). `replay_prob=0.5`
    balances policy updates against keeping the buffer fed with fresh high-regret
    levels -- raising it trains faster on random levels but starves (and lowers
    the quality of) the buffer, which is the curriculum PLR exists to build. If
    you change `num_envs`, rescale `minibatch_size` to keep N/minibatch ≈ 32.

    Regret is logged every step (cheap: the oracle optimum is solved on generate
    steps and cached, the achieved return comes free from the rollout), split into
    `regret/generate` (fresh levels, the DR-equivalent signal) and `regret/replay`
    (buffer levels, which should run high). PPO `log_every`-th steps record a
    metric dict tagged with the true `step` index and the branch taken.
    """
    gen = config.gen
    net = config.net
    reward_fn = config.reward_fn
    num_train_steps = config.num_train_steps
    num_envs = config.num_envs
    num_env_steps = config.num_env_steps
    num_epochs = config.num_epochs
    minibatch_size = config.minibatch_size
    discount_rate = config.discount_rate
    entropy_coeff = config.entropy_coeff
    lr = config.lr
    device = config.device
    seed = config.seed
    progress = config.progress
    buffer_capacity = config.buffer_capacity
    replay_prob = config.replay_prob
    staleness_coeff = config.staleness_coeff
    temperature = config.temperature
    minimum_fill_ratio = config.minimum_fill_ratio
    duplicate_check = config.duplicate_check
    log_every = config.log_every
    eval_fn = config.eval_fn
    eval_every = config.eval_every
    wandb_project = config.wandb_project
    wandb_run_name = config.wandb_run_name

    device = torch.device(device) if device is not None else default_device()
    net = net.to(device)
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)

    pholder = gen(num_envs=1, generator=generator)[0]
    sampler = LevelSampler(
        pholder_level=pholder,
        capacity=buffer_capacity,
        replay_prob=replay_prob,
        staleness_coeff=staleness_coeff,
        temperature=temperature,
        minimum_fill_ratio=minimum_fill_ratio,
        duplicate_check=duplicate_check,
        seed=seed,
    )

    wandb_run = None
    if wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algo": "plr",
                "num_train_steps": num_train_steps,
                "num_envs": num_envs,
                "num_env_steps": num_env_steps,
                "num_epochs": num_epochs,
                "minibatch_size": minibatch_size,
                "discount_rate": discount_rate,
                "entropy_coeff": entropy_coeff,
                "lr": lr,
                "seed": seed,
                "buffer_capacity": buffer_capacity,
                "replay_prob": replay_prob,
                "staleness_coeff": staleness_coeff,
                "temperature": temperature,
                "minimum_fill_ratio": minimum_fill_ratio,
                "device": str(device),
            },
        )

    def _ppo(envs: Environment, update: bool):
        return ppo_train_step_multienv(
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
            update=update,
        )

    history: list[dict[str, float]] = []
    num_dr = 0
    num_replay = 0
    steps = tqdm(range(num_train_steps)) if progress else range(num_train_steps)
    try:
        for step in steps:
            replay = sampler.sample_replay_decision()
            if not replay:
                # --- generate: score only, no gradient update (stop-gradient) ---
                envs = gen(num_envs=num_envs, generator=generator).to(device)
                metrics, returns = _ppo(envs, update=False)
                optimal = compute_optimal_return(
                    envs, discount_rate=discount_rate, horizon=num_env_steps
                )
                regret = optimal - returns.cpu()
                sampler.insert_batch(envs, regret, optimal)
                metrics["regret/generate"] = regret.mean().item()
                num_dr += 1
            else:
                # --- replay: real PPO update on high-regret buffer levels ---
                idx, envs = sampler.sample_replay_levels(num_envs)
                envs = envs.to(device)
                metrics, returns = _ppo(envs, update=True)
                optimal = sampler.get_optimal(idx)
                regret = optimal - returns.cpu()
                sampler.update_batch(idx, regret)
                metrics["regret/replay"] = regret.mean().item()
                num_replay += 1

            if step % log_every == 0:
                metrics["step"] = step
                metrics["branch"] = float(replay)
                metrics["num_dr_updates"] = num_dr
                metrics["num_replay_updates"] = num_replay
                metrics.update(_buffer_stats(sampler))
                if eval_fn is not None and eval_every > 0 and step % eval_every == 0:
                    metrics.update(eval_fn(net, step))
                history.append(metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                if progress:
                    steps.set_postfix(
                        {
                            "branch": "replay" if replay else "gen",
                            "regret": regret.mean().item(),
                            "buf": sampler.size,
                            "urns": metrics.get("buffer/mean_urns", 0.0),
                        }
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return net, history, sampler
