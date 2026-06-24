"""
Unified UED trainer for the pottery shop: DR, PLR-bot, and ACCEL in one loop.

This module hosts a *single* training backend (`train_agent`) driven by a
*single* config (`UEDConfig`). The different methods are just different points in
the config space, not different code paths:

| method  | `replay_prob` | `train_on_generate` | buffer        |
|---------|---------------|---------------------|---------------|
| DR      | 0             | True                | inert (unused)|
| PLR     | >0            | True                | regret-keyed  |
| PLR-bot | >0            | False (stop-grad)   | regret-keyed  |
| ACCEL   | >0 (+edit)    | False               | regret-keyed  |

Each step takes one branch:

* **replay** (only when the buffer is active and `sample_replay_decision` fires):
  sample a high-regret batch from the `LevelSampler`, take a *real* PPO update on
  it, and refresh those levels' regret. Replay steps *always* update the policy.
* **generate**: sample a fresh batch from `gen`. Whether this batch updates the
  policy is `train_on_generate` -- DR trains on it (`True`), PLR-bot scores it
  only (`False`, the stop-gradient that makes the regret buffer the adversary the
  student is trained against). When the buffer is active the fresh batch is scored
  by oracle regret and offered to the buffer; in pure DR the buffer is never read,
  so the (purely diagnostic) oracle solve is subsampled to `regret_frac` of the
  batch every `regret_every` steps instead of run in full every step.

So **DR is exactly `replay_prob=0, train_on_generate=True`**: the replay decision
never fires, every step generates-and-trains, and the buffer stays empty. The
`gen` seam, network, reward function, PPO step, and oracle solver are shared by
all methods; only the per-step branch differs.

`train_agent(config)` returns `(net, history, sampler)` (the `LevelSampler`, or
`None` in pure DR). See `run.py` for the unified `# %%` driver, which selects the
method by config.
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


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _buffer_stats(sampler: LevelSampler | None) -> dict[str, float]:
    """Buffer-composition diagnostics (the 'is it learning to build walls?' view)."""
    if sampler is None or sampler.size == 0:
        return {"buffer/size": 0}
    n = sampler.size
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
class UEDConfig:
    """All arguments for `train_agent`, with sensible defaults.

    `gen`, `net`, and `reward_fn` are required (they have no meaningful default);
    everything else is a PPO, curriculum, or logging hyperparameter. The defaults
    are tuned for PLR (`num_envs=256`, `num_env_steps=64`, `num_epochs=1`,
    `minibatch_size=512` -> ~32 gradient updates per collection); a fair DR-vs-PLR
    comparison should set every shared PPO hyperparameter identically and let only
    the curriculum knobs differ.

    Curriculum knobs:

    * `replay_prob` -- probability of a replay (vs generate) step once the buffer
      is filled. `0` => pure DR.
    * `train_on_generate` -- whether the *generate* branch takes a gradient update.
      DR sets `True` (it trains on every fresh batch); PLR-bot sets `False` (the
      stop-gradient: fresh levels are scored, not trained on, so only replayed
      high-regret levels move the policy).
    * `buffer_active` -- whether the regret buffer is engaged at all. Left `None`,
      it is auto-derived from `replay_prob` in `__post_init__` (`replay_prob > 0`).
      When inactive the buffer is never built and the oracle solve is diagnostic
      only (subsampled per `regret_frac`/`regret_every`).
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
    # --- curriculum mode knobs ---
    train_on_generate: bool = False
    buffer_active: bool | None = None  # auto: replay_prob > 0 (see __post_init__)
    # --- PLR buffer knobs (ignored when the buffer is inactive) ---
    buffer_capacity: int = 4096
    replay_prob: float = 0.5
    staleness_coeff: float = 0.1
    temperature: float = 0.1
    minimum_fill_ratio: float = 0.5
    duplicate_check: bool = True
    # --- DR diagnostic oracle solve (only used when the buffer is inactive) ---
    regret_frac: float = 0.1
    regret_every: int = 10
    # --- logging / eval ---
    log_every: int = 1
    eval_fn: Callable[[ActorCriticNetwork, int], dict[str, float]] | None = None
    eval_every: int = 0
    # write `net.state_dict()` to `checkpoint_path` every `checkpoint_every` steps
    # (0 disables) so an unattended run loses at most that many steps.
    checkpoint_path: str | None = None
    checkpoint_every: int = 0
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        # The buffer is engaged exactly when there is something to replay. Derive
        # it from replay_prob unless the caller has pinned it explicitly (ACCEL
        # will also OR in edit_prob here once it lands).
        if self.buffer_active is None:
            self.buffer_active = self.replay_prob > 0


def train_agent(
    config: UEDConfig,
) -> tuple[ActorCriticNetwork, list[dict[str, float]], LevelSampler | None]:
    """
    Train `net` in place with the unified UED loop. Returns
    `(net, history, sampler)`; `sampler` is the `LevelSampler` (so its buffer can
    be inspected/visualised) or `None` in pure DR (`buffer_active=False`).

    `gen` must accept `num_envs=...` and `generator=...` and return that many
    `Environment`s. The branch taken each step is set by `replay_prob` /
    `train_on_generate` / `buffer_active` (see `UEDConfig`).

    Regret logging: when the buffer is active every generate step scores the full
    batch (needed for admission) and logs `regret/generate`, while replay steps
    log `regret/replay`; in pure DR the oracle solve is diagnostic, so it is
    subsampled to `regret_frac` of the batch every `regret_every` steps and logged
    as `regret`. History is recorded every `log_every` steps when the buffer is
    active, and on every scored (`regret_every`) step in pure DR.
    """
    gen = config.gen
    net = config.net
    reward_fn = config.reward_fn
    num_envs = config.num_envs
    num_env_steps = config.num_env_steps
    discount_rate = config.discount_rate
    buffer_active = bool(config.buffer_active)
    train_on_generate = config.train_on_generate

    device = torch.device(config.device) if config.device is not None else default_device()
    net = net.to(device)
    # Keep the sampling generator on the training device for CUDA (lets layout
    # generation and action sampling run on-GPU); CPU/MPS keep a CPU generator.
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device).manual_seed(config.seed)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)

    sampler: LevelSampler | None = None
    if buffer_active:
        pholder = gen(num_envs=1, generator=generator)[0]
        sampler = LevelSampler(
            pholder_level=pholder,
            capacity=config.buffer_capacity,
            replay_prob=config.replay_prob,
            staleness_coeff=config.staleness_coeff,
            temperature=config.temperature,
            minimum_fill_ratio=config.minimum_fill_ratio,
            duplicate_check=config.duplicate_check,
            seed=config.seed,
        )
    # DR diagnostic-solve subsample size (unused when the buffer is active).
    regret_num_envs = max(1, int(num_envs * config.regret_frac))

    wandb_run = None
    if config.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "algo": "plr" if buffer_active else "dr",
                "num_train_steps": config.num_train_steps,
                "num_envs": num_envs,
                "num_env_steps": num_env_steps,
                "num_epochs": config.num_epochs,
                "minibatch_size": config.minibatch_size,
                "discount_rate": discount_rate,
                "entropy_coeff": config.entropy_coeff,
                "lr": config.lr,
                "seed": config.seed,
                "train_on_generate": train_on_generate,
                "buffer_active": buffer_active,
                "replay_prob": config.replay_prob,
                "buffer_capacity": config.buffer_capacity,
                "staleness_coeff": config.staleness_coeff,
                "temperature": config.temperature,
                "minimum_fill_ratio": config.minimum_fill_ratio,
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
            num_epochs=config.num_epochs,
            minibatch_size=config.minibatch_size,
            discount_rate=discount_rate,
            eligibility_rate=0.95,
            proximity_eps=0.1,
            critic_coeff=0.5,
            entropy_coeff=config.entropy_coeff,
            max_grad_norm=0.5,
            generator=generator,
            update=update,
        )

    history: list[dict[str, float]] = []
    num_generate = 0
    num_replay = 0
    steps = tqdm(range(config.num_train_steps)) if config.progress else range(config.num_train_steps)
    try:
        for step in steps:
            replay = buffer_active and sampler.sample_replay_decision()
            scored = buffer_active or (step % config.regret_every == 0)

            if replay:
                # --- replay: real PPO update on high-regret buffer levels ---
                idx, envs = sampler.sample_replay_levels(num_envs)
                envs = envs.to(device)
                metrics, returns = _ppo(envs, update=True)
                optimal = sampler.get_optimal(idx)
                regret = optimal - returns.cpu()
                sampler.update_batch(idx, regret)
                metrics["regret/replay"] = regret.mean().item()
                num_replay += 1
            else:
                # --- generate: train iff train_on_generate (DR=True, PLR-bot=False) ---
                envs = gen(num_envs=num_envs, generator=generator).to(device)
                metrics, returns = _ppo(envs, update=train_on_generate)
                if buffer_active:
                    # full-batch solve: every fresh level needs a regret score to
                    # be admitted to the buffer.
                    optimal = compute_optimal_return(
                        envs, discount_rate=discount_rate, horizon=num_env_steps
                    )
                    regret = optimal - returns.cpu()
                    sampler.insert_batch(envs, regret, optimal)
                    metrics["regret/generate"] = regret.mean().item()
                elif scored:
                    # pure DR: oracle solve is diagnostic only -> subsample it.
                    k = regret_num_envs
                    optimal = compute_optimal_return(
                        envs[:k], discount_rate=discount_rate, horizon=num_env_steps
                    )
                    regret = optimal - returns[:k].cpu()
                    metrics["regret"] = regret.mean().item()
                num_generate += 1

            record = (step % config.log_every == 0) if buffer_active else scored
            if record:
                metrics["step"] = step
                metrics["branch"] = float(replay)
                metrics["num_generate_steps"] = num_generate
                metrics["num_replay_updates"] = num_replay
                metrics.update(_buffer_stats(sampler))
                if config.eval_fn is not None and config.eval_every > 0 and step % config.eval_every == 0:
                    metrics.update(config.eval_fn(net, step))
                history.append(metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                if config.progress:
                    postfix = {"return": metrics.get("return", 0.0)}
                    if buffer_active:
                        postfix["branch"] = "replay" if replay else "gen"
                        postfix["buf"] = sampler.size
                        postfix["urns"] = metrics.get("buffer/mean_urns", 0.0)
                    else:
                        postfix["regret"] = metrics.get("regret", 0.0)
                    steps.set_postfix(postfix)
            if (
                config.checkpoint_path
                and config.checkpoint_every
                and step > 0
                and step % config.checkpoint_every == 0
            ):
                torch.save(net.state_dict(), config.checkpoint_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return net, history, sampler
