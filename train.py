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

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch
from tqdm import tqdm

from agent import ActorCriticNetwork
from evaluation import RewardFunction
from potteryshop import Environment, Item
from ppo import ppo_train_step_multienv
from rewards import DISCOUNT_RATE
from solver import compute_optimal_return

# Peak DP state count (chunk_size * N) the oracle is allowed to materialise at
# once, where N = cells * 2 * 2^shards * 3^urns. The per-level transition tables
# are O(chunk * N), so this bounds peak solver memory regardless of how dense a
# level is. ~1e6 states keeps the GPU footprint to a few hundred MB, leaving the
# training rollout buffers room (the full-batch solve used to OOM here).
_REGRET_STATE_BUDGET = 1_000_000


def _mean_optimal_return(
    envs: Environment,
    discount_rate: float,
    horizon: int,
) -> float:
    """
    Exact mean oracle optimal return over the whole batch, without ever sizing
    the DP to the batch's densest level.

    `compute_optimal_return` pads its compact state to the *batch maximum* shard
    and urn counts, so a single dense level inflates `N = C·2·2^S·3^U` — and
    thus memory — for every level sharing its chunk. We instead group levels by
    their exact `(shards, urns)` counts: each group gets its own minimal `N`, so
    sparse groups solve in wide parallel chunks while the rare dense groups fall
    back to small chunks (one level at a time in the limit). Every level is still
    scored, so the returned mean is exact — no subsampling, no distortion.
    """
    items = envs.init_items_map.flatten(1)  # [B, cells]
    n_shard = (items == Item.SHARDS).sum(1)
    n_urn = (items == Item.URN).sum(1)
    cells = envs.world_size ** 2

    total = 0.0
    count = 0
    # one group per distinct (shards, urns) pair; within a group N is constant
    for s, u in torch.unique(torch.stack([n_shard, n_urn], dim=1), dim=0).tolist():
        idx = ((n_shard == s) & (n_urn == u)).nonzero(as_tuple=True)[0]
        n_states = cells * 2 * (1 << s) * (3 ** u)
        chunk_size = max(1, _REGRET_STATE_BUDGET // n_states)  # serial when dense
        optimal = compute_optimal_return(
            envs[idx],
            discount_rate=discount_rate,
            horizon=horizon,
            chunk_size=chunk_size,
        )
        total += optimal.sum().item()
        count += idx.numel()
    return total / count


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
    progress: bool = True,
    regret_frac: float = 0.1,
    regret_every: int = 10,
    eval_fn: Callable[[ActorCriticNetwork, int], dict[str, float]] | None = None,
    eval_every: int = 0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> tuple[ActorCriticNetwork, list[dict[str, float]]]:
    """
    Train `net` in place, resampling `num_envs` random layouts from `gen` every
    step. Returns the (in-place updated) network and a list of metric dicts.

    `gen` must accept `num_envs=...` and `generator=...` keyword arguments and
    return a batch of that many environments (e.g. `generate` partially applied
    with `world_size`/`num_shards`/`num_urns`).

    PPO trains on every step, but metrics are only computed and recorded every
    `regret_every`-th step (default 10) to keep the oracle solve cheap. The
    returned history therefore has one entry per scored step, each tagged with
    its true `"step"` index.

    `regret_frac` controls how many of each scored step's layouts the oracle
    solves to log mean regret (default 10%). The exact full-batch solve costs
    ~10x the PPO step, so only the first `regret_frac` are scored. Both terms of
    the regret -- the oracle optimum and the policy's achieved return -- are
    averaged over exactly this same `envs[:regret_num_envs]` slice, so they pair
    per level and the mean regret stays non-negative.

    If `wandb_project` is given, the scored-step metrics are also logged to
    Weights & Biases under that project (optional dependency, imported lazily;
    `wandb_run_name` names the run). Logging is off by default.
    """
    device = torch.device(device) if device is not None else default_device()
    net = net.to(device)
    regret_num_envs = max(1, int(num_envs * regret_frac))

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

    # The oracle regret solve is independent of the PPO update (both only read
    # the step's layouts) and uses little memory, but the PPO step leaves the GPU
    # ~half idle. On CUDA we run the solve on a side stream from a worker thread
    # so its kernels interleave with the PPO step's, hiding most of its cost. The
    # solve scores only the first `regret_num_envs` layouts (the full-batch solve
    # costs ~10x the PPO step) and groups them by density inside the helper so a
    # rare dense layout can't OOM. On CPU/MPS we just solve inline.
    overlap = device.type == "cuda"
    regret_stream = torch.cuda.Stream() if overlap else None
    regret_pool = ThreadPoolExecutor(max_workers=1) if overlap else None

    # optional Weights & Biases logging (lazy import: only required if enabled)
    wandb_run = None
    if wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "num_train_steps": num_train_steps,
                "num_envs": num_envs,
                "num_env_steps": num_env_steps,
                "num_epochs": num_epochs,
                "minibatch_size": minibatch_size,
                "discount_rate": discount_rate,
                "entropy_coeff": entropy_coeff,
                "lr": lr,
                "seed": seed,
                "regret_frac": regret_frac,
                "regret_every": regret_every,
                "device": str(device),
            },
        )

    def _solve_async(envs_slice, ready_event):
        with torch.cuda.stream(regret_stream):
            regret_stream.wait_event(ready_event)  # wait until the slice is ready
            return _mean_optimal_return(
                envs_slice, discount_rate=discount_rate, horizon=num_env_steps
            )

    history: list[dict[str, float]] = []
    steps = tqdm(range(num_train_steps)) if progress else range(num_train_steps)
    try:
        for step in steps:
            envs = gen(num_envs=num_envs, generator=generator).to(device)
            # PPO trains every step, but metrics (including regret) are only
            # computed and logged on every `regret_every`-th step.
            score = step % regret_every == 0
            regret_envs = envs[:regret_num_envs]

            if score and overlap:
                ready = torch.cuda.Event()
                ready.record()  # default stream: the slice is now produced
                regret_future = regret_pool.submit(_solve_async, regret_envs, ready)

            metrics, returns_per_env = ppo_train_step_multienv(
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
            if not score:
                continue

            # regret on the SAME layouts the oracle scored: optimal minus the
            # policy's achieved return over `regret_envs`. Pairing the two terms
            # per level keeps each summand >= 0 (the optimum upper-bounds any
            # rollout), so the mean can't go negative from subsample noise.
            if overlap:
                optimal_mean = regret_future.result()
            else:
                optimal_mean = _mean_optimal_return(
                    regret_envs, discount_rate=discount_rate, horizon=num_env_steps
                )
            achieved_mean = returns_per_env[:regret_num_envs].mean().item()
            metrics["regret"] = optimal_mean - achieved_mean
            metrics["step"] = step  # history is sparse; record the true step index
            # optional held-out evaluation (e.g. rare-env regret on a fixed set).
            # `eval_every` should be a multiple of `regret_every` so the eval
            # metrics land on a scored step and are logged alongside the rest.
            if eval_fn is not None and eval_every > 0 and step % eval_every == 0:
                metrics.update(eval_fn(net, step))
            history.append(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if progress:
                steps.set_postfix(
                    {
                        "return": metrics["return"],
                        "loss": metrics["loss"],
                        "regret": metrics["regret"],
                    }
                )
    finally:
        if regret_pool is not None:
            regret_pool.shutdown()
        if wandb_run is not None:
            wandb_run.finish()

    return net, history
