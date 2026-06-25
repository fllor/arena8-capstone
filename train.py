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
  so the (purely diagnostic) oracle solve is subsampled to `dr_diag_frac` of the
  batch every `dr_diag_every` steps instead of run in full every step.

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
from editor import edit_levels
from evaluation import RewardFunction, compute_return
from level_sampler import LevelSampler
from potteryshop import Environment, Item, collect_rollout, tree_map
from ppo import ppo_train_step_multienv
from rewards import DISCOUNT_RATE, reward_break
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
        return {}
    n = sampler.size
    items = sampler.levels.init_items_map[:n]
    urns = (items == Item.URN).flatten(1).sum(1).float()
    scores = sampler.scores[:n]
    return {
        "buffer/mean_score": scores.mean().item(),
        "buffer/max_score": scores.max().item(),
        # eviction floor + fill: with the floor near 0 and the buffer full, capacity
        # is *not* binding (it holds low-regret junk a bigger buffer would too).
        "buffer/min_score": sampler.min_score,
        "buffer/size": float(n),
        "buffer/fill_ratio": n / sampler.capacity,
        "buffer/mean_urns": urns.mean().item(),
        "buffer/max_urns": urns.max().item(),
    }


def _admission_stats(sampler: LevelSampler) -> dict[str, float]:
    """Admission diagnostics from the most recent insert (insert steps only).

    `admit_rate` is the fraction of freshly-offered levels the buffer kept. If it
    decays to ~0 while the buffer is full, incoming levels can no longer displace
    incumbents -- the buffer has saturated and capacity is the binding constraint.
    """
    offered = sampler.last_offered
    if offered == 0:
        return {}
    return {
        "buffer/admit_count": float(sampler.last_admitted),
        "buffer/admit_rate": sampler.last_admitted / offered,
        "buffer/dup_rate": sampler.last_dup / offered,
    }


def _run_state(
    step: int,
    replay: bool,
    num_generate: int,
    num_replay: int,
    grad_rate: float,
    rollout_rate: float,
) -> dict[str, float]:
    """Per-step run bookkeeping (which step/branch, cumulative branch counts).

    `grad_step`/`rollout_step` are common x-axes for fair DR-vs-PLR plots: the
    *expected* number of gradient updates / fresh-level rollouts done by `step`
    (= step * the respective per-step rate). DR does both every step (rates 1, 1
    -> both equal `step`); PLR-bot updates only on replay steps and generates
    only on the rest (rates replay_prob, 1-replay_prob). Plot any metric against
    `grad_step` to compare at equal gradient budget, `rollout_step` at equal
    fresh-sample budget.
    """
    return {
        "step": step,
        "grad_step": step * grad_rate,
        "rollout_step": step * rollout_rate,
        "branch": float(replay),
        "num_generate_steps": num_generate,
        "num_replay_updates": num_replay,
    }


def _progress_postfix(
    metrics: dict[str, float], sampler: LevelSampler | None, replay: bool
) -> dict[str, object]:
    """tqdm postfix: return + branch/buffer state (PLR) or diagnostic regret (DR)."""
    postfix: dict[str, object] = {"return": metrics.get("ppo/return", 0.0)}
    if sampler is not None:
        postfix["cycle"] = "replay" if replay else "generate"
        postfix["buf/urns"] = metrics.get("buffer/mean_urns", 0.0)
    else:
        postfix["regret"] = metrics.get("regret/generate", 0.0)
    return postfix


# Rollout modes the held-out eval reports side by side. Greedy is the headline
# (reproducible, no sampling noise, and what the behavioural break-rate probe
# assumes); stochastic mirrors the on-policy training distribution.
_EVAL_MODES = (("greedy", True), ("stochastic", False))


@torch.no_grad()
def compute_eval_metrics(
    net: ActorCriticNetwork,
    eval_sets: dict[str, Environment],
    *,
    reward_fn: RewardFunction,
    discount_rate: float,
    horizon: int,
    device: torch.device | str,
    optima: dict[str, torch.Tensor] | None = None,
    solved_eps: float = 0.05,
    stochastic_seed: int = 1234,
) -> dict[str, float]:
    """Single source of truth for held-out evaluation metrics.

    For every named population in `eval_sets` (e.g. ``"random"`` /  ``"walls"``),
    rolls `net` out under BOTH a greedy and a stochastic policy and reports the
    full metric set against the exact oracle optimum:

    * ``eval/{set}/optimal``                  -- mean oracle return (mode-independent)
    * ``eval/{set}/{mode}/regret``            -- mean regret (optimal - achieved, >=0)
    * ``eval/{set}/{mode}/max_regret``        -- worst-case regret over the set
    * ``eval/{set}/{mode}/solved``            -- fraction with regret < `solved_eps`
    * ``eval/{set}/{mode}/break``             -- fraction that smash >=1 urn (the
                                                 behavioural GMG probe)
    * ``eval/{set}/{mode}/achieved``          -- mean achieved return

    where ``{mode}`` is ``greedy`` or ``stochastic``. `optima` may be passed in to
    reuse a precomputed oracle solve (the training loop solves each set once up
    front); otherwise it is computed here.
    """
    device = torch.device(device)
    net = net.to(device)
    if optima is None:
        optima = {
            name: compute_optimal_return(
                envs, discount_rate=discount_rate, horizon=horizon
            )
            for name, envs in eval_sets.items()
        }

    out: dict[str, float] = {}
    for name, envs in eval_sets.items():
        optimal = optima[name]
        out[f"eval/{name}/optimal"] = optimal.mean().item()
        B = envs.num_envs
        envs_dev = envs.to(device)
        for mode, deterministic in _EVAL_MODES:
            roll = collect_rollout(
                env=envs_dev,
                policy_fn=net.policy,
                num_steps=horizon,
                generator=torch.Generator().manual_seed(stochastic_seed),
                device=device,
                deterministic=deterministic,
            )
            flat = tree_map(lambda x: x.flatten(0, 1), roll.transitions)
            rew = reward_fn(flat.state, flat.action, flat.next_state).view(B, horizon)
            achieved = compute_return(rew, discount_rate=discount_rate).cpu()
            regret = (optimal - achieved).clamp_min(0)
            breaks = reward_break(flat.state, flat.action, flat.next_state).view(B, horizon)
            broke = (breaks.sum(dim=1) > 0).float()
            p = f"eval/{name}/{mode}"
            out[f"{p}/regret"] = regret.mean().item()
            out[f"{p}/max_regret"] = regret.max().item()
            out[f"{p}/solved"] = (regret < solved_eps).float().mean().item()
            out[f"{p}/break"] = broke.mean().item()
            out[f"{p}/achieved"] = achieved.mean().item()
    return out


def _eval_summary_line(step: int, eval_metrics: dict[str, float], names) -> str:
    """One-line headless digest of `compute_eval_metrics` (greedy regret + break)."""
    body = " | ".join(
        f"{name} regret {eval_metrics.get(f'eval/{name}/stochastic/regret', float('nan')):+.3f} "
        f"break {eval_metrics.get(f'eval/{name}/stochastic/break', float('nan')):.2f}"
        for name in names
    )
    return f"[eval step {step}] {body}"


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
      only (subsampled per `dr_diag_frac`/`dr_diag_every`).
    """

    gen: Callable[..., Environment]             # function to generate new environments
    net: ActorCriticNetwork                     # agent
    reward_fn: RewardFunction                   # reward function
    num_train_steps: int = 4096                 # number of training cycles
    num_envs: int = 256                         # batch size
    num_env_steps: int = 64                     # episode length
    num_epochs: int = 1                         # how many PPO training epochs
    num_minibatches : int = 32                  # number of minibatches
    minibatch_size: int = 0                     # minibatch size (populated automatically)
    discount_rate: float = DISCOUNT_RATE        # reward discount factor
    entropy_coeff: float = 0.01                 # coefficient of entropy loss term
    lr: float = 0.003                           # learning rate
    device: torch.device | str | None = None    # compute device
    seed: int = 1                               # RNG seed
    # --- curriculum mode knobs ---
    train_on_generate: bool = False     # do gradient update on generate cycle
    # --- ACCEL editor ---
    edit_prob: float = 0.0              # prob. a replay step is followed by an edit (>0 => ACCEL)
    num_edits: int = 3                  # elementary edits applied per level when editing
    edit_mode: str = "toggle"           # edit distribution: "toggle" (count-changing) or "walk" (urn-walk, count-conserving)
    # --- PLR buffer  ---
    buffer_active: bool | None = None   # use buffer (populated automatically)
    buffer_capacity: int = 4096         # maximum buffer capacity
    replay_prob: float = 0.5            # probability of replay cycle
    staleness_coeff: float = 0.1        # weight of staleness term
    temperature: float = 0.1            # temperature for sampling levels
    minimum_fill_ratio: float = 0.5     # fill buffer before allowing replay cycles
    duplicate_check: bool = True        # check for duplicates in buffer
    buffer_load_path: str | None = None # warm-start the buffer from a saved snapshot (refit to buffer_capacity)
    buffer_save_path: str | None = None # save the buffer alongside net checkpoints + at end
    # --- DR diagnostic oracle solve (only used when the buffer is inactive) ---
    dr_diag_every: int = 10             # how often to compute regret
    dr_diag_frac: float = 0.1           # fraction of environments to compute regret for
    # --- eval ---
    eval_sets: dict[str, Environment] | None = None # Held-out eval populations
    eval_every: int = 10                # how often to compute eval metrics
    eval_solved_eps: float = 0.05       # regret tolerance for determining solved status
    # --- checkpoint ---
    checkpoint_path: str | None = None  # path to save checkpoint during training
    checkpoint_every: int = 0           # how often to save checkpoint
    # --- weights and biases logging ---
    wandb_project: str | None = None    # project name
    wandb_run_name: str | None = None   # run name

    def __post_init__(self) -> None:
        # The buffer is engaged exactly when there is something to replay or edit.
        # Derive it from replay_prob/edit_prob unless the caller pinned it.
        if self.buffer_active is None:
            self.buffer_active = self.replay_prob > 0 or self.edit_prob > 0

        assert (
            (self.minibatch_size == 0 and self.num_minibatches > 0) or
            (self.minibatch_size > 0 and self.num_minibatches == 0)
        )
        if self.minibatch_size == 0:
            self.minibatch_size = self.num_envs * self.num_env_steps // self.num_minibatches
        elif self.num_minibatches == 0:
            self.num_minibatches = self.num_envs * self.num_env_steps // self.minibatch_size


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

    A history row is recorded every step (recording is cheap); the row carries
    whatever metrics that step produced. Regret logging: when the buffer is active
    every generate step scores the full batch (needed for admission) and logs
    `regret/generate`, while replay steps log `regret/replay` from cached optima --
    both free, so logged every step. In pure DR the oracle solve is diagnostic and
    skippable, so it is subsampled to `dr_diag_frac` of the batch every
    `dr_diag_every` steps and logged under the same `regret/generate` key (other
    rows simply omit it), so DR and PLR generate-regret plot together.
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
        if config.buffer_load_path is not None:
            sampler.load(config.buffer_load_path)
            print(
                f"loaded buffer from {config.buffer_load_path}: "
                f"{sampler.size}/{sampler.capacity} levels, "
                f"score [{sampler.min_score:.3f}, {sampler.scores[:sampler.size].max():.3f}]",
                flush=True,
            )
    # DR diagnostic-solve subsample size (unused when the buffer is active).
    dr_diag_num_envs = max(1, int(num_envs * config.dr_diag_frac))

    # Per-step rates for the `grad_step`/`rollout_step` fair-comparison axes (see
    # _run_state). A gradient update happens on every replay step and on generate
    # steps iff `train_on_generate`; a fresh-level rollout happens on every
    # generate step. DR (replay_prob=0, train_on_generate=True) -> both rates 1.
    grad_rate = config.replay_prob + (1 - config.replay_prob) * float(train_on_generate)
    rollout_rate = 1 - config.replay_prob

    # Held-out eval: solve each population's oracle optimum once up front (it is
    # policy-independent), then re-roll the policy against it every eval_every.
    eval_enabled = config.eval_sets is not None and config.eval_every > 0
    eval_optima: dict[str, torch.Tensor] | None = None
    if eval_enabled:
        eval_optima = {
            name: compute_optimal_return(
                envs, discount_rate=discount_rate, horizon=num_env_steps
            )
            for name, envs in config.eval_sets.items()
        }

    wandb_run = None
    if config.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "algo": (
                    "accel" if config.edit_prob > 0
                    else "plr" if buffer_active else "dr"
                ),
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
                "edit_prob": config.edit_prob,
                "num_edits": config.num_edits,
                "edit_mode": config.edit_mode,
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
    postfix = dict()
    num_generate = 0
    num_replay = 0
    steps = tqdm(range(config.num_train_steps))
    try:
        for step in steps:
            replay = buffer_active and sampler.sample_replay_decision()
            # DR only: the diagnostic oracle solve is skippable, so throttle it.
            # (PLR always solves every generate step for buffer admission.)
            solve_diag = not buffer_active and step % config.dr_diag_every == 0

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

                # --- ACCEL edit sub-step (Parker-Holder et al. 2022, Alg. 1) ---
                # Mutate the just-replayed parents, score the children by oracle
                # regret under the *current* policy (stop-gradient -- the student
                # only ever trains on replay), and offer them to the buffer. The
                # editor builds the rare urn-walls plain PLR can only wait to
                # sample. Reuses the generate branch's admission path verbatim.
                if config.edit_prob > 0 and (
                    torch.rand((), generator=generator, device=generator.device)
                    < config.edit_prob
                ):
                    children = edit_levels(
                        envs, num_edits=config.num_edits, generator=generator,
                        edit_mode=config.edit_mode,
                    )
                    postfix["cycle"] = "edit"
                    steps.set_postfix(postfix)
                    _, c_returns = _ppo(children, update=False)
                    c_optimal = compute_optimal_return(
                        children, discount_rate=discount_rate, horizon=num_env_steps
                    )
                    c_regret = c_optimal - c_returns.cpu()
                    sampler.insert_batch(children, c_regret, c_optimal)
                    metrics.update(_admission_stats(sampler))
                    metrics["regret/edit"] = c_regret.mean().item()
                    metrics["edit/mean_urns"] = (
                        (children.init_items_map == Item.URN)
                        .flatten(1).sum(1).float().mean().item()
                    )
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
                    metrics.update(_admission_stats(sampler))
                    metrics["regret/generate"] = regret.mean().item()
                elif solve_diag:
                    # pure DR: oracle solve is diagnostic only -> subsample it.
                    k = dr_diag_num_envs
                    optimal = compute_optimal_return(
                        envs[:k], discount_rate=discount_rate, horizon=num_env_steps
                    )
                    regret = optimal - returns[:k].cpu()
                    # same quantity as PLR's generate-branch regret (fresh-level
                    # optimal - achieved), just subsampled -> share the key so
                    # DR and PLR plot together.
                    metrics["regret/generate"] = regret.mean().item()
                num_generate += 1

            # Record every step (recording is cheap); the row carries whatever
            # metrics this step produced -- e.g. DR rows off `dr_diag_every` omit
            # `regret/generate`, PLR rows omit the branch they didn't take. The branch
            # above sets the PPO + regret keys; the helpers add run/buffer/eval.
            metrics.update(
                _run_state(step, replay, num_generate, num_replay, grad_rate, rollout_rate)
            )
            metrics.update(_buffer_stats(sampler))
            if eval_enabled and step % config.eval_every == 0:
                metrics.update(compute_eval_metrics(
                    net, config.eval_sets,
                    reward_fn=reward_fn, discount_rate=discount_rate,
                    horizon=num_env_steps, device=device, optima=eval_optima,
                    solved_eps=config.eval_solved_eps,
                ))
                line = _eval_summary_line(step, metrics, config.eval_sets)
                tqdm.write(line)
            history.append(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            postfix = _progress_postfix(metrics, sampler, replay)
            steps.set_postfix(postfix)
            if (
                config.checkpoint_path
                and config.checkpoint_every
                and step > 0
                and step % config.checkpoint_every == 0
            ):
                torch.save(net.state_dict(), config.checkpoint_path)
                if config.buffer_save_path and sampler is not None:
                    sampler.save(config.buffer_save_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if config.buffer_save_path and sampler is not None:
        sampler.save(config.buffer_save_path)

    return net, history, sampler
