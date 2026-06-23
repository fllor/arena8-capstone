"""
Prioritised Level Replay (PLR-perp / robust PLR) for the pottery shop.

Implements the curator half of Dual Curriculum Design (Jiang et al. 2021,
`doc/2110.02439.txt`, Algorithm 1) with the oracle-latest regret estimator from
Abdel Sadek, Farrugia-Roberts et al. (RLC 2025, `doc/2507.03068.txt`, eq. 4):

    regret(theta) = max_pi' V(pi', theta)  -  V_latest(pi, theta)
                    \___ solver.py oracle      \___ policy rollout return

The student is PPO (`ppo_train_step_multienv`); the adversary is buffer curation
with a random generator (no learned adversary). Each training step is either:

  * EXPLORE (replay-decision d=0): sample a *fresh* batch from `gen`, score every
    level by oracle regret, admit the high-regret ones to the buffer -- and take
    NO gradient step (the "perp"/stop-gradient of PLR-perp: avoid training on
    randomly-sampled levels so the equilibrium policy is minimax-regret; see
    2110.02439 Cor. 1 / lines 327-331).
  * REPLAY (d=1): sample a batch from the buffer by regret priority, run one PPO
    update on it, then re-score those levels and refresh their staleness.

This is the curriculum drop-in for `train_agent_multienv` (in `train.py`, the DR
baseline, left untouched). ACCEL (Day 3) extends this by editing buffer levels
before re-scoring -- the buffer + scoring machinery here is what it builds on.
"""

from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from agent import ActorCriticNetwork
from evaluation import RewardFunction, compute_return
from potteryshop import Environment, collect_rollout, tree_map
from ppo import ppo_train_step_multienv
from rewards import DISCOUNT_RATE
from solver import compute_optimal_return_grouped
from train import default_device


class PLRBuffer:
    """
    A fixed-capacity store of levels keyed by oracle regret, with rank-based
    prioritised sampling (Jiang et al. 2021, Sec. 3).

    Stores each level's layout (robot/items/bin), its last regret score, and the
    step it was last scored (for staleness). Levels live on CPU to keep GPU
    memory for the rollout buffers; `sample` returns a CPU `Environment` batch
    that the caller moves to the training device.

    Sampling probability mixes a rank-based regret term with a staleness term
    (Jiang et al. Sec. 3, eq. for `P_replay`):

        P = (1 - rho) * P_regret + rho * P_staleness
        P_regret(level) prop. (1 / rank)^(1 / beta)   (rank 1 = highest regret)

    `rho` (staleness coeff) keeps stale scores from going unrefreshed; lower
    `beta` makes the regret term peakier (more aggressively favours the top).
    """

    def __init__(
        self,
        capacity: int,
        world_size: int,
        staleness_coeff: float = 0.1,
        prioritisation_beta: float = 0.3,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.rho = staleness_coeff
        self.beta = prioritisation_beta
        ws = world_size
        self.robot = torch.zeros((capacity, 2), dtype=torch.long)
        self.items = torch.zeros((capacity, ws, ws), dtype=torch.long)
        self.bin = torch.zeros((capacity, 2), dtype=torch.long)
        self.scores = torch.zeros(capacity, dtype=torch.float32)
        self.last_seen = torch.zeros(capacity, dtype=torch.long)
        self.size = 0
        # dedicated CPU generator: multinomial over a CPU probability vector
        # needs a CPU generator, independent of the (possibly CUDA) train rng.
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    @property
    def mean_score(self) -> float:
        return self.scores[: self.size].mean().item() if self.size else 0.0

    def _envs_at(self, idx: torch.Tensor) -> Environment:
        return Environment(
            init_robot_pos=self.robot[idx].clone(),
            init_items_map=self.items[idx].clone(),
            bin_pos=self.bin[idx].clone(),
        )

    def insert(self, envs: Environment, scores: torch.Tensor, step: int) -> None:
        """
        Admit fresh levels, keeping the top-`capacity` by regret. New candidates
        are concatenated with the current contents and the highest-scoring
        `capacity` are retained (so a high-regret newcomer evicts the lowest-
        regret incumbent; if there is room, everything is kept).
        """
        r = envs.init_robot_pos.detach().cpu()
        it = envs.init_items_map.detach().cpu()
        b = envs.bin_pos.detach().cpu()
        s = scores.detach().cpu().float()
        n = s.shape[0]
        last = torch.full((n,), step, dtype=torch.long)

        cat_r = torch.cat([self.robot[: self.size], r])
        cat_it = torch.cat([self.items[: self.size], it])
        cat_b = torch.cat([self.bin[: self.size], b])
        cat_s = torch.cat([self.scores[: self.size], s])
        cat_last = torch.cat([self.last_seen[: self.size], last])

        total = self.size + n
        if total <= self.capacity:
            sel = torch.arange(total)
        else:
            sel = torch.topk(cat_s, self.capacity).indices
        self.size = min(total, self.capacity)
        self.robot[: self.size] = cat_r[sel]
        self.items[: self.size] = cat_it[sel]
        self.bin[: self.size] = cat_b[sel]
        self.scores[: self.size] = cat_s[sel]
        self.last_seen[: self.size] = cat_last[sel]

    def sample(self, num: int, step: int) -> tuple[Environment, torch.Tensor]:
        """
        Draw `num` levels (with replacement) by the mixed regret/staleness
        priority. Returns the CPU `Environment` batch and the buffer indices it
        came from (so the caller can re-score those slots after a PPO update).
        """
        assert self.size > 0, "cannot sample from an empty buffer"
        scores = self.scores[: self.size]

        # rank-based regret priority: rank 1 = highest regret.
        order = torch.argsort(scores, descending=True)
        ranks = torch.empty(self.size, dtype=torch.float32)
        ranks[order] = torch.arange(1, self.size + 1, dtype=torch.float32)
        w_regret = (1.0 / ranks) ** (1.0 / self.beta)
        p_regret = w_regret / w_regret.sum()

        # staleness priority: favour levels not scored recently.
        age = (step - self.last_seen[: self.size]).float()
        p_stale = age / age.sum() if age.sum() > 0 else torch.full_like(age, 1.0 / self.size)

        p = (1.0 - self.rho) * p_regret + self.rho * p_stale
        idx = torch.multinomial(p, num, replacement=True, generator=self._gen)
        return self._envs_at(idx), idx

    def update(self, idx: torch.Tensor, scores: torch.Tensor, step: int) -> None:
        """Refresh regret scores and staleness for replayed levels."""
        s = scores.detach().cpu().float()
        # de-duplicate (sampling is with replacement): keep the last write per
        # slot, which is well-defined since duplicates carry the same level.
        self.scores[idx] = s
        self.last_seen[idx] = step


def train_agent_plr(
    gen: Callable[..., Environment],
    net: ActorCriticNetwork,
    reward_fn: RewardFunction,
    num_train_steps: int = 300,
    num_envs: int = 4096,
    num_env_steps: int = 64,
    num_epochs: int = 1,
    minibatch_size: int = 16384,
    discount_rate: float = DISCOUNT_RATE,
    entropy_coeff: float = 0.01,
    lr: float = 0.003,
    device: torch.device | str | None = None,
    seed: int = 1,
    progress: bool = True,
    replay_prob: float = 0.5,
    buffer_capacity: int | None = None,
    staleness_coeff: float = 0.1,
    prioritisation_beta: float = 0.3,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> tuple[ActorCriticNetwork, list[dict[str, float]], PLRBuffer]:
    """
    Train `net` with PLR-perp: a regret-keyed level buffer feeds PPO replay
    updates while fresh random levels are used only to curate the buffer.

    Same `gen` signature as `train_agent_multienv` (`gen(num_envs=, generator=)`),
    so it is a drop-in swap for the DR baseline. Returns the trained network, the
    per-replay-step metric history (same keys as the DR trainer plus `regret`,
    `buffer_size`, `buffer_mean_score`, so `run.py`'s plotting still works), and
    the final buffer (for visualising what the curriculum learned to keep).

    `replay_prob` is the Bernoulli p of Algorithm 1: the fraction of steps that
    train on the buffer (the rest explore + curate). Until the buffer has any
    content, every step is forced to explore.

    Cost note: unlike the DR trainer (which oracle-scores only a 10% slice every
    10th step), PLR oracle-scores *every* batch -- explore steps to decide
    admission, replay steps to refresh priorities. The per-level solver groups by
    density so this stays bounded, but keep `num_envs` modest (a few thousand).
    """
    device = torch.device(device) if device is not None else default_device()
    net = net.to(device)
    if buffer_capacity is None:
        buffer_capacity = num_envs

    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    decision_gen = torch.Generator().manual_seed(seed + 1)  # CPU: replay coin flips
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)

    buffer = PLRBuffer(
        capacity=buffer_capacity,
        world_size=gen(num_envs=1, generator=torch.Generator().manual_seed(0)).world_size,
        staleness_coeff=staleness_coeff,
        prioritisation_beta=prioritisation_beta,
        seed=seed,
    )

    wandb_run = None
    if wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algo": "PLR-perp",
                "num_train_steps": num_train_steps,
                "num_envs": num_envs,
                "num_env_steps": num_env_steps,
                "num_epochs": num_epochs,
                "minibatch_size": minibatch_size,
                "discount_rate": discount_rate,
                "entropy_coeff": entropy_coeff,
                "lr": lr,
                "seed": seed,
                "replay_prob": replay_prob,
                "buffer_capacity": buffer_capacity,
                "staleness_coeff": staleness_coeff,
                "prioritisation_beta": prioritisation_beta,
                "device": str(device),
            },
        )

    def _achieved_returns(rollout) -> torch.Tensor:
        B, T = rollout.transitions.action.shape
        flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
        with torch.no_grad():
            rewards = reward_fn(flat.state, flat.action, flat.next_state).view(B, T)
        return compute_return(rewards, discount_rate)

    history: list[dict[str, float]] = []
    steps = tqdm(range(num_train_steps)) if progress else range(num_train_steps)
    try:
        for step in steps:
            coin = torch.rand(1, generator=decision_gen).item()
            replay = len(buffer) > 0 and coin < replay_prob

            if not replay:
                # EXPLORE: fresh levels, score + curate, NO gradient step.
                envs = gen(num_envs=num_envs, generator=generator).to(device)
                with torch.no_grad():
                    rollout = collect_rollout(
                        env=envs, policy_fn=net.policy, num_steps=num_env_steps,
                        generator=generator, device=device, deterministic=False,
                    )
                achieved = _achieved_returns(rollout).cpu()
                optimal = compute_optimal_return_grouped(
                    envs, discount_rate=discount_rate, horizon=num_env_steps
                )
                regret = (optimal - achieved).clamp_min(0)
                buffer.insert(envs, regret, step)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "explore_regret": regret.mean().item(),
                            "buffer_size": len(buffer),
                            "buffer_mean_score": buffer.mean_score,
                        },
                        step=step,
                    )
                if progress:
                    steps.set_postfix(
                        {"mode": "explore", "buf": len(buffer),
                         "buf_score": round(buffer.mean_score, 3)}
                    )
                continue

            # REPLAY: buffer levels, one PPO update, then re-score + refresh.
            envs, idx = buffer.sample(num_envs, step)
            envs = envs.to(device)
            metrics, returns_per_env = ppo_train_step_multienv(
                net=net, envs=envs, reward_fn=reward_fn, optimiser=optimiser,
                num_env_steps=num_env_steps, num_epochs=num_epochs,
                minibatch_size=minibatch_size, discount_rate=discount_rate,
                eligibility_rate=0.95, proximity_eps=0.1, critic_coeff=0.5,
                entropy_coeff=entropy_coeff, max_grad_norm=0.5, generator=generator,
            )
            optimal = compute_optimal_return_grouped(
                envs, discount_rate=discount_rate, horizon=num_env_steps
            )
            regret = (optimal - returns_per_env.cpu()).clamp_min(0)
            buffer.update(idx, regret, step)

            metrics["regret"] = regret.mean().item()
            metrics["step"] = step
            metrics["buffer_size"] = len(buffer)
            metrics["buffer_mean_score"] = buffer.mean_score
            history.append(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if progress:
                steps.set_postfix(
                    {"mode": "replay", "return": round(metrics["return"], 3),
                     "regret": round(metrics["regret"], 3), "buf": len(buffer)}
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return net, history, buffer
