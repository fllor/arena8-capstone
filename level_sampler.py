"""
Prioritised level buffer for regret-based UED (PLR-bot / ACCEL), a PyTorch port
of the jaxued `LevelSampler` (doc/jaxued/src/jaxued/level_sampler.py).

A `LevelSampler` curates a fixed-capacity buffer of pottery-shop layouts keyed by
*regret*. The training loop drives it: each step it either replays a high-regret
buffer level (and takes a PPO update on it) or generates a fresh level (scored
only, no update -- the PLR-bot stop-gradient). High-regret levels survive; the
rest are evicted. See `plr.py` for the driver.

Differences from the jaxued reference, all deliberate:

* **Stateful, not functional.** JAX needs a pure `sampler` dict threaded through
  every call; PyTorch does not, so the buffer state lives on the object and the
  methods mutate it in place. The method names mirror the reference for
  cross-reference.
* **Scoring is external.** The reference computes MaxMC / positive-value-loss
  inside the loop; we hand the buffer an oracle regret score
  (`optimal_return - achieved_return`) from `solver.compute_optimal_return*`.
* **`optimal` replaces `levels_extra["max_return"]`.** The oracle optimum is
  policy-independent, so it is computed once when a level enters the buffer and
  cached forever -- never recomputed on replay (only the achieved-return term of
  the regret goes stale, and that is refreshed by `update_batch`).
* **Batch replay sampling computes weights once** (the reference recomputes them
  per draw as timestamps tick within the batch); with a small staleness
  coefficient this intra-batch refresh is negligible.

The buffer lives on CPU (a few hundred KB for 4096 tiny grids); the driver moves
the sampled/generated batch to the training device. Stochastic methods draw from
the sampler's own CPU `Generator` (seeded at construction) so a CUDA training
generator never has to match the buffer's device.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

from potteryshop import Environment

Prioritization = Literal["rank"]


def _empty_like_levels(pholder: Environment, capacity: int) -> Environment:
    """A batched `Environment` of `capacity` zero-filled copies of `pholder`."""
    return dataclasses.replace(
        pholder,
        **{
            f.name: torch.zeros(
                (capacity, *getattr(pholder, f.name).shape),
                dtype=getattr(pholder, f.name).dtype,
            )
            for f in dataclasses.fields(pholder)
        },
    )


class LevelSampler:
    """
    A fixed-capacity, regret-prioritised level buffer.

    Args:
        pholder_level: a single (unbatched) `Environment` used only to size and
            type the buffer's level storage.
        capacity: maximum number of levels held.
        replay_prob: probability of taking a replay (vs generate) step, once the
            buffer is filled past `minimum_fill_ratio`.
        staleness_coeff: weight of the staleness term in the replay distribution
            (`P = (1 - c)*P_score + c*P_staleness`).
        temperature: rank-prioritization temperature (`P_score ∝ (1/rank)^(1/T)`).
        minimum_fill_ratio: no replay step is taken until the buffer is at least
            this full (until then every step is generate-only).
        prioritization: only "rank" is implemented (the reference's default).
        duplicate_check: if set, re-inserting a level already in the buffer
            updates its score/timestamp in place instead of adding a duplicate.
        seed: seeds the sampler's internal CPU generator.
    """

    def __init__(
        self,
        pholder_level: Environment,
        capacity: int = 4096,
        replay_prob: float = 0.5,
        staleness_coeff: float = 0.1,
        temperature: float = 0.1,
        minimum_fill_ratio: float = 0.5,
        prioritization: Prioritization = "rank",
        duplicate_check: bool = True,
        seed: int = 0,
    ):
        assert prioritization == "rank", "only rank prioritization is implemented"
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.staleness_coeff = staleness_coeff
        self.temperature = temperature
        self.minimum_fill_ratio = minimum_fill_ratio
        self.duplicate_check = duplicate_check
        self.generator = torch.Generator().manual_seed(seed)

        # buffer state (all on CPU)
        self.levels = _empty_like_levels(pholder_level.to("cpu"), capacity)
        self.scores = torch.full((capacity,), -torch.inf, dtype=torch.float32)
        self.timestamps = torch.zeros(capacity, dtype=torch.long)
        self.optimal = torch.zeros(capacity, dtype=torch.float32)
        self.size = 0
        self.episode_count = 0

    # -- weights -------------------------------------------------------------

    def _fill_mask(self) -> torch.Tensor:
        return torch.arange(self.capacity) < self.size

    def score_weights(self) -> torch.Tensor:
        """Rank-prioritised score weights, shape [capacity], summing to 1."""
        mask = self._fill_mask()
        masked_scores = torch.where(mask, self.scores, torch.tensor(-torch.inf))
        # rank 1 = highest score; descending sort, then invert the permutation
        order = torch.argsort(masked_scores, descending=True)
        ranks = torch.empty(self.capacity, dtype=torch.long)
        ranks[order] = torch.arange(self.capacity) + 1
        w = torch.where(mask, 1.0 / ranks, torch.zeros(())) ** (1.0 / self.temperature)
        return w / w.sum()

    def staleness_weights(self) -> torch.Tensor:
        """Staleness weights (linear in episodes-since-last-seen), shape [capacity]."""
        mask = self._fill_mask()
        staleness = (self.episode_count - self.timestamps).to(torch.float32)
        w = torch.where(mask, staleness, torch.zeros(()))
        if w.sum() > 0:
            return w / w.sum()
        return mask.to(torch.float32) / max(self.size, 1)

    def level_weights(self) -> torch.Tensor:
        """Full replay distribution: blend of score and staleness weights."""
        return (
            1 - self.staleness_coeff
        ) * self.score_weights() + self.staleness_coeff * self.staleness_weights()

    # -- replay sampling -----------------------------------------------------

    def sample_replay_decision(self) -> bool:
        """True => take a replay step this cycle; False => generate a fresh batch."""
        filled = self.size / self.capacity
        if filled < self.minimum_fill_ratio:
            return False
        return bool(torch.rand((), generator=self.generator) < self.replay_prob)

    def sample_replay_levels(self, num: int) -> tuple[torch.Tensor, Environment]:
        """
        Sample `num` buffer levels (with replacement) by the replay distribution.

        Bumps `episode_count` by `num` and refreshes the sampled levels'
        timestamps so they count as freshly seen. Returns their buffer indices
        and the corresponding batched `Environment`.
        """
        weights = self.level_weights()
        idx = torch.multinomial(weights, num, replacement=True, generator=self.generator)
        self.episode_count += num
        self.timestamps[idx] = self.episode_count
        return idx, self.levels[idx]

    # -- insertion / update --------------------------------------------------

    def find(self, env: Environment) -> int:
        """Buffer index of `env` (a single level), or -1 if absent."""
        mask = self._fill_mask()
        eq = mask.clone()
        for f in dataclasses.fields(env):
            buf = getattr(self.levels, f.name).reshape(self.capacity, -1)
            qry = getattr(env.to("cpu"), f.name).reshape(-1)
            eq &= (buf == qry).all(dim=1)
        if bool(eq.any()):
            return int(eq.to(torch.int8).argmax())
        return -1

    def _duplicate_indices(self, envs: Environment) -> torch.Tensor:
        """
        For each level in `envs`, the buffer index it duplicates, or -1.

        Vectorised `[B, capacity]` equality across all level fields (masked to
        filled slots). Replaces a per-level `find()` loop -- the reference checks
        duplicates one at a time, but a single batched comparison is far cheaper
        and only differs on the rare within-batch duplicate (two identical fresh
        levels), which is immaterial here.
        """
        B = envs.num_envs
        mask = self._fill_mask()  # [capacity]
        eq = mask[None, :].expand(B, self.capacity).clone()
        for f in dataclasses.fields(envs):
            buf = getattr(self.levels, f.name).reshape(self.capacity, -1)  # [K, d]
            qry = getattr(envs, f.name).reshape(B, -1)  # [B, d]
            eq &= (buf[None] == qry[:, None]).all(dim=2)  # [B, K]
        has = eq.any(dim=1)
        return torch.where(has, eq.to(torch.int8).argmax(dim=1), torch.full((B,), -1))

    def insert_batch(
        self,
        envs: Environment,
        scores: torch.Tensor,
        optimal: torch.Tensor,
    ) -> None:
        """
        Insert a batch of freshly-scored levels, keeping the `capacity` highest
        regret levels overall.

        Vectorised replacement for the reference's sequential per-level scan: a
        level is admitted iff its score exceeds that of the weakest level it would
        displace (preserved exactly by a single top-`capacity` selection over the
        pooled incumbent + newcomer scores; empty slots hold `-inf` so they fill
        first). Duplicates already in the buffer have their score/timestamp
        refreshed in place rather than added again.

        One deliberate deviation from the reference: the eviction *victim* is the
        lowest-*score* level, not the lowest combined score+staleness *weight*.
        This keeps exactly the highest-regret set (which is the buffer's whole
        purpose); with temperature 0.1 and staleness coeff 0.1 the victim choice
        among low-score levels is a second-order effect.
        """
        envs = envs.to("cpu")
        scores = scores.cpu().float().clone()
        optimal = optimal.cpu().float()
        B = envs.num_envs
        self.episode_count += B  # every offered level counts as an episode seen

        # refresh any duplicates in place, and drop them from the insert pool
        if self.duplicate_check:
            dup = self._duplicate_indices(envs)  # [B]
            is_dup = dup >= 0
            d_idx = dup[is_dup]
            self.scores[d_idx] = scores[is_dup]
            self.timestamps[d_idx] = self.episode_count
            scores[is_dup] = -torch.inf  # exclude from the top-k pool below

        # pool incumbents + newcomers, keep the top `capacity` by score
        new_ts = torch.full((B,), self.episode_count, dtype=torch.long)
        cat_scores = torch.cat([self.scores, scores])  # [K + B]
        cat_optimal = torch.cat([self.optimal, optimal])
        cat_ts = torch.cat([self.timestamps, new_ts])
        cat_fields = {
            f.name: torch.cat(
                [getattr(self.levels, f.name), getattr(envs, f.name)], dim=0
            )
            for f in dataclasses.fields(envs)
        }

        keep = torch.topk(cat_scores, self.capacity).indices
        self.scores = cat_scores[keep]
        self.optimal = cat_optimal[keep]
        self.timestamps = cat_ts[keep]
        self.levels = dataclasses.replace(
            self.levels, **{name: t[keep] for name, t in cat_fields.items()}
        )
        self.size = int(torch.isfinite(self.scores).sum())

    def update_batch(self, idx: torch.Tensor, scores: torch.Tensor) -> None:
        """Refresh the regret scores of replayed levels (the optimum is cached)."""
        self.scores[idx.cpu()] = scores.cpu()

    # -- introspection (for logging / the buffer-composition money shot) -----

    def get_optimal(self, idx: torch.Tensor) -> torch.Tensor:
        """Cached oracle optimum for buffer indices `idx`."""
        return self.optimal[idx.cpu()]

    @property
    def num_filled(self) -> int:
        return self.size
