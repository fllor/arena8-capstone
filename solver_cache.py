"""
Memoisation for the oracle solver (`solver.compute_optimal_return`).

The optimal return is an exact function of the layout *and* the reward/discount/
horizon settings, so a layout only ever needs solving once per setting. Two
sources of repetition make caching pay off on the 4x4/5x5 grids:

* **within a batch:** a batch of fresh `generate` levels (or ACCEL edit children)
  contains duplicate layouts; we collapse them with `unique` and solve each once.
* **across steps (and runs):** the same layouts recur across training steps and
  across separate runs; a sorted `(key, value)` table answers repeats with a
  `searchsorted`, and the table can be persisted to disk and reloaded.

Steps 1+2 are the in-run cache (dedup + cross-step table). Step 3 adds optional
cross-run persistence (`cache_dir`): the table is written to / merged from disk so
later runs start warm.

Correctness key
---------------
The cache is bucketed by a *signature* = (discount, horizon, the five resolved
`reward2` knobs). Reward overrides resolve exactly as in `solver` (an explicit
kwarg wins, else the live `rewards.*` global), so changing a reward param opens a
fresh bucket rather than serving a stale return. Persistence keeps this guarantee:
each bucket is stored in its own file named by a hash of the signature, with the
full signature embedded for verification -- a cache built under one reward config
can never be served to a run using another.

Level keys
----------
Each layout is packed into a single collision-free `int64` (a perfect hash, not a
lossy one): 2 bits per cell for the item (EMPTY/SHARDS/URN) plus the robot and bin
cell indices. This is exact for `world_size <= 5` (60 bits); larger grids overflow
int64 and would need a real hash -- asserted, not silently wrong.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor

import rewards
import solver
from potteryshop import Environment, Item
from rewards import DISCOUNT_RATE

# Bit layout of a level key (see module docstring). Items occupy bits
# [0, 2*W*W); robot/bin cell indices (0..W*W-1, <= 24 for W=5) sit above them.
_ROBOT_SHIFT = 50  # 2 * 5 * 5 == 50 item bits below this
_BIN_SHIFT = 55    # + 5 bits for the robot cell


def level_keys(envs: Environment, device: torch.device) -> Tensor:
    """Collision-free int64 key per level: items (2 bits/cell) | robot | bin."""
    W = envs.world_size
    assert W <= 5, (
        f"level_keys packs into int64 only for world_size <= 5, got {W}; "
        "larger grids need a wider key / real hash"
    )
    B = envs.num_envs
    items = envs.init_items_map.reshape(B, -1).to(device).long()  # [B, W*W], 0..2
    shifts = 2 * torch.arange(items.shape[1], device=device)
    code = (items << shifts).sum(1)  # [B]
    robot = (envs.init_robot_pos[:, 0] * W + envs.init_robot_pos[:, 1]).to(device).long()
    bin_ = (envs.bin_pos[:, 0] * W + envs.bin_pos[:, 1]).to(device).long()
    return code | (robot << _ROBOT_SHIFT) | (bin_ << _BIN_SHIFT)


def _sorted_union(
    k1: Tensor, v1: Tensor, k2: Tensor, v2: Tensor
) -> tuple[Tensor, Tensor]:
    """Sorted union of two (key, value) sets, de-duplicated on key. Values for a
    duplicated key are assumed equal (same signature => same optimal return), so
    keeping either is correct."""
    mk = torch.cat([k1, k2])
    mv = torch.cat([v1, v2])
    order = mk.argsort()
    mk, mv = mk[order], mv[order]
    keep = torch.ones(mk.numel(), dtype=torch.bool, device=mk.device)
    keep[1:] = mk[1:] != mk[:-1]
    return mk[keep], mv[keep]


class SolverCache:
    """Memoiser around `solver.compute_optimal_return`.

    Call `.optimal(envs, ...)` exactly where you'd call
    `solver.compute_optimal_return(envs, ...)`; the return value is identical (a
    CPU float32 `[B]` tensor in input order). With `cache_dir` set, existing
    on-disk tables are loaded at construction and `.save()` merges the current
    tables back to disk. `.stats()` exposes hit/miss counters for logging.
    """

    def __init__(self, cache_dir: str | os.PathLike | None = None) -> None:
        # signature -> (sorted int64 keys [N], float32 vals [N])
        self._buckets: dict[tuple, tuple[Tensor, Tensor]] = {}
        self.device: torch.device | None = None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # cumulative counters (over the run)
        self.lookups = 0        # levels queried through the cache
        self.hits = 0           # served from the table (no solve)
        self.solves = 0         # distinct layouts handed to the real solver
        self.dup_collapsed = 0  # within-batch duplicate misses folded away
        self.preloaded = 0      # entries loaded from disk at construction
        if self.cache_dir is not None:
            self._load()

    # --- persistence ------------------------------------------------------------

    @staticmethod
    def _sig_hash(sig: tuple) -> str:
        """Stable short filename hash of a signature tuple."""
        s = "|".join(repr(x) for x in sig)
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def _load(self) -> None:
        """Load every on-disk bucket whose signature file is present (CPU tensors;
        migrated to the compute device on the first `.optimal` call)."""
        if not self.cache_dir.exists():
            return
        for f in sorted(self.cache_dir.glob("cache_*.pt")):
            try:
                blob = torch.load(f, map_location="cpu")
                sig = tuple(blob["signature"])
                keys, vals = blob["keys"], blob["vals"]
            except Exception:
                continue  # skip corrupt / incompatible files rather than crash
            self._buckets[sig] = (keys, vals)
            self.preloaded += int(keys.numel())

    def save(self) -> None:
        """Merge the in-memory tables into the on-disk cache (atomic per file).

        Each bucket is unioned with whatever is already on disk (so a concurrent or
        earlier run's entries are preserved) and written via a temp file + rename.
        Note: the read-merge-write is not locked, so two runs flushing the *same*
        signature concurrently can drop some of each other's entries -- harmless
        (correctness is unaffected; at worst a few layouts get re-solved later)."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for sig, (k, v) in self._buckets.items():
            path = self.cache_dir / f"cache_{self._sig_hash(sig)}.pt"
            mk, mv = k.cpu(), v.cpu()
            if path.exists():
                try:
                    blob = torch.load(path, map_location="cpu")
                    mk, mv = _sorted_union(blob["keys"], blob["vals"], mk, mv)
                except Exception:
                    pass  # unreadable existing file: overwrite with what we have
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"signature": list(sig), "keys": mk, "vals": mv}, tmp)
            os.replace(tmp, path)

    # --- core -------------------------------------------------------------------

    def _signature(self, discount_rate: float, horizon: int, overrides: dict) -> tuple:
        def resolve(name: str, default):
            v = overrides.get(name)
            return default if v is None else v

        knobs = (
            resolve("break_penalty", rewards.BREAK_PENALTY),
            resolve("per_step_cost", rewards.STEP_COST),
            resolve("shaping_coeff", rewards.SHAPING_COEFF),
            resolve("waste_penalty", rewards.WASTE_PENALTY),
            resolve("bin_reward", rewards.BIN_REWARD),
        )
        return (
            round(float(discount_rate), 12),
            int(horizon),
            *(round(float(k), 12) for k in knobs),
        )

    @torch.no_grad()
    def optimal(
        self,
        envs: Environment,
        *,
        discount_rate: float = DISCOUNT_RATE,
        horizon: int = 64,
        **kwargs,
    ) -> Tensor:
        """Cached `compute_optimal_return`. Same signature, same CPU `[B]` output."""
        B = envs.num_envs
        # Single env or empty batch: not worth keying; defer to the real solver.
        if B is None or B == 0 or envs.world_size > 5:
            return solver.compute_optimal_return(
                envs, discount_rate=discount_rate, horizon=horizon, **kwargs
            )

        if self.device is None:
            self.device = (
                envs.device if envs.device.type == "cuda" else torch.device("cpu")
            )
            # migrate any disk-preloaded buckets onto the compute device
            self._buckets = {
                sig: (k.to(self.device), v.to(self.device))
                for sig, (k, v) in self._buckets.items()
            }
        dev = self.device
        sig = self._signature(discount_rate, horizon, kwargs)
        keys = level_keys(envs, dev)  # [B] int64
        self.lookups += B

        out = torch.empty(B, dtype=torch.float32)
        ck, cv = self._buckets.get(
            sig,
            (
                torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.float32, device=dev),
            ),
        )

        # --- table lookup: which queried levels are already solved? ---
        if ck.numel():
            pos = torch.searchsorted(ck, keys)
            posc = pos.clamp(max=ck.numel() - 1)
            hit = (pos < ck.numel()) & (ck[posc] == keys)
        else:
            posc = torch.zeros(B, dtype=torch.long, device=dev)
            hit = torch.zeros(B, dtype=torch.bool, device=dev)

        hit_cpu = hit.cpu()
        n_hit = int(hit_cpu.sum())
        self.hits += n_hit
        if n_hit:
            out[hit_cpu] = cv[posc[hit]].cpu()

        # --- misses: solve each distinct layout once, then broadcast back ---
        miss = ~hit
        if bool(miss.any()):
            miss_idx = miss.nonzero(as_tuple=True)[0]   # batch positions [M]
            miss_keys = keys[miss_idx]                  # [M]
            uniq, inv = miss_keys.unique(return_inverse=True)  # uniq [U], inv [M]
            U = uniq.numel()
            # one representative batch position per distinct layout (any is fine)
            rep = torch.empty(U, dtype=torch.long, device=dev)
            rep[inv] = torch.arange(miss_idx.numel(), device=dev)
            rep_batch = miss_idx[rep]                   # [U] positions into the batch

            solved = solver.compute_optimal_return(
                envs[rep_batch], discount_rate=discount_rate, horizon=horizon, **kwargs
            )  # CPU [U]
            self.solves += U
            self.dup_collapsed += int(miss_idx.numel()) - U

            out[miss.cpu()] = solved[inv.cpu()]

            # merge the freshly solved (uniq, solved) into the sorted table. The
            # uniq keys are misses by construction, so they're new -- concat + sort.
            self._buckets[sig] = _sorted_union(
                ck, cv, uniq, solved.to(dev)
            )

        return out

    def size(self) -> int:
        """Total distinct layouts memoised across all signature buckets."""
        return sum(int(k.numel()) for k, _ in self._buckets.values())

    def stats(self) -> dict[str, float]:
        """Cumulative cache counters, ready to merge into a metrics dict."""
        return {
            "solver_cache/lookups": self.lookups,
            "solver_cache/hits": self.hits,
            "solver_cache/solves": self.solves,
            "solver_cache/dup_collapsed": self.dup_collapsed,
            "solver_cache/hit_rate": self.hits / max(1, self.lookups),
            "solver_cache/preloaded": self.preloaded,
            "solver_cache/size": self.size(),
        }
