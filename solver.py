"""
Oracle optimal-return solver for the pottery shop, in batched PyTorch.

Computes the *exact* optimal discounted return achievable in each layout, over
the same finite horizon the policy is evaluated on. This is the baseline for the
regret signal that drives PLR / ACCEL:

    regret = optimal_achievable_return - policy_return

Why not Dijkstra (as the early plan assumed)? The task is not a shortest path:

* movement is unconstrained (no walls -- edges clamp, urns are passable), so all
  cost lives in *discounting* and *urn penalties*, not in blocked cells;
* the inventory holds one pile at a time, so `num_shards` piles force one
  round-trip to the bin each -- a routing/scheduling problem, not a single path;
* stepping onto an urn pays `-2` AND drops a real SHARDS pile that the optimal
  policy can then deliver for `+1` (net `-1` per smashed urn after cleanup);
* a smashed urn cell is free to re-cross afterwards, so breaking a doorway only
  pays off when reused across several deliveries -- this couples the trips and is
  the whole point of the GMG story;
* with gamma ~= 0.995 the discount saving from a shorter path is tiny; breaking
  is optimal essentially only when walking *around* would overrun the horizon.

Because the grid is tiny we just solve the real finite-horizon MDP exactly by
backward value iteration over a *compact* state that drops everything
reward-irrelevant:

    state = (robot_cell, holding-a-shard?, shards-remaining bitmask, urn-states)

Each original shard cell is present/absent (2 states); each urn cell is one of
3 states (intact -> shard-pile -> cleared). So the per-level state count is

    C * 2 * 2^S * 3^U      (C = cells, S = #shards, U = #urns)

= 16 * 2 * 16 * 9 = 4608 for the default 4x4 / 4-shard / 2-urn config. Backward
induction is `horizon` vectorised `gather + max` passes across all levels and
states at once -- milliseconds on GPU.

The per-step reward mirrors `rewards.reward2` exactly (bin reward + potential
shaping on the holding indicator - break penalty); see `_REWARD2` constants. If
`reward2` changes, update those constants.

Heterogeneous batches are handled by padding to S = max #shards, U = max #urns;
levels with fewer items get inert slots (absent shard / pre-cleared urn).
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

from potteryshop import Action, Environment, Item
from rewards import DISCOUNT_RATE

# Constants mirroring `rewards.reward2` (keep in sync with rewards.py).
_BIN_REWARD = 1.0       # reward_bin: +1 per delivery
_BREAK_PENALTY = 2.0    # reward_no_break: -2 per urn smashed
_SHAPING_COEFF = 0.5    # reward_shaped divides the pickup-shaping term by 2

# (row, col) deltas per Action, matching potteryshop.step.
_DELTAS = [(-1, 0), (0, -1), (+1, 0), (0, +1), (0, 0), (0, 0)]
_MOVE_ACTIONS = (Action.UP, Action.LEFT, Action.DOWN, Action.RIGHT)


def _layout_tables(envs: Environment, device: torch.device):
    """
    Extract, per level, the compact-state scaffolding:

    * shard_slot[B, C]: slot index k for each cell that holds an original shard,
      else -1.
    * urn_slot[B, C]:  slot index k for each cell that holds an urn, else -1.
    * init_shardmask[B], init_urncode[B]: the start masks (all shards present,
      real urns intact, padded urn slots pre-cleared).
    * robot_cell[B], bin_cell[B].
    * S, U: padded slot counts (max over the batch).
    """
    B = envs.num_envs
    W = envs.world_size
    C = W * W
    items = envs.init_items_map.reshape(B, C).to(device)
    is_shard = items == Item.SHARDS
    is_urn = items == Item.URN

    S = int(is_shard.sum(1).max().item()) if B else 0
    U = int(is_urn.sum(1).max().item()) if B else 0
    S = max(S, 0)
    U = max(U, 0)

    neg = torch.full((B, C), -1, dtype=torch.long, device=device)
    shard_slot = torch.where(is_shard, is_shard.long().cumsum(1) - 1, neg)
    urn_slot = torch.where(is_urn, is_urn.long().cumsum(1) - 1, neg)

    S_b = is_shard.sum(1)  # [B]
    U_b = is_urn.sum(1)
    # all real shards present: low S_b bits set
    init_shardmask = (1 << S_b) - 1 if S > 0 else torch.zeros(B, dtype=torch.long, device=device)
    init_shardmask = init_shardmask.to(torch.long)
    # urn code: real urns intact (trit 0); padded slots [U_b, U) pre-cleared (trit 2)
    if U > 0:
        pow3 = (3 ** torch.arange(U, device=device)).long()  # [U]
        k = torch.arange(U, device=device)
        padded = (k[None, :] >= U_b[:, None]).long()  # [B, U]
        init_urncode = (padded * (2 * pow3[None, :])).sum(1)  # [B]
    else:
        init_urncode = torch.zeros(B, dtype=torch.long, device=device)

    robot = envs.init_robot_pos.to(device)
    bin_ = envs.bin_pos.to(device)
    robot_cell = (robot[:, 0] * W + robot[:, 1]).long()
    bin_cell = (bin_[:, 0] * W + bin_[:, 1]).long()

    return shard_slot, urn_slot, init_shardmask, init_urncode, robot_cell, bin_cell, S, U, C, W


@torch.no_grad()
def _build_transitions(
    shard_slot: Int[Tensor, "B C"],
    urn_slot: Int[Tensor, "B C"],
    bin_cell: Int[Tensor, "B"],
    S: int,
    U: int,
    C: int,
    W: int,
    discount: float,
    break_penalty: float,
    per_step_cost: float,
    device: torch.device,
):
    """
    Build the deterministic transition table for a (chunk of a) batch.

    Returns:
      next_idx [B, N, 6] long  -- successor compact-state index per (state, action)
      reward   [B, N, 6] float -- immediate reward2 of that transition
      M_shard, M_urn, N        -- radices and total state count
    """
    B = shard_slot.shape[0]
    M_shard = 1 << S
    M_urn = 3 ** U
    N = C * 2 * M_shard * M_urn

    # Decode every compact-state index into its components (shared across levels).
    idx = torch.arange(N, device=device)
    urncode = idx % M_urn
    rest = idx // M_urn
    shardmask = rest % M_shard
    rest = rest // M_shard
    holding = rest % 2
    robot = rest // 2
    row, col = robot // W, robot % W

    pow3 = (3 ** torch.arange(max(U, 1), device=device)).long()  # [U] (>=1 entry)

    next_idx = torch.empty((B, N, 6), dtype=torch.long, device=device)
    reward = torch.empty((B, N, 6), dtype=torch.float32, device=device)

    holding_b = holding[None, :].expand(B, N)
    shardmask_b = shardmask[None, :].expand(B, N)
    urncode_b = urncode[None, :].expand(B, N)
    cur = robot  # [N], current cell
    cur_is_bin = cur[None, :] == bin_cell[:, None]  # [B, N]
    urn_slot_cur = urn_slot[:, cur]  # [B, N]
    shard_slot_cur = shard_slot[:, cur]

    for a in range(6):
        dr, dc = _DELTAS[a]
        if a in _MOVE_ACTIONS:
            dest = (torch.clamp(row + dr, 0, W - 1) * W
                    + torch.clamp(col + dc, 0, W - 1))  # [N]
        else:  # PICKUP / PUTDOWN leave the robot in place
            dest = cur

        # --- smash on entry: stepping onto an intact urn -> shard-pile, pays -break ---
        urn_slot_dest = urn_slot[:, dest]  # [B, N]
        valid_urn_dest = urn_slot_dest >= 0
        uslot_d = urn_slot_dest.clamp(min=0)
        upow_d = pow3[uslot_d]
        trit_dest = (urncode_b // upow_d) % 3
        smash = valid_urn_dest & (trit_dest == 0)
        urncode_after = urncode_b + smash.long() * upow_d  # trit 0 -> 1
        reward_smash = -break_penalty * smash.float()

        if a in _MOVE_ACTIONS:
            new_robot = dest[None, :].expand(B, N)
            new_holding = holding_b
            new_shardmask = shardmask_b
            new_urncode = urncode_after
            shaping = (discount * new_holding.float() - holding_b.float()) * _SHAPING_COEFF
            rew = reward_smash + shaping - per_step_cost

        elif a == Action.PICKUP:
            new_robot = robot[None, :].expand(B, N)
            # shard pile present at current cell? (original shard OR smashed urn)
            valid_shard = shard_slot_cur >= 0
            sslot = shard_slot_cur.clamp(min=0)
            shard_bit = torch.bitwise_right_shift(shardmask_b, sslot) & 1
            shard_here = valid_shard & (shard_bit == 1)
            valid_urn = urn_slot_cur >= 0
            uslot = urn_slot_cur.clamp(min=0)
            upow = pow3[uslot]
            urn_here = valid_urn & ((urncode_b // upow) % 3 == 1)  # trit 1 = shard-pile
            can_pickup = (holding_b == 0) & (shard_here | urn_here)
            new_holding = torch.where(can_pickup, torch.ones_like(holding_b), holding_b)
            pick_shard = can_pickup & shard_here
            pow2 = torch.bitwise_left_shift(torch.ones_like(sslot), sslot)
            new_shardmask = shardmask_b - pick_shard.long() * pow2  # clear the set bit
            pick_urn = can_pickup & urn_here
            new_urncode = urncode_b + pick_urn.long() * upow  # trit 1 -> 2
            shaping = (discount * new_holding.float() - holding_b.float()) * _SHAPING_COEFF
            rew = shaping - per_step_cost

        else:  # Action.PUTDOWN
            new_robot = robot[None, :].expand(B, N)
            deliver = (holding_b == 1) & cur_is_bin
            new_holding = torch.where(deliver, torch.zeros_like(holding_b), holding_b)
            new_shardmask = shardmask_b
            new_urncode = urncode_b  # non-bin putdowns modelled as no-ops (never optimal)
            shaping = (discount * new_holding.float() - holding_b.float()) * _SHAPING_COEFF
            rew = _BIN_REWARD * deliver.float() + shaping - per_step_cost

        nxt = ((new_robot * 2 + new_holding) * M_shard + new_shardmask) * M_urn + new_urncode
        next_idx[:, :, a] = nxt
        reward[:, :, a] = rew

    return next_idx, reward, M_shard, M_urn, N


@torch.no_grad()
def compute_optimal_return(
    envs: Environment,
    discount_rate: float = DISCOUNT_RATE,
    horizon: int = 64,
    break_penalty: float = _BREAK_PENALTY,
    per_step_cost: float = 0.0,
    device: torch.device | str | None = None,
    chunk_size: int = 512,
) -> Float[Tensor, "B"]:
    """
    Exact optimal discounted return for each layout in `envs`, over `horizon`
    steps with discount `discount_rate`, under the `reward2` reward.

    Returns a CPU float tensor of shape [B] (one optimal return per level),
    matching the per-level layout of `evaluate_behaviour`'s policy returns so
    regret is just `compute_optimal_return(envs) - policy_returns`.
    """
    if device is None:
        device = envs.device if envs.device.type == "cuda" else torch.device("cpu")
    device = torch.device(device)

    B = envs.num_envs
    assert B is not None, "pass a *batch* of environments (leading batch dim)"

    shard_slot, urn_slot, init_shardmask, init_urncode, robot_cell, bin_cell, S, U, C, W = (
        _layout_tables(envs, device)
    )

    out = torch.empty(B, dtype=torch.float32)
    for lo in range(0, B, chunk_size):
        hi = min(lo + chunk_size, B)
        sl = slice(lo, hi)
        next_idx, reward, M_shard, M_urn, N = _build_transitions(
            shard_slot[sl], urn_slot[sl], bin_cell[sl],
            S, U, C, W, discount_rate, break_penalty, per_step_cost, device,
        )
        Bc = hi - lo
        # backward induction: V_t = max_a [ r + gamma * V_{t+1}(next) ]
        V = torch.zeros((Bc, N), dtype=torch.float32, device=device)
        flat_idx = next_idx.reshape(Bc, N * 6)
        for _ in range(horizon):
            V_next = torch.gather(V, 1, flat_idx).reshape(Bc, N, 6)
            V = (reward + discount_rate * V_next).amax(dim=2)
        start = (((robot_cell[sl] * 2 + 0) * M_shard + init_shardmask[sl]) * M_urn
                 + init_urncode[sl])  # holding=0 at start
        out[lo:hi] = V.gather(1, start[:, None]).squeeze(1).cpu()
    return out


@torch.no_grad()
def compute_regret(
    envs: Environment,
    policy_returns: Float[Tensor, "B"],
    **kwargs,
) -> Float[Tensor, "B"]:
    """
    Oracle regret per level: optimal_return - policy_return. `policy_returns`
    typically comes from `evaluation.evaluate_behaviour(envs, net, reward2)`.
    Extra kwargs pass through to `compute_optimal_return`.
    """
    optimal = compute_optimal_return(envs, **kwargs)
    return optimal - policy_returns.detach().cpu().float()


# # #
# Self-validation: replay the DP's greedy optimum through the REAL env and check
# the realised reward2 return matches V_0. This cross-checks the compact model
# (transitions + reward) against potteryshop.step + rewards.reward2.


@torch.no_grad()
def _validate(envs: Environment, horizon: int = 64, discount: float = DISCOUNT_RATE,
              tol: float = 1e-4, device: torch.device | str | None = None) -> None:
    from rewards import reward2

    device = torch.device(device) if device is not None else torch.device("cpu")
    B = envs.num_envs
    shard_slot, urn_slot, init_shardmask, init_urncode, robot_cell, bin_cell, S, U, C, W = (
        _layout_tables(envs, device)
    )
    next_idx, reward, M_shard, M_urn, N = _build_transitions(
        shard_slot, urn_slot, bin_cell, S, U, C, W, discount, _BREAK_PENALTY, 0.0, device,
    )
    # store every V_t so we can extract the greedy action at each step
    V_stack = [torch.zeros((B, N), dtype=torch.float32, device=device)]
    flat_idx = next_idx.reshape(B, N * 6)
    for _ in range(horizon):
        V_next = torch.gather(V_stack[-1], 1, flat_idx).reshape(B, N, 6)
        V_stack.append((reward + discount * V_next).amax(dim=2))
    V_stack = V_stack[::-1]  # V_stack[t] = value with (horizon - t) steps left -> reverse so [0]=V_0
    start = (((robot_cell * 2) * M_shard + init_shardmask) * M_urn + init_urncode)
    V0 = V_stack[0].gather(1, start[:, None]).squeeze(1)

    # roll the greedy optimum through the real env, scoring with reward2
    env = envs.to(device)
    state = env.reset(num_rollouts=B)
    idx = start.clone()
    realised = torch.zeros(B, dtype=torch.float32, device=device)
    batch = torch.arange(B, device=device)
    for t in range(horizon):
        Q = reward[batch, idx] + discount * torch.gather(
            V_stack[t + 1], 1, next_idx[batch, idx]
        )  # [B, 6]
        action = Q.argmax(dim=1)
        next_state = env.step(state, action)
        r = reward2(state, action, next_state)
        realised += (discount ** t) * r
        state = next_state
        idx = next_idx[batch, idx, action]

    gap = (realised - V0).abs()
    print(f"  validation: max |realised - V0| = {gap.max().item():.2e} "
          f"(mean V0 = {V0.mean().item():+.3f})")
    assert gap.max().item() < tol, (
        f"compact model disagrees with the real env by {gap.max().item():.2e} "
        f"(> tol {tol}); the transition/reward model is wrong"
    )
    print("  validation PASSED")


def _urn_wall_level(world_size: int = 5) -> Environment:
    """Hand-built deployment level: a vertical urn wall between robot and bin."""
    W = world_size
    items = torch.zeros((W, W), dtype=torch.long)
    mid = W // 2
    items[:, mid] = Item.URN              # full vertical wall
    items[0, 0] = Item.SHARDS             # shards on the robot's side
    items[W - 1, 0] = Item.SHARDS
    items[1, 0] = Item.SHARDS
    robot = torch.tensor([mid, 0])
    bin_ = torch.tensor([mid, W - 1])     # bin on the far side
    return Environment(
        init_robot_pos=robot[None].expand(1, 2).clone(),
        init_items_map=items[None].clone(),
        bin_pos=bin_[None].expand(1, 2).clone(),
    )


if __name__ == "__main__":
    import functools
    from generate import generate

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    print("random fixed-bin layouts (4x4, 4 shards, 2 urns):")
    envs = generate(world_size=4, num_shards=4, num_urns=2, num_envs=64, generator=g)
    opt = compute_optimal_return(envs)
    print(f"  optimal return: mean {opt.mean():+.3f}  min {opt.min():+.3f}  max {opt.max():+.3f}")
    _validate(envs[:16])

    print("\nhand-built urn-wall deployment level (5x5):")
    wall = _urn_wall_level(5)
    opt = compute_optimal_return(wall)
    print(f"  optimal return: {opt.item():+.3f}")
    _validate(wall)

    # Sanity: with break penalty huge, the wall optimum should match routing that
    # never breaks (or be much lower if walking around overruns the horizon).
    print("\n  optimal with break_penalty=100 (breaking suppressed):")
    opt_nb = compute_optimal_return(wall, break_penalty=100.0)
    print(f"    {opt_nb.item():+.3f}  (vs {opt.item():+.3f} at default penalty)")
