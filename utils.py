"""
Shared helpers extracted from the interactive `run.py` driver so they can be
reused outside the notebook cells.
"""

import torch

import visualise
from evaluation import compute_return
from evalsuite import wall_envs  # canonical urn-wall set (re-exported for callers)
from potteryshop import collect_rollout, tree_map
from rewards import reward2, DISCOUNT_RATE
from solver import compute_optimal_return
from train import default_device

__all__ = ["wall_envs", "rollout_regret_grid"]


def rollout_regret_grid(
    net,
    envs,
    *,
    grid_width=None,
    watch_steps=96,
    seed=3,
    device=None,
    plot=True,
    do_print=True,
):
    """Roll out `net` on a batch of layouts and score per-level oracle regret.

    Collects a rollout with `net` on `envs` and compares the achieved discounted
    return against the oracle optimum (per-level regret = optimal - achieved).

    Args:
        net: the actor-critic agent to roll out.
        envs: a batched `Environment` to roll out on (e.g. from `gen(...)` or
            `wall_envs(...)`).
        grid_width: columns to use when displaying the rollout grid; defaults to
            one row (`envs.num_envs`).
        watch_steps: rollout horizon.
        seed: seed for the rollout actions.
        device: torch device; defaults to `default_device()`.
        plot: if True, display the rollout grid.
        do_print: if True, print the per-env optimal/achieved/regret table.

    Returns:
        dict with per-env tensors `optimal`, `achieved`, `regret`, plus the
        `rollout`.
    """
    if device is None:
        device = default_device()

    num_envs = envs.num_envs
    if grid_width is None:
        grid_width = num_envs

    rollout = collect_rollout(
        env=envs,
        policy_fn=net.policy,
        num_steps=watch_steps,
        generator=torch.Generator().manual_seed(seed),
        device=device,
        deterministic=False,
    )

    # Optimal return (oracle) vs the return actually achieved in the rollout.
    # The achieved return is scored on the *same* trajectory being animated, so
    # the numbers match what you see; the gap is the per-level regret.
    optimal = compute_optimal_return(envs, horizon=watch_steps)
    flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
    rewards = reward2(flat.state, flat.action, flat.next_state).view(num_envs, watch_steps)
    achieved = compute_return(rewards, discount_rate=DISCOUNT_RATE).cpu()
    regret = optimal - achieved

    if do_print:
        print(f"{'env (row,col)':>14}  {'optimal':>8}  {'achieved':>8}  {'regret':>8}")
        for b in range(num_envs):
            print(
                f"{f'{b:>2} ({b // grid_width},{b % grid_width})':>14}"
                f"  {optimal[b].item():>+8.3f}  {achieved[b].item():>+8.3f}  {regret[b].item():>+8.3f}"
            )
        print(
            f"{'mean':>14}  {optimal.mean().item():>+8.3f}"
            f"  {achieved.mean().item():>+8.3f}  {regret.mean().item():>+8.3f}"
        )

    if plot:
        visualise.display_rollouts(envs, rollout, grid_width=grid_width)

    return {
        "optimal": optimal,
        "achieved": achieved,
        "regret": regret,
        "rollout": rollout,
    }
