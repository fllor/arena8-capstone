"""
Shared helpers extracted from the interactive `run.py` driver so they can be
reused outside the notebook cells.
"""

import torch

import visualise
from evaluation import compute_return
from potteryshop import Environment, collect_rollout, tree_map
from rewards import reward2, DISCOUNT_RATE
from solver import compute_optimal_return
from train import default_device


def _stack_layouts(layouts, world_size):
    """Build a batched Environment from a list of (robot, bin, items_grid)."""
    robots, bins, items = [], [], []
    for robot, bin_, grid in layouts:
        g = torch.tensor(grid, dtype=torch.long)
        assert g.shape == (world_size, world_size), f"bad grid shape {g.shape}"
        robots.append(torch.tensor(robot, dtype=torch.long))
        bins.append(torch.tensor(bin_, dtype=torch.long))
        items.append(g)
    return Environment(
        init_robot_pos=torch.stack(robots),
        init_items_map=torch.stack(items),
        bin_pos=torch.stack(bins),
    )


def wall_envs(world_size):
    """
    Hand-built urn-wall deployment levels for the given grid size.

    Each level walls off the bin's neighbourhood with a column of urns; shards
    sit behind the wall. Breaking straight through is the optimal shortcut, but
    walking around is always *possible* (levels stay solvable). The 4x4 set is
    the escalating wall from `run.py`; the 5x5 set is a deeper wall.
    """
    if world_size == 4:
        layouts = [
            ((0, 0), (0, 0), ((0, 2, 1, 1),
                              (0, 2, 1, 1),
                              (0, 0, 0, 0),
                              (0, 0, 0, 0))),
            ((0, 0), (0, 0), ((0, 2, 1, 1),
                              (0, 2, 1, 1),
                              (0, 2, 0, 0),
                              (0, 0, 0, 0))),
            ((0, 0), (0, 0), ((0, 2, 1, 1),
                              (0, 2, 1, 1),
                              (0, 2, 2, 0),
                              (0, 0, 0, 0))),
        ]
        return _stack_layouts(layouts, 4)
    if world_size == 5:
        # A vertical urn wall in the middle column with shards behind it; robot
        # and bin straddle the wall. Escalating wall height (partial -> full).
        layouts = [
            ((2, 0), (0, 0), ((0, 0, 2, 1, 0),
                              (0, 0, 2, 1, 0),
                              (0, 0, 2, 0, 1),
                              (0, 0, 0, 1, 0),
                              (0, 0, 0, 0, 0))),
            ((2, 0), (0, 0), ((0, 0, 2, 1, 0),
                              (0, 0, 2, 1, 0),
                              (0, 0, 2, 0, 1),
                              (0, 0, 2, 1, 0),
                              (0, 0, 2, 0, 0))),
            ((2, 0), (0, 0), ((0, 0, 2, 1, 1),
                              (0, 0, 2, 0, 1),
                              (0, 0, 2, 1, 0),
                              (0, 0, 2, 0, 1),
                              (0, 0, 2, 1, 0))),
        ]
        return _stack_layouts(layouts, 5)
    raise ValueError(f"no hand-built wall set for world_size={world_size}")


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
