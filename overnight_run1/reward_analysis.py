"""
Oracle analysis of the reward landscape -- no training required.

For each (break_penalty, step_cost) reward setting we ask the *exact* solver two
questions that decide whether goal misgeneralisation can even appear:

  * walls: on each hand-built urn wall, is BREAKING THROUGH optimal? (compare the
    optimal return to the optimal with breaking suppressed, break_penalty=1e6).
  * random training layouts: in what FRACTION is breaking optimal, and how much
    return does the "never break" proxy give up on average (mean regret of the
    best non-breaking policy vs the true optimum)?

Reading the table:
  * GMG window (experiment B): breaking optimal on walls (wall_break high) but
    almost never in training (train_break_frac ~ 0). An agent trained here learns
    "never break" and should misgeneralise on the walls.
  * All-optimal regime (experiment A): make breaking clearly optimal (low break
    penalty / high step cost) so the agent that breaks through is simply optimal.

Runs on CPU (the batches are small), so it is safe to run alongside a GPU job.
"""

from __future__ import annotations

import torch

import rewards
from evalsuite import build_eval_sets, wall_envs
# master renamed the grouped/safe solver entry point to `compute_optimal_return`.
from solver import compute_optimal_return as compute_optimal_return_grouped

NO_BREAK = 1e6  # break penalty so large the optimal never smashes an urn


def analyse(world_size: int, break_penalty: float, step_cost: float,
            random_envs, walls, horizon: int) -> dict:
    rewards.set_reward_params(break_penalty=break_penalty, step_cost=step_cost)

    # walls: is breaking optimal on each?
    opt_w = compute_optimal_return_grouped(walls, horizon=horizon)
    opt_w_nb = compute_optimal_return_grouped(walls, horizon=horizon, break_penalty=NO_BREAK)
    wall_breaks = (opt_w > opt_w_nb + 1e-3)
    wall_gap = (opt_w - opt_w_nb)  # how much breaking is worth on each wall

    # random training layouts: fraction where breaking is optimal + proxy regret
    opt_r = compute_optimal_return_grouped(random_envs, horizon=horizon)
    opt_r_nb = compute_optimal_return_grouped(random_envs, horizon=horizon, break_penalty=NO_BREAK)
    train_break_frac = (opt_r > opt_r_nb + 1e-3).float().mean().item()
    proxy_regret = (opt_r - opt_r_nb).clamp_min(0).mean().item()  # avg loss of never-break

    return dict(
        break_penalty=break_penalty, step_cost=step_cost,
        wall_break_optimal=[bool(b) for b in wall_breaks.tolist()],
        wall_break_gap=[round(g, 3) for g in wall_gap.tolist()],
        train_break_frac=round(train_break_frac, 4),
        train_proxy_regret=round(proxy_regret, 4),
    )


def main() -> None:
    import sys
    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    n_random = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    shard_mean, urn_mean = (1.7, 1.3) if ws == 4 else (2.0, 1.7)
    print(f"world_size={ws} horizon={horizon} n_random={n_random} "
          f"shard_mean={shard_mean} urn_mean={urn_mean}", flush=True)
    sets = build_eval_sets(ws, shard_mean=shard_mean, urn_mean=urn_mean, n_random=n_random)
    random_envs, walls = sets["random"], sets["walls"]

    print(f"{'bp':>5} {'step':>5} | {'wall breaks':>12} {'wall gaps':>22} "
          f"| {'train_break%':>12} {'proxy_regret':>12}")
    grid = []
    for bp in (3.0, 2.0, 1.5, 1.0, 0.5):
        for sc in (0.02, 0.1, 0.2):
            r = analyse(ws, bp, sc, random_envs, walls, horizon)
            grid.append(r)
            nb = sum(r["wall_break_optimal"])
            print(f"{bp:>5} {sc:>5} | {nb}/{len(r['wall_break_optimal'])} broken "
                  f"{str(r['wall_break_gap']):>22} | {r['train_break_frac']*100:>10.2f}% "
                  f"{r['train_proxy_regret']:>12.4f}", flush=True)
    import json
    out = f"reward_analysis_{ws}x{ws}.jsonl"
    with open(out, "w") as f:
        for r in grid:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
