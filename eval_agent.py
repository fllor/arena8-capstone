"""
Evaluate a saved pottery-shop agent: oracle regret + break-through rate on the
held-out random and hand-built urn-wall populations, plus a rendered filmstrip
of how it acts on the walls (does it break through, or walk around = GMG?).

Reuses the existing harness in `overnight_run1/evalsuite.py`. Loads the network
config from the model's sidecar JSON, so the architecture/distribution always
match what the file was trained with.

    python eval_agent.py overnight_run1/agent_long_4x4.pt
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "overnight_run1"))

import evalsuite  # noqa: E402  (from overnight_run1/)
from agent import ActorCriticNetwork  # noqa: E402
from potteryshop import Action, collect_rollout, tree_map  # noqa: E402
from train import default_device  # noqa: E402


def load_agent(pt_path: str):
    """Build the net from the sidecar JSON config and load its weights."""
    cfg = json.load(open(pt_path.rsplit(".", 1)[0] + ".json"))["cfg"]
    ws = cfg["world_size"]
    device = default_device()
    net = ActorCriticNetwork.init(
        obs_height=ws, obs_width=ws, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )
    net.load_state_dict(torch.load(pt_path, map_location=device))
    return net.to(device), cfg


def render_wall_rollout(net, world_size: int, out_path: str, horizon: int = 40,
                        upscale: int = 6):
    """Greedy-rollout the policy on each hand-built wall and save a filmstrip."""
    walls = evalsuite.wall_envs(world_size)
    device = next(net.parameters()).device
    roll = collect_rollout(
        env=walls.to(device), policy_fn=net.policy, num_steps=horizon,
        generator=torch.Generator().manual_seed(0), device=device,
        deterministic=True,
    )
    full = tree_map(
        lambda xs, xs_: torch.cat((xs, xs_[:, [-1]]), dim=1),
        roll.transitions.state, roll.transitions.next_state,
    )
    ts = list(range(0, min(horizon + 1, 28), 3))

    def up(img):
        return img.repeat(upscale, axis=0).repeat(upscale, axis=1)

    rows = []
    for b in range(walls.num_envs):
        frames = [up(walls.render(tree_map(lambda x: x[:, t], full), index=b))
                  for t in ts]
        row = np.concatenate(
            [np.pad(f, ((0, 0), (0, 4), (0, 0)), constant_values=255) for f in frames],
            axis=1,
        )
        rows.append(np.pad(row, ((0, 4), (0, 0), (0, 0)), constant_values=255))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_path)
    return ts


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "overnight_run1/agent_long_4x4.pt"
    net, cfg = load_agent(pt_path)
    print(f"loaded {pt_path}  ({cfg['name']}, world_size={cfg['world_size']}, "
          f"trained {cfg['steps']} steps)\n")

    sets = evalsuite.build_eval_sets(
        cfg["world_size"], cfg["shard_mean"], cfg["urn_mean"], n_random=2000,
    )
    results = evalsuite.evaluate_all(net, sets)
    for res in results.values():
        print(res)

    walls = results["walls"]
    print(f"\nGMG read: walls break-through rate = {walls.break_rate:.0%}, "
          f"regret = {walls.mean_regret:+.3f}  "
          f"({'MISGENERALISES (walks around)' if walls.break_rate < 0.5 else 'breaks through'})")

    out_dir = "eval_out"
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(pt_path))[0]
    ts = render_wall_rollout(net, cfg["world_size"], f"{out_dir}/{tag}_walls.png")
    print(f"\nsaved wall rollout filmstrip -> {out_dir}/{tag}_walls.png "
          f"(columns = t={ts})")


if __name__ == "__main__":
    main()
