"""
Render animated GIFs of a trained agent acting in the pottery shop -- headless,
no notebook frontend needed.

Reuses `visualise.animate_rollouts` (the same frame builder behind the notebook's
`display_rollout(s)`) but writes the GIF bytes to a file instead of an ipywidget.
The agent is rebuilt with the fixed `run.py` architecture (net_channels=16,
net_width=64, 5 conv / 2 dense), so no sidecar JSON is required.

Examples
--------
    # one combined grid GIF on the hand-built urn walls (the GMG money shot):
    python make_gifs.py agent_accel_walk_500.pt --envs walls

    # a grid GIF on random layouts, plus one GIF per env:
    python make_gifs.py agent_dr_10k.pt --envs random --grid-width 5 --per-env

Output goes to ./gif_out/.
"""

from __future__ import annotations

import argparse
import functools
import io
import os

import einops
import numpy as np
import torch
from PIL import Image

import visualise
from agent import ActorCriticNetwork
from evalsuite import wall_envs
from generate import generate
from potteryshop import Action, collect_rollout, tree_map
from solver import optimal_rollout
from train import default_device

WORLD_SIZE = 4  # the run.py default; bump to 5 if you trained 5x5 agents


def build_net(device):
    """Rebuild the run.py architecture and load saved weights into it."""
    return ActorCriticNetwork.init(
        obs_height=WORLD_SIZE, obs_width=WORLD_SIZE,
        net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2,
        num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    ).to(device)


def save_gif(frames, path, upscale, duration=100):
    """Upscale (nearest-neighbour) and write a frame stack to an animated GIF."""
    frames = einops.repeat(
        np.asarray(frames), "t h w rgb -> t (h h2) (w w2) rgb", h2=upscale, w2=upscale
    )
    imgs = [Image.fromarray(f) for f in frames]
    # optimize=False + disposal=1 writes each frame in full (no transparent
    # inter-frame diffs), so even isolated-frame extraction renders correctly.
    imgs[0].save(
        path, format="gif", save_all=True, append_images=imgs[1:],
        duration=duration, loop=0, optimize=False, disposal=1,
    )
    print(f"wrote {path}  ({len(imgs)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", nargs="?",
                    help="path to a saved agent .pt (run.py architecture); "
                         "omit when using --optimal")
    ap.add_argument("--optimal", action="store_true",
                    help="render the oracle's optimal trajectory instead of an agent")
    ap.add_argument("--envs", choices=["walls", "random"], default="walls")
    ap.add_argument("--grid-width", type=int, default=3,
                    help="envs per row in the grid GIF (random only; walls uses its 3)")
    ap.add_argument("--horizon", type=int, default=64, help="rollout length (steps)")
    ap.add_argument("--seed", type=int, default=3, help="random-env / sampling seed")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax actions instead of sampling (default: stochastic)")
    ap.add_argument("--per-env", action="store_true",
                    help="also emit one GIF per env, not just the combined grid")
    ap.add_argument("--out-dir", default="gif_out")
    args = ap.parse_args()
    if not args.optimal and args.agent is None:
        ap.error("pass an agent .pt, or use --optimal to render the oracle path")

    device = default_device()

    # Build the env population.
    if args.envs == "walls":
        envs = wall_envs(WORLD_SIZE)
        # grid the levels as wide as possible while still tiling evenly (the
        # hand-built wall set size varies: 2 on 4x4, more on 5x5).
        B = envs.init_items_map.shape[0]
        grid_width = min(3, B)
        while B % grid_width:
            grid_width -= 1
    else:
        n = args.grid_width ** 2
        gen = functools.partial(generate, world_size=WORLD_SIZE,
                                shard_mean=1.7, urn_mean=1.3)
        envs = gen(num_envs=n, generator=torch.Generator().manual_seed(args.seed))
        grid_width = args.grid_width

    # Roll out: either the oracle optimum, or the trained agent's policy.
    if args.optimal:
        roll = optimal_rollout(envs, horizon=args.horizon, device=device)
        tag = "optimal"
        print(f"rendering oracle optimal trajectory on {device}")
    else:
        net = build_net(device)
        net.load_state_dict(torch.load(args.agent, map_location=device))
        print(f"loaded {args.agent} on {device}")
        roll = collect_rollout(
            env=envs.to(device), policy_fn=net.policy, num_steps=args.horizon,
            generator=torch.Generator().manual_seed(args.seed), device=device,
            deterministic=args.deterministic,
        )
        tag = os.path.splitext(os.path.basename(args.agent))[0]
    roll = tree_map(lambda x: x.cpu(), roll)
    envs = envs.to("cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # Combined grid GIF (one tile per env, animated together).
    grid_frames = visualise.animate_rollouts(envs[0], roll, grid_width=grid_width)
    save_gif(grid_frames, f"{args.out_dir}/{tag}_{args.envs}_grid.gif", upscale=8)

    # Optional per-env GIFs.
    if args.per_env:
        for b in range(envs.num_envs):
            one = tree_map(lambda x: x[b:b + 1], roll)
            frames = visualise.animate_rollouts(envs[0], one, grid_width=1)
            save_gif(frames, f"{args.out_dir}/{tag}_{args.envs}_env{b}.gif", upscale=12)


if __name__ == "__main__":
    main()
