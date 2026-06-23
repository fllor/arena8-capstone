"""
PLR rank-prioritised sampling demo (headless).

Samples a batch of fresh random levels, scores each by ORACLE REGRET
(regret = optimal_return - policy_return, ONE scalar per level), then draws a
handful of levels with PLR-style rank-based prioritisation and renders both the
chosen layouts and how the current agent acts in them.

Saves PNGs to ./plr_demo_out/ (no notebook frontend needed).
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch
from PIL import Image

from agent import ActorCriticNetwork
from evaluation import compute_return
from generate import generate
from potteryshop import Action, collect_rollout, tree_map
from rewards import reward2, DISCOUNT_RATE
from solver import compute_optimal_return
from train import default_device

OUT = "plr_demo_out"
os.makedirs(OUT, exist_ok=True)

# ---- config (mirrors run.py world_size=4) ---------------------------------
world_size = 4
shard_mean, urn_mean = 1.7, 1.3
NUM_LEVELS = 256       # fresh batch to score
NUM_PICK = 9           # how many to sample by priority
HORIZON = 64
BETA = 0.5             # PLR rank-prioritisation temperature (lower = peakier)
UPSCALE = 6

device = default_device()
print(f"device: {device}")

gen = functools.partial(
    generate, world_size=world_size, shard_mean=shard_mean, urn_mean=urn_mean
)

# ---- load the trained agent ------------------------------------------------
net = ActorCriticNetwork.init(
    obs_height=world_size, obs_width=world_size,
    net_channels=16, net_width=64,
    num_conv_layers=5, num_dense_layers=2,
    num_actions=len(Action),
    generator=torch.Generator().manual_seed(1),
)
net.load_state_dict(torch.load("agent.pt", map_location=device))
net = net.to(device)

# ---- sample + score a fresh batch by oracle regret -------------------------
envs = gen(num_envs=NUM_LEVELS, generator=torch.Generator().manual_seed(7)).to(device)

# achieved return: deterministic rollout (matches the deterministic oracle)
rollout = collect_rollout(
    env=envs, policy_fn=net.policy, num_steps=HORIZON, device=device,
    deterministic=True,
)
flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
rewards = reward2(flat.state, flat.action, flat.next_state).view(NUM_LEVELS, HORIZON)
achieved = compute_return(rewards, DISCOUNT_RATE).cpu()

optimal = compute_optimal_return(envs, horizon=HORIZON)  # [B] on CPU
regret = (optimal - achieved).clamp_min(0)               # per-level regret scalar

# ---- PLR rank-based prioritisation ----------------------------------------
# rank 1 = highest regret. P(level) prop. (1/rank)^(1/beta), sampled w/o replacement.
order = torch.argsort(regret, descending=True)           # indices, high->low regret
ranks = torch.empty(NUM_LEVELS)
ranks[order] = torch.arange(1, NUM_LEVELS + 1, dtype=torch.float)
weights = (1.0 / ranks) ** (1.0 / BETA)
probs = weights / weights.sum()
pick = torch.multinomial(probs, NUM_PICK, replacement=False,
                         generator=torch.Generator().manual_seed(0))

print(f"\nregret over {NUM_LEVELS} fresh levels: "
      f"mean={regret.mean():.3f} max={regret.max():.3f} "
      f"frac>0.05={ (regret>0.05).float().mean():.2%}")
print(f"\n{'pick':>4} {'level':>5} {'rank':>5} {'P':>7} "
      f"{'optimal':>8} {'achieved':>8} {'regret':>8}")
for i, b in enumerate(pick.tolist()):
    print(f"{i:>4} {b:>5} {int(ranks[b]):>5} {probs[b]:>7.4f} "
          f"{optimal[b]:>+8.3f} {achieved[b]:>+8.3f} {regret[b]:>+8.3f}")

# ---- render the chosen layouts + rollouts ----------------------------------
sel = envs[pick.tolist()]                # batch of NUM_PICK selected envs
proto = sel                              # render() only needs world_size + state

# deterministic rollout in the selected levels (this IS how the agent acts now)
sel_roll = collect_rollout(
    env=sel, policy_fn=net.policy, num_steps=HORIZON, device=device,
    deterministic=True,
)
# full state sequence: states + final next_state -> [NUM_PICK, HORIZON+1]
full = tree_map(
    lambda xs, xs_: torch.cat((xs, xs_[:, [-1]]), dim=1),
    sel_roll.transitions.state, sel_roll.transitions.next_state,
)

def up(img):  # nearest-neighbour upscale
    return img.repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1)

# (a) initial-layout grid (3x3)
gw = 3
tiles = [up(proto.render(tree_map(lambda x: x[:, 0], full), index=b))
         for b in range(NUM_PICK)]
h, w, _ = tiles[0].shape
grid = np.ones(((h + 4) * gw, (w + 4) * gw, 3), dtype=np.uint8) * 255
for b, t in enumerate(tiles):
    r, c = divmod(b, gw)
    grid[r*(h+4):r*(h+4)+h, c*(w+4):c*(w+4)+w] = t
Image.fromarray(grid).save(f"{OUT}/layouts.png")

# (b) per-level filmstrip of how the agent acts (timesteps across, levels down)
ts = list(range(0, 28, 3))               # show first ~28 steps
rows = []
for b in range(NUM_PICK):
    frames = [up(proto.render(tree_map(lambda x: x[:, t], full), index=b)) for t in ts]
    row = np.concatenate(
        [np.pad(f, ((0, 0), (0, 4), (0, 0)), constant_values=255) for f in frames],
        axis=1,
    )
    rows.append(np.pad(row, ((0, 4), (0, 0), (0, 0)), constant_values=255))
film = np.concatenate(rows, axis=0)
Image.fromarray(film).save(f"{OUT}/rollouts_filmstrip.png")

print(f"\nsaved {OUT}/layouts.png and {OUT}/rollouts_filmstrip.png "
      f"(filmstrip columns = t={ts})")
