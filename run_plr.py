"""
Interactive driver for robust PLR (PLR-bot) training in the pottery shop.

The DR counterpart is `run.py`; this swaps `train_agent_multienv` for
`train_agent_plr`, which curates a regret-prioritised level buffer instead of
training on every freshly-sampled batch. Structured as `# %%` cells (run
cell-by-cell in VS Code / Jupyter, or top-to-bottom as a plain script).

The PPO defaults in `train_agent_plr` are already tuned (see its docstring: ~32
updates per collection, lr 0.003, replay_prob 0.5); this driver just wires the
generator, network, eval and buffer inspection together.
"""

# %%

import functools

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from evalsuite import build_walls, make_eval_fn
from generate import generate
from plr import PLRConfig, train_agent_plr
from potteryshop import Action, Item
from rewards import reward2
from train import default_device

device = default_device()
print(f"using device: {device}")

# %%
# Configuration -- same base distribution as the DR run (see run.py). PLR's job
# is to amplify the rare high-regret urn-walls this generator only occasionally
# samples; the base distribution itself is unchanged.

world_size = 4
shard_mean = 1.7
urn_mean = 1.3

WANDB_PROJECT = None  # set to a project name to log return/regret/buffer metrics

gen = functools.partial(
    generate,
    world_size=world_size,
    shard_mean=shard_mean,
    urn_mean=urn_mean,
)

# %%
# Build the agent (identical architecture to run.py)

net = ActorCriticNetwork.init(
    obs_height=world_size,
    obs_width=world_size,
    net_channels=16,
    net_width=64,
    num_conv_layers=5,
    num_dense_layers=2,
    num_actions=len(Action),
    generator=torch.Generator().manual_seed(1),
)

# %%
# Train with robust PLR. Held-out regret (random + hand-built urn-walls) is
# logged every `eval_every` steps via the eval hook, alongside the buffer's
# composition so you can watch it concentrate on denser, higher-regret levels.

eval_fn = make_eval_fn(gen, device)

config = PLRConfig(
    gen=gen,
    net=net,
    reward_fn=reward2,
    num_train_steps=600,  # ~5 min on an A40
    num_envs=256*8,
    replay_prob=0.5,
    buffer_capacity=4096,
    device=device,
    seed=1,
    eval_fn=eval_fn,
    eval_every=100,
    log_every=10,
    wandb_project=WANDB_PROJECT,
)

net, history, sampler = train_agent_plr(config)

# %%
# Plot training-side regret (split by branch) and the buffer composition over
# time -- the buffer's mean urn count climbing is PLR learning to prefer walls.

steps = [m["step"] for m in history]


def series(key):
    return [m["step"] for m in history if key in m], [
        m[key] for m in history if key in m
    ]


fig, axes = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
for ax, (key, label) in zip(
    axes,
    [
        ("regret/replay", "replay regret (buffer levels)"),
        ("regret/generate", "generate regret (fresh levels)"),
        ("buffer/mean_urns", "buffer mean urn count"),
        ("buffer/max_score", "buffer max regret score"),
    ],
):
    xs, ys = series(key)
    ax.plot(xs, ys, lw=0.8)
    ax.set_ylabel(label)
    ax.grid(True, lw=0.4, alpha=0.5)
axes[-1].set_xlabel("train step")
fig.tight_layout()
plt.show()

# %%
# Held-out eval curves: random competence vs urn-wall performance.

fig, ax = plt.subplots(figsize=(7, 4))
for key, label in [
    ("eval/random_regret", "random regret"),
    ("eval/wall_regret", "urn-wall regret"),
]:
    xs, ys = series(key)
    ax.plot(xs, ys, marker="o", label=label)
ax.set_xlabel("train step")
ax.set_ylabel("held-out oracle regret")
ax.legend()
ax.grid(True, lw=0.4, alpha=0.5)
plt.show()

# %%
# Inspect the highest-regret levels the buffer has collected (the walls it built
# a curriculum around). Needs a notebook frontend.

n = sampler.size
top = sampler.scores[:n].topk(min(16, n)).indices
top_levels = sampler.levels[top]
urns = (top_levels.init_items_map == Item.URN).flatten(1).sum(1)
print("top buffer regret scores:", sampler.scores[top].tolist())
print("their urn counts:", urns.tolist())
visualise.display_envs(top_levels, grid_width=4, title="highest-regret buffer levels")

# %% Visualise grid of rollouts
grid_width = 6
watch_envs = gen(num_envs=grid_width**2, generator=torch.Generator().manual_seed(3))
regrets = rollout_regret_grid(net, watch_envs, grid_width=grid_width, device=device)
# %% Test walking around wall
walls = wall_envs(world_size)
wall_regrets = rollout_regret_grid(net, walls, device=device)
# %%
