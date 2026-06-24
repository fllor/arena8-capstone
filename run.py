"""
Interactive driver for UED training in the pottery shop -- DR and PLR in one file.

Structured as `# %%` cells (run cell-by-cell in VS Code / Jupyter, or
top-to-bottom as a plain script). All the heavy lifting lives in the sibling
modules; this file just wires them together and inspects the results.

Pick the curriculum with `METHOD` in the config cell (or as the first CLI arg):

* `"dr"`           -- domain randomisation: train on every fresh batch, no buffer.
* `"plr25/50/75"`  -- PLR-bot (robust PLR) at replay_prob 0.25/0.50/0.75: a
                      regret-keyed buffer with replay-only gradients (the
                      stop-gradient). The project's headline method.

Every shared PPO hyperparameter is held identical across methods; only the
curriculum knobs (`replay_prob`, `train_on_generate`) and the step budget differ,
so a DR-vs-PLR comparison isolates the curriculum. The step budget is set for an
**equal gradient-update count**: DR updates every step, PLR only on its replay
steps (~`replay_prob` of them), so PLR runs `1/replay_prob`x more steps.
"""

# %%

import sys
import functools
import csv

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from evalsuite import build_eval_sets
from generate import generate
from potteryshop import Action, Item
from rewards import reward2
from train import UEDConfig, default_device, train_agent
from utils import rollout_regret_grid, wall_envs

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

# Which curriculum to run. DR vs robust PLR
METHOD = "dr"  # "dr" | "plr50"
# Number of gradient updates (DR equivalent)
# ~1 min / 100 steps for DR
NUM_GRAD_UPDATES = 500

if len(sys.argv) > 1:
    METHOD = sys.argv[1]
if len(sys.argv) > 2:
    NUM_GRAD_UPDATES = int(sys.argv[2])

CURRICULA = {
    "dr":    dict(replay_prob=0.00, train_on_generate=True,  wandb_run_name=f"dr_{NUM_GRAD_UPDATES}"),
    "plr50": dict(replay_prob=0.50, train_on_generate=False, wandb_run_name=f"plr_p50_{NUM_GRAD_UPDATES}"),
    "plr75": dict(replay_prob=0.75, train_on_generate=False, wandb_run_name=f"plr_p75_{NUM_GRAD_UPDATES}"),
    "plr25": dict(replay_prob=0.25, train_on_generate=False, wandb_run_name=f"plr_p25_{NUM_GRAD_UPDATES}"),
    #"plr_plain": dict(replay_prob=0.5, train_on_generate=True),
}
assert METHOD in CURRICULA
assert NUM_GRAD_UPDATES > 0
curriculum = CURRICULA[METHOD]
RUN_NAME = curriculum["wandb_run_name"]
print("Run:", RUN_NAME)

# Mean shard/urn COUNT per env (each floored at 1, drawn from a truncated
# geometric — most layouts stay sparse, dense urn-walls keep a small non-zero
# probability in the tail). `urn_mean` is the key knob: keep it low so "walk
# around urns" is almost always optimal (the GMG proxy forms and forced walls
# stay RARE — that rarity is exactly what PLR⊥ has to overcome). Raise it if you
# want walls to show up more often in random eval. On 5x5 a wall needs ~5 urns,
# so walls live in the geometric tail.
world_size = 4
if world_size == 4:
    shard_mean = 1.7
    urn_mean = 1.3
elif world_size == 5:
    shard_mean = 2.0
    urn_mean = 1.7

# Where to save/load the trained network (per-method so DR and PLR don't clobber
# each other). Set LOAD_AGENT = True to skip training and load weights instead
# (the architecture below must match the one the file was saved from).
MODEL_PATH = f"agent_{RUN_NAME}.pt"
LOAD_AGENT = False

# Optional Weights & Biases logging. Set WANDB_PROJECT to a project name to log
# the per-step metrics there; leave it None to disable.
WANDB_PROJECT = None
WANDB_PROJECT = "arena8-capstone"

# the fixed-bin layout distribution, shared by training, eval, and viz
gen = functools.partial(
    generate,
    world_size=world_size,
    shard_mean=shard_mean,
    urn_mean=urn_mean,
)

# %% Build the agent

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

# %% Train (or load a saved agent instead)

# Held-out eval populations (in-distribution random + hand-built urn-walls).
# `train_agent` scores these with `compute_eval_metrics` every `eval_every` steps.
eval_sets = build_eval_sets(world_size, shard_mean, urn_mean)

# Equal gradient-update budget across methods: DR updates every step, PLR only on
# its replay steps. Scale the step count by 1/replay_prob so both take the same
# number of PPO updates (the fair DR-vs-PLR comparison; see module docstring).
replay_prob = curriculum["replay_prob"]
num_train_steps = (
    NUM_GRAD_UPDATES if replay_prob == 0 else round(NUM_GRAD_UPDATES / replay_prob)
)

config = UEDConfig(
    gen=gen,
    net=net,
    reward_fn=reward2,
    num_train_steps=num_train_steps+1,  # +1 to compute final metrics
    num_envs=8192,
    num_env_steps=64,
    num_epochs=1,
    num_minibatches=32,
    lr=0.003,  # large batch permits greater learning rate
    device=device,
    seed=1,
    eval_sets=eval_sets,
    wandb_project=WANDB_PROJECT,
    #wandb_run_name=METHOD,
    **curriculum,
)

if LOAD_AGENT:
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net = net.to(device)
    history, sampler = None, None
    print(f"loaded agent from {MODEL_PATH} (skipped training)")
else:
    net, history, sampler = train_agent(config)
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"saved agent to {MODEL_PATH} ({METHOD}, {num_train_steps} steps)")

# %%
# Plot training curves.
# A row is recorded every step (tagged with its `step` index), but rows carry
# different keys, so `series()` skips entries missing a key. DR plots
# ppo/return + ppo/loss + ppo/entropy (every step) and regret/generate (only on
# the subsampled `dr_diag_every` diagnostic steps); PLR plots branch-split regret +
# buffer composition (the buffer's mean urn count climbing is PLR learning to
# prefer walls).


def series(key):
    """(xs, ys) for a metric, skipping entries that don't carry it (PLR's
    generate/replay branches log different keys)."""
    pairs = [(m["step"], m[key]) for m in (history or []) if key in m]
    return [x for x, _ in pairs], [y for _, y in pairs]


if history is not None:
    if sampler is not None:  # PLR: regret-by-branch + buffer composition
        panels = [
            ("regret/replay", "replay regret (buffer levels)"),
            ("regret/generate", "generate regret (fresh levels)"),
            ("buffer/mean_urns", "buffer mean urn count"),
            ("buffer/max_score", "buffer max regret score"),
        ]
    else:  # DR: the classic return/loss/regret/entropy panels
        panels = [
            ("ppo/return", "mean discounted return"),
            ("ppo/loss", "PPO loss"),
            ("regret/generate", "mean oracle regret"),
            ("ppo/entropy", "policy entropy"),
        ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 9), sharex=True)
    for ax, (key, label) in zip(axes, panels):
        xs, ys = series(key)
        ax.plot(xs, ys, lw=0.8)
        ax.set_ylabel(label)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[-1].set_xlabel("train step")
    fig.tight_layout()
    plt.show()

# %%
# Held-out eval curves: random competence vs urn-wall performance

if history is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("eval/random/stochastic/regret", "random regret"),
        ("eval/walls/stochastic/regret", "urn-wall regret"),
    ]:
        xs, ys = series(key)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("train step")
    ax.set_ylabel("held-out oracle regret")
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.5)
    plt.show()

# %%
# PLR only: Inspect the highest-regret levels the buffer collected

if sampler is not None and sampler.size > 0:
    n = sampler.size
    top = sampler.scores[:n].topk(min(16, n)).indices
    top_levels = sampler.levels[top]
    urns = (top_levels.init_items_map == Item.URN).flatten(1).sum(1)
    print("top buffer regret scores:", sampler.scores[top].tolist())
    print("their urn counts:", urns.tolist())
    visualise.display_envs(
        top_levels, grid_width=4, title="highest-regret buffer levels"
    )

# %%
# Visualise grid of rollouts

grid_width = 6
watch_envs = gen(num_envs=grid_width**2, generator=torch.Generator().manual_seed(3))
regrets = rollout_regret_grid(net, watch_envs, grid_width=grid_width, device=device)

# %%
# Test walking around wall

walls = wall_envs(world_size)
wall_regrets = rollout_regret_grid(net, walls, device=device)

# %%
# Dump the eval-step rows to a CSV for offline processing / plotting.

if history is not None:
    eval_rows = history[:: config.eval_every]
    fieldnames = list(dict.fromkeys(k for row in eval_rows for k in row))
    dump_path = f"history_eval_{RUN_NAME}.csv"
    with open(dump_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="nan")
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"wrote {len(eval_rows)} eval rows x {len(fieldnames)} cols -> {dump_path}")
# %%
