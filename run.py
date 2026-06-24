"""
Interactive driver for UED training in the pottery shop -- DR and PLR in one file.

Structured as `# %%` cells (run cell-by-cell in VS Code / Jupyter, or
top-to-bottom as a plain script). All the heavy lifting lives in the sibling
modules; this file just wires them together and inspects the results.

Pick the curriculum with `METHOD` in the config cell:

* `"dr"`   -- domain randomisation: train on every fresh batch, no buffer.
* `"plr"`  -- PLR-bot (robust PLR): regret-keyed buffer, replay-only gradients
              (the stop-gradient). The project's headline method.
* `"plr_plain"` -- non-robust PLR: like `"plr"` but also trains on fresh batches.

Every shared PPO hyperparameter is held identical across methods; only the
curriculum knobs (`replay_prob`, `train_on_generate`) and the step budget differ,
so a DR-vs-PLR comparison isolates the curriculum. The step budget is set for an
**equal gradient-update count**: DR updates every step, PLR only on its replay
steps (~`replay_prob` of them), so PLR runs `1/replay_prob`x more steps.
"""

# %%

import functools

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from evalsuite import make_eval_fn
from generate import generate
from potteryshop import Action, Item
from rewards import reward2
from train import UEDConfig, default_device, train_agent
from utils import rollout_regret_grid, wall_envs

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

# Which curriculum to run. DR vs PLR-bot is the core comparison; "plr_plain" is a
# non-robust ablation. Each entry is just the two curriculum knobs on UEDConfig --
# everything else below is shared, so the comparison is clean.
METHOD = "plr"  # "dr" | "plr" | "plr_plain"
CURRICULA = {
    "dr": dict(replay_prob=0.0, train_on_generate=True),
    "plr": dict(replay_prob=0.5, train_on_generate=False),  # PLR-bot (stop-grad)
    "plr_plain": dict(replay_prob=0.5, train_on_generate=True),
}
curriculum = CURRICULA[METHOD]

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
MODEL_PATH = f"agent_{METHOD}.pt"
LOAD_AGENT = False

# Optional Weights & Biases logging. Set WANDB_PROJECT to a project name to log
# the scored-step metrics there; leave it None to disable.
WANDB_PROJECT = None
# WANDB_PROJECT = "arena8-capstone"

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

eval_fn = make_eval_fn(gen, device)

# Equal gradient-update budget across methods: DR updates every step, PLR only on
# its replay steps. Scale the step count by 1/replay_prob so both take the same
# number of PPO updates (the fair DR-vs-PLR comparison; see module docstring).
GRAD_COLLECTIONS = 300  # ~3 min for DR
replay_prob = curriculum["replay_prob"]
num_train_steps = (
    GRAD_COLLECTIONS if replay_prob == 0 else round(GRAD_COLLECTIONS / replay_prob)
)

config = UEDConfig(
    gen=gen,
    net=net,
    reward_fn=reward2,
    num_train_steps=num_train_steps,
    num_envs=8192,
    num_env_steps=64,
    num_epochs=1,
    minibatch_size=16384,  # 32 gradient updates per collection
    lr=0.003,  # large batch permits greater learning rate
    device=device,
    seed=1,
    log_every=10,
    eval_fn=eval_fn,
    eval_every=50,
    wandb_project=WANDB_PROJECT,
    wandb_run_name=METHOD,
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
# History is sparse (one entry per logged step, each tagged
# with its true `step` index). DR logs return/loss/regret/entropy on its scored
# steps; PLR logs branch-split regret + buffer composition (the buffer's mean urn
# count climbing is PLR learning to prefer walls).


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
            ("return", "mean discounted return"),
            ("loss", "PPO loss"),
            ("regret", "mean oracle regret"),
            ("entropy", "policy entropy"),
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
        ("eval/random_regret", "random regret"),
        ("eval/wall_regret", "urn-wall regret"),
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
