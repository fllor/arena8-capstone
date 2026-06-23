"""
Interactive driver for the 5x5 pottery shop -- a copy of `run.py` adapted to the
larger grid (world_size=5).

What changes vs the 4x4 `run.py`:
* `world_size = 5` (shard_mean/urn_mean follow the 5x5 branch);
* a longer rollout/eval horizon (`HORIZON = 96`) -- paths are longer on 5x5, and
  on 5x5 breaking through a wall *is* often optimal (the detour overruns the
  horizon), so GMG is natural here even at the default reward;
* more training (5x5 has a larger state space): see `NUM_TRAIN_STEPS` /
  `NUM_ENVS` below, tuned by the 5x5 HPO;
* the deployment wall probes come from `evalsuite.wall_envs(5)`.

Run cell-by-cell (`# %%`) in a notebook frontend, or top-to-bottom as a script.
"""

# %%

import functools

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from evalsuite import wall_envs
from evaluation import compute_return
from generate import generate
from potteryshop import Action, collect_rollout, tree_map
from rewards import reward2, DISCOUNT_RATE
# master renamed the grouped/safe solver entry point to `compute_optimal_return`.
from solver import compute_optimal_return as compute_optimal_return_grouped
from train import default_device, train_agent_multienv

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

world_size = 5
shard_mean = 2.0
urn_mean = 1.7
HORIZON = 96  # rollout / eval horizon (longer than 4x4's 64)

MODEL_PATH = "agent_5x5.pt"
LOAD_AGENT = False

WANDB_PROJECT = None
# WANDB_PROJECT = "arena8-capstone"

# 5x5 training config (from the 5x5 HPO; longer than 4x4). NUM_TRAIN_STEPS is set
# for a long-ish run -- shrink for a quick smoke test.
NUM_ENVS = 8192
MINIBATCH = 16384
LR = 0.003
NUM_ENV_STEPS = HORIZON
NUM_TRAIN_STEPS = 4000

gen = functools.partial(
    generate, world_size=world_size, shard_mean=shard_mean, urn_mean=urn_mean,
)

# %%
# Visualise a few sample layouts (needs a notebook frontend)

sample_envs = gen(num_envs=16, generator=torch.Generator().manual_seed(0))
visualise.display_envs(sample_envs, grid_width=4, title="5x5 training layouts")

# %%
# Build the agent

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
# Train (or load a saved agent)

if LOAD_AGENT:
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net = net.to(device)
    history = None
    print(f"loaded agent from {MODEL_PATH} (skipped training)")
else:
    net, history = train_agent_multienv(
        gen=gen,
        net=net,
        reward_fn=reward2,
        num_train_steps=NUM_TRAIN_STEPS,
        num_envs=NUM_ENVS,
        num_env_steps=NUM_ENV_STEPS,
        num_epochs=1,
        minibatch_size=MINIBATCH,
        lr=LR,
        device=device,
        seed=1,
        wandb_project=WANDB_PROJECT,
    )
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"saved agent to {MODEL_PATH}")

# %%
# Plot training curves

if history is not None:
    xs = [m["step"] for m in history]
    series = [
        ([m["return"] for m in history], "mean discounted return"),
        ([m["loss"] for m in history], "PPO loss"),
        ([m["regret"] for m in history], "mean oracle regret"),
        ([m["entropy"] for m in history], "policy entropy"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    for ax, (ys, ylabel) in zip(axes, series):
        ax.plot(xs, ys, lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[-1].set_xlabel("train step")
    fig.tight_layout()
    plt.show()

# %%
# Grid of rollouts + per-level regret

grid_width = 6
watch_envs = gen(num_envs=grid_width**2, generator=torch.Generator().manual_seed(3))
rollout = collect_rollout(
    env=watch_envs,
    policy_fn=net.policy,
    num_steps=HORIZON,
    generator=torch.Generator().manual_seed(3),
    device=device,
    deterministic=False,
)
optimal = compute_optimal_return_grouped(watch_envs, horizon=HORIZON)
flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
rewards = reward2(flat.state, flat.action, flat.next_state).view(grid_width**2, HORIZON)
achieved = compute_return(rewards, discount_rate=DISCOUNT_RATE).cpu()
regret = optimal - achieved
print(f"mean optimal {optimal.mean():+.3f}  achieved {achieved.mean():+.3f}  "
      f"regret {regret.mean():+.3f}")
visualise.display_rollouts(watch_envs, rollout, grid_width=grid_width)

# %%
# Deployment urn-wall probes (breaking through is often optimal on 5x5)

walls = wall_envs(5)
rollout = collect_rollout(
    env=walls.to(device),
    policy_fn=net.policy,
    num_steps=HORIZON,
    generator=torch.Generator().manual_seed(0),
    deterministic=False,
)
optimal = compute_optimal_return_grouped(walls, horizon=HORIZON)
flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
rewards = reward2(flat.state, flat.action, flat.next_state).view(walls.num_envs, HORIZON)
achieved = compute_return(rewards, discount_rate=DISCOUNT_RATE).cpu()
for i in range(walls.num_envs):
    print(f"wall {i}: optimal {optimal[i]:+.3f}  achieved {achieved[i]:+.3f}  "
          f"regret {(optimal[i] - achieved[i]):+.3f}")
visualise.display_rollouts(walls, rollout, grid_width=walls.num_envs)
# %%
