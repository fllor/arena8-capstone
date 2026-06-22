"""
Interactive driver for multi-environment training in the pottery shop.

Structured as `# %%` cells (run cell-by-cell in VS Code / Jupyter, or
top-to-bottom as a plain script). All the heavy lifting lives in the sibling
modules; this file just wires them together and inspects the results.

Mirrors the notebook's `net3` configuration: train on the fixed-bin
distribution with the intended reward (`reward2`).
"""

# %%

import functools

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from evaluation import compute_return, evaluate_behaviour
from generate import generate
from potteryshop import Action, Environment, collect_rollout, tree_map
from rewards import reward2, reward_break, DISCOUNT_RATE
from solver import compute_optimal_return
from train import default_device, train_agent_multienv

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

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

# Where to save/load the trained network. Set LOAD_AGENT = True to skip training
# and load weights from MODEL_PATH instead (the architecture below must match the
# one the file was saved from).
MODEL_PATH = "agent.pt"
LOAD_AGENT = False

# Optional Weights & Biases logging. Set WANDB_PROJECT to a project name to log
# the scored-step metrics (return/loss/regret) there; leave it None to disable.
WANDB_PROJECT = "arena8-capstone"

# the fixed-bin layout distribution, ready to pass to `train_agent_multienv`
gen = functools.partial(
    generate,
    world_size=world_size,
    shard_mean=shard_mean,
    urn_mean=urn_mean,
)

# %%
# Visualise a few sample layouts from the distribution (needs a notebook
# frontend: VS Code interactive / Jupyter)

sample_envs = gen(num_envs=16, generator=torch.Generator().manual_seed(0))
visualise.display_envs(sample_envs, grid_width=4, title="fixed-bin training layouts")

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
# Train across many environments at once (or load a saved agent instead).

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
        num_train_steps=300,  # ~3min
        num_envs=8192,
        num_epochs=1,
        minibatch_size=16384,  # 32 gradient updates per collection
        lr=0.003,  # large batch permits greater learning rate
        device=device,
        seed=1,
        wandb_project=WANDB_PROJECT,
    )
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"saved agent to {MODEL_PATH}")

# %%
# Plot the training return, loss, and regret (skipped when an agent was loaded)

if history is not None:
    # metrics are logged every `regret_every`-th step, so history is sparse;
    # each entry records its true `step` index to plot against.
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
# Visualise grid of rollouts

watch_steps = 96
grid_width = 6
watch_envs = gen(num_envs=grid_width**2, generator=torch.Generator().manual_seed(3))
rollout = collect_rollout(
    env=watch_envs,
    policy_fn=net.policy,
    num_steps=watch_steps,
    generator=torch.Generator().manual_seed(3),
    device=device,
    deterministic=False
)

# Optimal return (oracle) vs the return actually achieved in the rollout above.
# The achieved return is scored on the *same* trajectory being animated, so the
# numbers match what you see; the gap is the per-level regret.
optimal = compute_optimal_return(watch_envs, horizon=watch_steps)
flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
rewards = reward2(flat.state, flat.action, flat.next_state).view(grid_width**2, watch_steps)
achieved = compute_return(rewards, discount_rate=DISCOUNT_RATE).cpu()
regret = optimal - achieved

print(f"{'env (row,col)':>14}  {'optimal':>8}  {'achieved':>8}  {'regret':>8}")
for b in range(grid_width**2):
    print(
        f"{f'{b:>2} ({b // grid_width},{b % grid_width})':>14}"
        f"  {optimal[b].item():>+8.3f}  {achieved[b].item():>+8.3f}  {regret[b].item():>+8.3f}"
    )
print(
    f"{'mean':>14}  {optimal.mean().item():>+8.3f}"
    f"  {achieved.mean().item():>+8.3f}  {regret.mean().item():>+8.3f}"
)

visualise.display_rollouts(watch_envs, rollout, grid_width=grid_width)

# %%
# Test walking around wall
for env_layout in [
    (
        (0, 2, 1, 1),
        (0, 2, 1, 1),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ),
    (
        (0, 2, 1, 1),
        (0, 2, 1, 1),
        (0, 2, 0, 0),
        (0, 0, 0, 0),
    ),
    (
        (0, 2, 1, 1),
        (0, 2, 1, 1),
        (0, 2, 2, 0),
        (0, 0, 0, 0),
    )
]:
    probe_steps = 96
    env_probe = Environment(
        init_robot_pos=torch.tensor((0, 0), dtype=torch.long),
        init_items_map=torch.tensor(env_layout, dtype=torch.long),
        bin_pos=torch.tensor((0, 0), dtype=torch.long),
    )
    rollout = collect_rollout(
        env=env_probe.to(device),
        policy_fn=net.policy,
        num_steps=probe_steps,
        generator=torch.Generator().manual_seed(0),
        deterministic=False
    )

    probe_batch = Environment(
        init_robot_pos=env_probe.init_robot_pos[None],
        init_items_map=env_probe.init_items_map[None],
        bin_pos=env_probe.bin_pos[None],
    )
    optimal = compute_optimal_return(probe_batch, horizon=probe_steps)
    flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
    rewards = reward2(flat.state, flat.action, flat.next_state).view(1, probe_steps)
    achieved = compute_return(rewards, discount_rate=DISCOUNT_RATE).cpu()
    print(
        f"optimal {optimal.item():+.3f}  achieved {achieved.item():+.3f}"
        f"  regret {(optimal - achieved).item():+.3f}"
    )

    visualise.display_rollout(env_probe, rollout)
# %%
