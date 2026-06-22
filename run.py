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
from potteryshop import Action, collect_rollout, tree_map
from rewards import reward2, reward_break
from solver import compute_optimal_return
from train import default_device, train_agent_multienv

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

world_size = 4
num_shards = 4
num_urns = 2

# Where to save/load the trained network. Set LOAD_AGENT = True to skip training
# and load weights from MODEL_PATH instead (the architecture below must match the
# one the file was saved from).
MODEL_PATH = "agent.pt"
LOAD_AGENT = False

# the fixed-bin layout distribution, ready to pass to `train_agent_multienv`
gen = functools.partial(
    generate,
    world_size=world_size,
    num_shards=num_shards,
    num_urns=num_urns,
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
        num_train_steps=90,  # ~55s on the HPO box; reaches return ~3.63
        num_envs=8192,
        num_epochs=1,
        minibatch_size=16384,  # 32 gradient updates per collection (the sweet spot)
        lr=0.003,  # large batch permits ~3x the old 0.001 default
        device=device,
        seed=1,
    )
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"saved agent to {MODEL_PATH}")

# %%
# Plot the training return (skipped when an agent was loaded from file)

if history is not None:
    returns = [m["return"] for m in history]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(returns, lw=0.8)
    ax.set_xlabel("train step")
    ax.set_ylabel("mean discounted return")
    ax.set_title("training return")
    fig.tight_layout()
    plt.show()

# %%
# Evaluate behaviour on a fresh batch of layouts from the same distribution.
# `reward2` measures task success; `reward_break` probes urn-smashing.

num_eval = 1000
eval_envs = gen(num_envs=num_eval, generator=torch.Generator().manual_seed(0))

probes = {"reward2": reward2, "reward_break": reward_break}
fig, axes = plt.subplots(len(probes), figsize=(5, 3 * len(probes)))
for (name, reward_fn), ax in zip(probes.items(), axes):
    scores = evaluate_behaviour(
        env=eval_envs,
        net=net,
        reward_fn=reward_fn,
        num_rollouts=num_eval,
        generator=torch.Generator().manual_seed(0),
    )
    print(f"{name:>12}: mean return {scores.mean().item():+.3f}")
    ax.hist(scores.cpu().numpy(), bins=40)
    ax.set_title(name)
    ax.set_xlabel("return")
fig.tight_layout()
plt.show()

# %%
# Watch the trained agent act — the qualitative check the histograms can't give
# you (does it walk around urns, or break through them?). Animates a grid of
# rollouts; needs a notebook frontend.

watch_envs = gen(num_envs=16, generator=torch.Generator().manual_seed(3))
watch_steps = 64
grid_width = 4
rollout = collect_rollout(
    env=watch_envs,
    policy_fn=net.policy,
    num_steps=watch_steps,
    generator=torch.Generator().manual_seed(3),
    device=device,
)

# Optimal return (oracle) vs the return actually achieved in the rollout above.
# The achieved return is scored on the *same* trajectory being animated, so the
# numbers match what you see; the gap is the per-level regret.
optimal = compute_optimal_return(watch_envs, horizon=watch_steps)
flat = tree_map(lambda x: x.flatten(0, 1), rollout.transitions)
rewards = reward2(flat.state, flat.action, flat.next_state).view(16, watch_steps)
achieved = compute_return(rewards, discount_rate=0.995).cpu()
regret = optimal - achieved

print(f"{'env (row,col)':>14}  {'optimal':>8}  {'achieved':>8}  {'regret':>8}")
for b in range(16):
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
