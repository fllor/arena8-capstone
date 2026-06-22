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

from agent import ActorCriticNetwork
from evaluation import evaluate_behaviour
from generate import generate
from potteryshop import Action
from rewards import reward2, reward_break
from train import default_device, train_agent_multienv

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

world_size = 4
num_shards = 4
num_urns = 2

# the fixed-bin layout distribution, ready to pass to `train_agent_multienv`
gen = functools.partial(
    generate,
    world_size=world_size,
    num_shards=num_shards,
    num_urns=num_urns,
)

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
# Train across many environments at once

net, history = train_agent_multienv(
    gen=gen,
    net=net,
    reward_fn=reward2,
    num_train_steps=70,  # ~42s on the HPO box; reaches return ~3.63
    num_envs=8192,
    num_epochs=1,
    minibatch_size=16384,
    lr=0.003,
    device=device,
    seed=1,
)

# %%
# Plot the training return

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
