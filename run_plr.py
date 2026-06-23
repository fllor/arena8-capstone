"""
Interactive driver for PLR-perp (robust Prioritised Level Replay) training in
the pottery shop -- the Day-2 UED method, structured as `# %%` cells like
`run.py` (run cell-by-cell in VS Code / Jupyter, or top-to-bottom as a script).

Same shape as `run.py` (build -> show layouts -> train -> plot -> evaluate ->
watch rollouts), but the trainer is `plr.train_agent_plr`: a regret-keyed level
buffer feeds PPO replay updates while fresh random levels only curate the buffer
(the stop-gradient "perp" trick). Extra cells visualise *what the buffer learned
to keep* (the high-regret urn-walls) and evaluate break-through vs the DR
baseline on the held-out wall set.

See `plr.py` for the algorithm and `doc/2110.02439.txt` (Jiang et al.) Algorithm 1.
"""

# %%

import functools
import os
import sys

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from generate import generate
from potteryshop import Action, Environment, collect_rollout
from rewards import reward2
from plr import train_agent_plr
from train import default_device

# the shared eval harness lives in overnight_run1/
sys.path.insert(0, "overnight_run1")
import evalsuite  # noqa: E402

device = default_device()
print(f"using device: {device}")

# %%
# Configuration

# Layout distribution -- identical to run.py / the DR baseline, so the only
# difference in the comparison is the *curriculum*, not the env distribution.
world_size = 4
if world_size == 4:
    shard_mean, urn_mean = 1.7, 1.3
elif world_size == 5:
    shard_mean, urn_mean = 2.0, 1.7

# PLR-perp hyperparameters (paper-grounded defaults; see plr.py docstrings).
#   replay_prob       Bernoulli p of Algorithm 1: fraction of steps that REPLAY
#                     (train on the buffer). The rest EXPLORE (score + curate, no
#                     gradient). 0.5 is the paper's headline PLR-perp setting.
#   prioritisation_beta  rank-prioritisation temperature (lower = peakier toward
#                        the highest-regret levels). Paper searches {0.1, 0.3}.
#   staleness_coeff   rho: weight on the staleness term so old scores get
#                     refreshed. Paper searches {0.3, 0.7}; 0.1 is gentle.
replay_prob = 0.5
prioritisation_beta = 0.3
staleness_coeff = 0.1
buffer_capacity = 4096

# Start from scratch, or CONTINUE from the GMG baseline to ask "can PLR *repair*
# an already-misgeneralising agent?". Point at an overnight baseline .pt (its
# arch must match the net built below) or set to None to train fresh.
# Fine-tune the converged DR/GMG baseline: PLR only has to shift WALL behaviour,
# since in-distribution competence is already there (random regret ~0.008). Far
# cheaper than fresh-to-convergence, and isolates the PLR curriculum's effect.
CONTINUE_FROM = "overnight_run1/agent_long_4x4.pt"  # None to train from scratch

MODEL_PATH = "agent_plr_ft_long4x4.pt"  # fresh name -- guarded below, won't overwrite
LOAD_AGENT = False  # set True to skip training and load MODEL_PATH

WANDB_PROJECT = "arena8-capstone"  # None to disable W&B logging

gen = functools.partial(
    generate, world_size=world_size, shard_mean=shard_mean, urn_mean=urn_mean
)

# %%
# Visualise a few sample layouts from the base distribution (needs a notebook
# frontend). The buffer will curate the rare high-regret tail out of these.

sample_envs = gen(num_envs=16, generator=torch.Generator().manual_seed(0))
visualise.display_envs(sample_envs, grid_width=4, title="base (DR) training layouts")

# %%
# Build the agent (same architecture as run.py / the overnight baselines)

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
if CONTINUE_FROM is not None:
    net.load_state_dict(torch.load(CONTINUE_FROM, map_location=device))
    print(f"continuing from {CONTINUE_FROM}")
net = net.to(device)

# %%
# Train with PLR-perp (or load a saved agent instead).
#
# NB: PLR oracle-scores EVERY batch (explore = admission, replay = refresh), so
# it is heavier per step than the DR trainer's 10%-every-10-steps. Keep num_envs
# modest (a few thousand). buffer/history are returned for the cells below.

if LOAD_AGENT:
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net = net.to(device)
    history, buffer = None, None
    print(f"loaded agent from {MODEL_PATH} (skipped training)")
else:
    # fail fast on a name collision -- never clobber an existing checkpoint.
    assert not os.path.exists(MODEL_PATH), (
        f"{MODEL_PATH} already exists; change MODEL_PATH so training won't overwrite it"
    )
    net, history, buffer = train_agent_plr(
        gen=gen,
        net=net,
        reward_fn=reward2,
        num_train_steps=400,
        num_envs=2048,
        num_epochs=1,
        minibatch_size=16384,
        lr=0.003,
        replay_prob=replay_prob,
        buffer_capacity=buffer_capacity,
        prioritisation_beta=prioritisation_beta,
        staleness_coeff=staleness_coeff,
        device=device,
        seed=1,
        wandb_project=WANDB_PROJECT,
        wandb_run_name="plr-ft-long4x4",
    )
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"saved agent to {MODEL_PATH}")

# %%
# Plot training curves (history holds one entry per REPLAY step; each records its
# true `step` index). Bottom two panels are PLR-specific buffer diagnostics.

if history:
    xs = [m["step"] for m in history]
    series = [
        ([m["return"] for m in history], "mean discounted return"),
        ([m["loss"] for m in history], "PPO loss"),
        ([m["regret"] for m in history], "mean replay regret"),
        ([m["entropy"] for m in history], "policy entropy"),
        ([m["buffer_size"] for m in history], "buffer size"),
        ([m["buffer_mean_score"] for m in history], "buffer mean regret"),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(7, 12), sharex=True)
    for ax, (ys, ylabel) in zip(axes, series):
        ax.plot(xs, ys, lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[-1].set_xlabel("train step")
    fig.tight_layout()
    plt.show()

# %%
# THE MONEY SHOT: what did the buffer learn to KEEP? Show the highest-regret
# levels it is holding -- these should be the urn-wall / urn-blocking layouts
# that DR almost never trains on.

if buffer is not None and len(buffer) > 0:
    n_show = min(36, len(buffer))
    order = buffer.scores[: len(buffer)].argsort(descending=True)
    top = order[:n_show]
    top_envs = Environment(
        init_robot_pos=buffer.robot[top],
        init_items_map=buffer.items[top],
        bin_pos=buffer.bin[top],
    )
    top_scores = buffer.scores[top]
    print(f"buffer top-{n_show} regret: "
          f"{top_scores.max():.3f} (max) .. {top_scores.min():.3f} (min of shown)")
    visualise.display_envs(
        top_envs, grid_width=6,
        title=f"highest-regret levels held in the PLR buffer (of {len(buffer)})",
    )

# %%
# Evaluate: oracle regret + break-through rate on held-out RANDOM and WALL sets,
# using the shared harness. The headline GMG metric is the WALL break-rate.

eval_sets = evalsuite.build_eval_sets(
    world_size, shard_mean=shard_mean, urn_mean=urn_mean, n_random=2000
)
print("PLR agent:")
for res in evalsuite.evaluate_all(net, eval_sets).values():
    print("  ", res)

# Optional side-by-side with the DR/GMG baseline (if its weights are present).
BASELINE_PT = "overnight_run1/agent_long_4x4.pt"
try:
    base = ActorCriticNetwork.init(
        obs_height=world_size, obs_width=world_size, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )
    base.load_state_dict(torch.load(BASELINE_PT, map_location=device))
    base = base.to(device)
    print("\nDR baseline (agent_long_4x4):")
    for res in evalsuite.evaluate_all(base, eval_sets).values():
        print("  ", res)
except FileNotFoundError:
    print(f"\n(baseline {BASELINE_PT} not found; skipping side-by-side)")

# %%
# Watch the PLR agent on the hand-built urn-walls: does it break THROUGH now,
# or still walk around (GMG)? Compare frame-by-frame with the baseline filmstrip
# from `eval_agent.py`.

wall_envs = evalsuite.wall_envs(world_size)
wall_rollout = collect_rollout(
    env=wall_envs.to(device),
    policy_fn=net.policy,
    num_steps=96,
    generator=torch.Generator().manual_seed(0),
    device=device,
    deterministic=False,
)
visualise.display_rollouts(wall_envs, wall_rollout, grid_width=wall_envs.num_envs)

# %%
# Watch a grid of rollouts on fresh random layouts (in-distribution behaviour).

grid_width = 6
watch_envs = gen(num_envs=grid_width**2, generator=torch.Generator().manual_seed(3))
rollout = collect_rollout(
    env=watch_envs,
    policy_fn=net.policy,
    num_steps=96,
    generator=torch.Generator().manual_seed(3),
    device=device,
    deterministic=False,
)
visualise.display_rollouts(watch_envs, rollout, grid_width=grid_width)
# %%
