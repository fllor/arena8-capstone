"""
Interactive driver for the DR (domain randomisation) BASELINE in the pottery
shop -- the control for the PLR-perp comparison. Structured as `# %%` cells like
`run.py` / `run_plr.py`.

Identical to `run_plr.py` in every respect -- same layout distribution, network,
hyperparameters, eval harness, and rollout viz -- EXCEPT the curriculum: this
trains with plain `train.train_agent_multienv` (every fresh batch is trained on,
no buffer, no regret prioritisation). So a side-by-side of the two agents
isolates the effect of the PLR curriculum and nothing else.

Budget note: PLR-perp takes gradient steps on only its REPLAY steps (~half, at
replay_prob=0.5); explore steps are scoring-only. The paper compares methods at
an EQUIVALENT NUMBER OF GRADIENT UPDATES (Jiang et al. 2021, "half as many
gradient updates"). So this baseline trains for `num_train_steps` =
round(plr_steps * replay_prob) by default. Flip BUDGET to "total" to instead
match the total env-generation budget (same step count as PLR).
"""

# %%

import functools
import sys

import matplotlib.pyplot as plt
import torch

import visualise
from agent import ActorCriticNetwork
from generate import generate
from potteryshop import Action, collect_rollout
from rewards import reward2
from train import default_device, train_agent_multienv

# the shared eval harness lives in overnight_run1/
sys.path.insert(0, "overnight_run1")
import evalsuite  # noqa: E402

device = default_device()
print(f"using device: {device}")

# %%
# Configuration -- MUST match run_plr.py so only the curriculum differs.

world_size = 4
if world_size == 4:
    shard_mean, urn_mean = 1.7, 1.3
elif world_size == 5:
    shard_mean, urn_mean = 2.0, 1.7

# The PLR run these settings mirror (run_plr.py defaults).
PLR_STEPS = 400
PLR_REPLAY_PROB = 0.5

# "gradient" -> match PLR's gradient-update count (fair policy-training budget,
# the paper's comparison). "total" -> match PLR's total step/env-generation count.
BUDGET = "gradient"
if BUDGET == "gradient":
    num_train_steps = round(PLR_STEPS * PLR_REPLAY_PROB)  # ~200
elif BUDGET == "total":
    num_train_steps = PLR_STEPS                           # 400

MODEL_PATH = "agent_baseline.pt"
LOAD_AGENT = False  # set True to skip training and load MODEL_PATH

# The PLR agent to compare against in the eval cell (from run_plr.py).
PLR_PT = "our_first_plr_agent.pt"

WANDB_PROJECT = "arena8-capstone"  # None to disable W&B logging

gen = functools.partial(
    generate, world_size=world_size, shard_mean=shard_mean, urn_mean=urn_mean
)

# %%
# Visualise a few sample layouts from the distribution (needs a notebook frontend)

sample_envs = gen(num_envs=16, generator=torch.Generator().manual_seed(0))
visualise.display_envs(sample_envs, grid_width=4, title="DR baseline training layouts")

# %%
# Build the agent (same architecture + init seed as run_plr.py, so both runs
# start from the *same* random weights -- a clean control).

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
# Train with plain DR (or load a saved agent instead). Same PPO hyperparameters
# as the PLR run; the only difference is no buffer / no prioritised replay.

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
        num_train_steps=num_train_steps,
        num_envs=2048,
        num_epochs=1,
        minibatch_size=16384,
        lr=0.003,
        entropy_coeff=0.01,
        device=device,
        seed=1,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=f"dr-baseline-{BUDGET}",
    )
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"saved agent to {MODEL_PATH} ({num_train_steps} steps, BUDGET={BUDGET})")

# %%
# Plot training curves (DR logs every `regret_every`-th step, so history is
# sparse; each entry records its true `step` index). Same four panels as run.py.

if history:
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
# Evaluate: oracle regret + break-through rate on held-out RANDOM and WALL sets.
# Headline GMG metric is the WALL break-rate. Side-by-side with the PLR agent.

eval_sets = evalsuite.build_eval_sets(
    world_size, shard_mean=shard_mean, urn_mean=urn_mean, n_random=2000
)
print("DR baseline:")
for res in evalsuite.evaluate_all(net, eval_sets).values():
    print("  ", res)

try:
    plr_net = ActorCriticNetwork.init(
        obs_height=world_size, obs_width=world_size, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )
    plr_net.load_state_dict(torch.load(PLR_PT, map_location=device))
    plr_net = plr_net.to(device)
    print(f"\nPLR agent ({PLR_PT}):")
    for res in evalsuite.evaluate_all(plr_net, eval_sets).values():
        print("  ", res)
except FileNotFoundError:
    print(f"\n(PLR agent {PLR_PT} not found; skipping side-by-side)")

# %%
# Watch the DR agent on the hand-built urn-walls: it should WALK AROUND (GMG).

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
