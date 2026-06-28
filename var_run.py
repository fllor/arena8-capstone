"""Headless PLR variance run for the pottery-shop GMG project (remote machine).

Runs a fresh, non-normalised plr50 -- 10000 plr steps = 5000 DR-equivalent
gradient updates -- at a given seed and logs to W&B. Used to characterise PLR's
run-to-run variance (the same-seed CUDA-nondeterminism divergence seen locally:
one run solves the walls, another barely breaks). Pure training path -- no
visualise/notebook deps -- so it runs anywhere torch+jaxtyping are installed.

Both the network init and the training/sampling seed are set from <seed>, so each
run is an independent sample (CUDA nondeterminism adds further spread on top).

Usage: python var_run.py <seed> [num_grad_updates=5000]
"""

import functools
import sys

import torch

from agent import ActorCriticNetwork
from evalsuite import build_eval_sets
from generate import generate
from potteryshop import Action
from rewards import reward2
from train import UEDConfig, default_device, train_agent

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NUM_GRAD_UPDATES = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
RUN = f"plr_p50_seed{SEED}"

device = default_device()
print(f"=== {RUN}: device={device}, {NUM_GRAD_UPDATES} grad updates ===", flush=True)

world_size, shard_mean, urn_mean = 5, 1.7, 1.3
gen = functools.partial(
    generate, world_size=world_size, shard_mean=shard_mean, urn_mean=urn_mean
)
net = ActorCriticNetwork.init(
    obs_height=world_size, obs_width=world_size, net_channels=16, net_width=64,
    num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
    generator=torch.Generator().manual_seed(SEED),
)
eval_sets = build_eval_sets(world_size, shard_mean, urn_mean)
num_train_steps = NUM_GRAD_UPDATES * 2  # plr50: replay_prob=0.5 -> 2x steps per grad update

config = UEDConfig(
    gen=gen, net=net, reward_fn=reward2,
    num_train_steps=num_train_steps + 1,  # +1 to log final metrics
    num_envs=4096, num_env_steps=64, num_epochs=1, num_minibatches=32,
    buffer_capacity=32768, lr=0.003, entropy_coeff=0.01, step_offset=0,
    device=device, seed=SEED, eval_sets=eval_sets,
    replay_prob=0.5, train_on_generate=False, normalise_score=False,
    checkpoint_path=f"agent_{RUN}.pt", checkpoint_every=100,
    buffer_save_path=f"buffer_{RUN}.pt", history_save_path=f"history_{RUN}.pt",
    solver_cache_dir="solver_cache",
    wandb_project="arena8-capstone-5x5", wandb_run_name=RUN,
)
net, history, sampler = train_agent(config)
torch.save(net.state_dict(), f"agent_{RUN}.pt")
if sampler is not None:
    sampler.save(f"buffer_{RUN}.pt")
print(f"=== done {RUN} ===", flush=True)
