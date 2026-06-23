"""
HPO / training driver scored on oracle REGRET (not just mean return).

Unlike `hpo.py` (which maximised mean held-out return, a metric that saturates
near the optimum for any healthy config), this scores each config on the project
metric: oracle regret on a fixed held-out set of random layouts (with its tail)
plus the hand-built urn-wall probes. Use it to confirm the tuned training
hyperparameters are on the frontier for *rare-env* competence, and as the shared
training+eval driver for the reward-tuning experiments.

Usage:
    python hpo_regret.py STEPS envs:epochs:mb:lr [envs:epochs:mb:lr ...]
e.g.
    python hpo_regret.py 190 8192:1:16384:0.003 4096:1:8192:0.003
"""

from __future__ import annotations

import functools
import json
import sys
import time

import torch

from agent import ActorCriticNetwork
from evalsuite import build_eval_sets, evaluate_all
from generate import generate
from potteryshop import Action
from rewards import reward2
from train import default_device, train_agent_multienv

RESULTS_FILE = "hpo_regret_results.jsonl"
WORLD_SIZE = 4
SHARD_MEAN, URN_MEAN = 1.7, 1.3

device = default_device()
gen = functools.partial(generate, world_size=WORLD_SIZE, shard_mean=SHARD_MEAN, urn_mean=URN_MEAN)
eval_sets = build_eval_sets(WORLD_SIZE, SHARD_MEAN, URN_MEAN, n_random=512)


def build_net() -> ActorCriticNetwork:
    return ActorCriticNetwork.init(
        obs_height=WORLD_SIZE, obs_width=WORLD_SIZE, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )


def run(num_envs: int, num_epochs: int, minibatch_size: int, lr: float, steps: int) -> dict:
    net = build_net()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    net, hist = train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=steps,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=max(10, steps // 10),
        regret_frac=0.0, progress=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    res = evaluate_all(net, eval_sets)
    return dict(
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, steps=steps, wall=round(wall, 1),
        rand_regret=round(res["random"].mean_regret, 3),
        rand_max_regret=round(res["random"].max_regret, 3),
        rand_solved=round(res["random"].frac_solved, 3),
        wall_regret=round(res["walls"].mean_regret, 3),
        wall_break=round(res["walls"].break_rate, 3),
    )


def main(argv: list[str]) -> None:
    steps = int(argv[0])
    specs = []
    for a in argv[1:]:
        parts = a.split(":")
        ne, ep, mb = int(parts[0]), int(parts[1]), int(parts[2])
        lr = float(parts[3]) if len(parts) > 3 else 0.003
        specs.append((ne, ep, mb, lr))
    for (ne, ep, mb, lr) in specs:
        try:
            res = run(ne, ep, mb, lr, steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res = dict(num_envs=ne, num_epochs=ep, minibatch_size=mb, lr=lr, steps=steps, error="OOM")
        except Exception as exc:  # noqa: BLE001
            res = dict(num_envs=ne, num_epochs=ep, minibatch_size=mb, lr=lr, steps=steps, error=str(exc)[:160])
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(res) + "\n")
        print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
