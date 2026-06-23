"""
Headless overnight PLR-perp training driver (fresh init).

Mirrors overnight_run1/experiment.py (the DR driver): trains, periodically evals
oracle regret + break-rate on a held-out random + urn-wall set, checkpoints, runs
a larger final eval, and saves the net plus a `cfg`+`final` sidecar JSON so
`eval_agent.py` can load it later. The trainer is `plr.train_agent_plr`.

Designed to run unattended in the background. Example (logs to a file, survives
hangup):

    nohup python train_plr_overnight.py --steps 2500 --wandb \
        --name plr-fresh-2500 --save agent_plr_fresh_2500.pt \
        > plr_fresh_2500.log 2>&1 &

Refuses to overwrite an existing --save target. Checkpoints to
<save>_ckpt.pt every --checkpoint-every steps, so a crash loses at most that many.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import torch

import rewards
from agent import ActorCriticNetwork
from generate import generate
from potteryshop import Action
from rewards import reward2
from plr import PLRConfig, train_agent_plr
from train import default_device

# shared eval harness lives in overnight_run1/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "overnight_run1"))
from evalsuite import build_eval_sets, evaluate_all  # noqa: E402


def build_net(world_size: int, seed: int) -> ActorCriticNetwork:
    return ActorCriticNetwork.init(
        obs_height=world_size, obs_width=world_size, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(seed),
    )


def make_eval_fn(eval_sets, horizon):
    """Logged into training: held-out random + wall regret/break-rate."""
    def eval_fn(net, step):
        res = evaluate_all(net, eval_sets, horizon=horizon)
        out = {}
        for name, r in res.items():
            out[f"{name}_regret"] = r.mean_regret
            out[f"{name}_solved"] = r.frac_solved
            out[f"{name}_break"] = r.break_rate
        out["walls_max_regret"] = res["walls"].max_regret
        rr = res["random"]
        print(f"EVAL step {step:>5}  rand_regret {rr.mean_regret:.4f}  "
              f"rand_solved {rr.frac_solved:.3f}  "
              f"walls_regret {res['walls'].mean_regret:.3f}  "
              f"walls_break {res['walls'].break_rate:.2f}", flush=True)
        return out
    return eval_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="plr-fresh-2500", help="run name (W&B + sidecar)")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--save", default="agent_plr_fresh_2500.pt")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--shard-mean", type=float, default=None)
    p.add_argument("--urn-mean", type=float, default=None)
    # PPO / PLR config (defaults match run_plr.py)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--num-env-steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--replay-prob", type=float, default=0.5)
    p.add_argument("--buffer-capacity", type=int, default=4096)
    p.add_argument("--beta", type=float, default=0.3, help="rank prioritisation temperature")
    p.add_argument("--rho", type=float, default=0.1, help="staleness coefficient")
    # eval / checkpoint / logging
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--n-random", type=int, default=512)
    p.add_argument("--n-random-final", type=int, default=2000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="arena8-capstone")
    args = p.parse_args()

    # never clobber an existing checkpoint
    assert not os.path.exists(args.save), (
        f"{args.save} already exists; pass a different --save so we don't overwrite it"
    )
    ckpt_path = args.save.rsplit(".", 1)[0] + "_ckpt.pt"

    if args.shard_mean is None:
        args.shard_mean = 1.7 if args.world_size == 4 else 2.0
    if args.urn_mean is None:
        args.urn_mean = 1.3 if args.world_size == 4 else 1.7

    rparams = rewards.reward_params()  # use rewards.py defaults (matches the baseline)
    device = default_device()
    gen = functools.partial(
        generate, world_size=args.world_size,
        shard_mean=args.shard_mean, urn_mean=args.urn_mean,
    )
    eval_sets = build_eval_sets(
        args.world_size, args.shard_mean, args.urn_mean, n_random=args.n_random
    )
    eval_fn = make_eval_fn(eval_sets, horizon=args.num_env_steps)

    cfg = dict(
        name=args.name, algo="PLR-perp", steps=args.steps, world_size=args.world_size,
        shard_mean=args.shard_mean, urn_mean=args.urn_mean,
        num_envs=args.num_envs, minibatch=args.minibatch, num_epochs=args.num_epochs,
        num_env_steps=args.num_env_steps, lr=args.lr, entropy_coeff=args.entropy_coeff,
        seed=args.seed, replay_prob=args.replay_prob, buffer_capacity=args.buffer_capacity,
        prioritisation_beta=args.beta, staleness_coeff=args.rho, reward_params=rparams,
    )
    print("CONFIG " + json.dumps(cfg), flush=True)
    print(f"device={device}  save->{args.save}  ckpt->{ckpt_path}", flush=True)

    net = build_net(args.world_size, args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    # master's PLR takes a single PLRConfig. `--beta` (rank prioritisation
    # temperature) maps to `temperature`; the oracle optimum is cached in the
    # LevelSampler so replay steps never re-solve (the speedup over the old loop).
    config = PLRConfig(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=args.steps,
        num_envs=args.num_envs, num_env_steps=args.num_env_steps,
        num_epochs=args.num_epochs, minibatch_size=args.minibatch,
        entropy_coeff=args.entropy_coeff, lr=args.lr, device=device, seed=args.seed,
        replay_prob=args.replay_prob, buffer_capacity=args.buffer_capacity,
        temperature=args.beta, staleness_coeff=args.rho,
        eval_fn=eval_fn, eval_every=args.eval_every,
        checkpoint_path=ckpt_path, checkpoint_every=args.checkpoint_every,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.name, progress=False,
    )
    net, history, sampler = train_agent_plr(config)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    # larger final eval for the headline numbers
    final_sets = build_eval_sets(
        args.world_size, args.shard_mean, args.urn_mean, n_random=args.n_random_final
    )
    final = evaluate_all(net, final_sets, horizon=args.num_env_steps)
    print(f"\n=== {args.name}  (wall {wall/60:.1f} min, {args.steps} steps) ===", flush=True)
    for r in final.values():
        print("  " + str(r), flush=True)

    # buffer mean regret over filled slots (empty slots hold -inf in LevelSampler)
    filled_scores = sampler.scores[torch.isfinite(sampler.scores)]
    buffer_mean_regret = filled_scores.mean().item() if filled_scores.numel() else 0.0
    summary = dict(cfg=cfg, wall_min=round(wall / 60, 2),
                   buffer_mean_regret=round(buffer_mean_regret, 4), final={
        name: dict(mean_regret=r.mean_regret, max_regret=r.max_regret,
                   frac_solved=r.frac_solved, break_rate=r.break_rate,
                   mean_optimal=r.mean_optimal, mean_achieved=r.mean_achieved, n=r.n)
        for name, r in final.items()
    })
    torch.save(net.state_dict(), args.save)
    with open(args.save.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved net -> {args.save}  config+metrics -> "
          f"{args.save.rsplit('.', 1)[0]}.json", flush=True)
    print("SUMMARY " + json.dumps(summary["final"]), flush=True)


if __name__ == "__main__":
    main()
