"""
Unified training+eval driver for the pottery-shop GMG experiments.

One entry point for every training run in this project (long baseline run,
reward-tuning sweeps, 5x5 repeat). It:

  * optionally overrides the tunable reward parameters (kept in lock-step with
    the oracle solver via `rewards.set_reward_params`),
  * trains with the tuned PPO config (overridable),
  * periodically evaluates oracle regret on a fixed held-out set (random
    layouts + hand-built urn walls) and logs it -- to stdout, to W&B if
    requested, and into the returned history,
  * runs a larger final evaluation,
  * saves the network plus a sidecar JSON with the full run config + final
    metrics.

Examples:
  # ~1h baseline long run on 4x4, logged to W&B, saved:
  python experiment.py --name long-4x4 --steps 6000 --wandb --save agent_long_4x4.pt

  # a reward-tuning point (lower break penalty, higher step cost):
  python experiment.py --name rtA-bp1.0-sc0.05 --steps 1200 \
      --break-penalty 1.0 --step-cost 0.05 --save agent_rtA.pt
"""

from __future__ import annotations

import argparse
import functools
import json
import time

import torch

import rewards
from agent import ActorCriticNetwork
from evalsuite import build_eval_sets, evaluate_all
from generate import generate
from potteryshop import Action
from rewards import reward2
from train import default_device, train_agent_multienv


def build_net(world_size: int) -> ActorCriticNetwork:
    return ActorCriticNetwork.init(
        obs_height=world_size, obs_width=world_size, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )


def make_eval_fn(eval_sets, horizon):
    """Closure logged into training history: mean rare-env regret + break-rate."""
    def eval_fn(net, step):
        res = evaluate_all(net, eval_sets, horizon=horizon)
        out = {}
        for name, r in res.items():
            out[f"{name}_regret"] = r.mean_regret
            out[f"{name}_solved"] = r.frac_solved
            out[f"{name}_break"] = r.break_rate
        out["walls_max_regret"] = res["walls"].max_regret
        # progress line so a long unattended run is observable in its log
        rr = res["random"]
        print(f"EVAL step {step:>5}  rand_regret {rr.mean_regret:.4f}  "
              f"rand_solved {rr.frac_solved:.3f}  rand_max {rr.max_regret:.2f}  "
              f"walls_regret {res['walls'].mean_regret:.3f}  "
              f"walls_break {res['walls'].break_rate:.2f}", flush=True)
        return out
    return eval_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name (W&B + sidecar)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--save", default=None, help="path to save the trained net")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--shard-mean", type=float, default=None)
    p.add_argument("--urn-mean", type=float, default=None)
    # PPO config (defaults = the tuned 4x4 config)
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--num-env-steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1)
    # reward knobs (None -> keep rewards.py default)
    p.add_argument("--break-penalty", type=float, default=None)
    p.add_argument("--shaping-coeff", type=float, default=None)
    p.add_argument("--step-cost", type=float, default=None)
    p.add_argument("--waste-penalty", type=float, default=None)
    # eval / logging
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--n-random", type=int, default=512)
    p.add_argument("--n-random-final", type=int, default=2000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="arena8-capstone")
    args = p.parse_args()

    # default shard/urn means by grid size (match run.py)
    if args.shard_mean is None:
        args.shard_mean = 1.7 if args.world_size == 4 else 2.0
    if args.urn_mean is None:
        args.urn_mean = 1.3 if args.world_size == 4 else 1.7

    # apply reward overrides BEFORE building eval/solver so everything is in sync
    rewards.set_reward_params(
        break_penalty=args.break_penalty, shaping_coeff=args.shaping_coeff,
        step_cost=args.step_cost, waste_penalty=args.waste_penalty,
    )
    rparams = rewards.reward_params()

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
        name=args.name, steps=args.steps, world_size=args.world_size,
        shard_mean=args.shard_mean, urn_mean=args.urn_mean,
        num_envs=args.num_envs, minibatch=args.minibatch, num_epochs=args.num_epochs,
        num_env_steps=args.num_env_steps, lr=args.lr, entropy_coeff=args.entropy_coeff,
        seed=args.seed, reward_params=rparams,
    )
    print("CONFIG " + json.dumps(cfg), flush=True)

    net = build_net(args.world_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    net, hist = train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=args.steps,
        num_envs=args.num_envs, num_env_steps=args.num_env_steps,
        num_epochs=args.num_epochs, minibatch_size=args.minibatch,
        entropy_coeff=args.entropy_coeff, lr=args.lr, device=device, seed=args.seed,
        regret_every=max(10, args.eval_every), regret_frac=0.05,
        eval_fn=eval_fn, eval_every=args.eval_every,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.name,
        progress=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    # larger final eval for the headline numbers
    final_sets = build_eval_sets(
        args.world_size, args.shard_mean, args.urn_mean, n_random=args.n_random_final
    )
    final = evaluate_all(net, final_sets, horizon=args.num_env_steps)
    print(f"\n=== {args.name}  (wall {wall/60:.1f} min, {args.steps} steps) ===", flush=True)
    print(f"reward params: {rparams}", flush=True)
    for r in final.values():
        print("  " + str(r), flush=True)

    summary = dict(cfg=cfg, wall_min=round(wall / 60, 2), final={
        name: dict(mean_regret=r.mean_regret, max_regret=r.max_regret,
                   frac_solved=r.frac_solved, break_rate=r.break_rate,
                   mean_optimal=r.mean_optimal, mean_achieved=r.mean_achieved, n=r.n)
        for name, r in final.items()
    })
    if args.save:
        torch.save(net.state_dict(), args.save)
        sidecar = args.save.rsplit(".", 1)[0] + ".json"
        with open(sidecar, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsaved net -> {args.save}  config+metrics -> {sidecar}", flush=True)
    with open("experiment_summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print("SUMMARY " + json.dumps(summary["final"]), flush=True)


if __name__ == "__main__":
    main()
