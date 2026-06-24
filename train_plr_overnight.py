"""
Headless overnight PLR-perp training driver (fresh init).

Mirrors overnight_run1/experiment.py (the DR driver): trains, periodically evals
oracle regret + break-rate on a held-out random + urn-wall set, checkpoints, runs
a larger final eval, and saves the net plus a `cfg`+`final` sidecar JSON so
`eval_agent.py` can load it later. The trainer is `train.train_agent` (PLR config).

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
import time

import torch

import rewards
from agent import ActorCriticNetwork
from evalsuite import build_eval_sets
from generate import generate
from potteryshop import Action
from rewards import DISCOUNT_RATE, reward2
from train import UEDConfig, compute_eval_metrics, default_device, train_agent


def build_net(world_size: int, seed: int) -> ActorCriticNetwork:
    return ActorCriticNetwork.init(
        obs_height=world_size, obs_width=world_size, net_channels=16, net_width=64,
        num_conv_layers=5, num_dense_layers=2, num_actions=len(Action),
        generator=torch.Generator().manual_seed(seed),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="plr-fresh-2500", help="run name (W&B + sidecar)")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--save", default="agent_plr_fresh_2500.pt")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--shard-mean", type=float, default=None)
    p.add_argument("--urn-mean", type=float, default=None)
    # PPO / PLR config (defaults match run.py)
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
    # The unified trainer takes a single UEDConfig; PLR-perp is replay_prob>0 with
    # train_on_generate=False (the default). `--beta` (rank prioritisation
    # temperature) maps to `temperature`; the oracle optimum is cached in the
    # LevelSampler so replay steps never re-solve (the speedup over the old loop).
    config = UEDConfig(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=args.steps,
        num_envs=args.num_envs, num_env_steps=args.num_env_steps,
        num_epochs=args.num_epochs, minibatch_size=args.minibatch,
        entropy_coeff=args.entropy_coeff, lr=args.lr, device=device, seed=args.seed,
        replay_prob=args.replay_prob, buffer_capacity=args.buffer_capacity,
        temperature=args.beta, staleness_coeff=args.rho,
        eval_sets=eval_sets, eval_every=args.eval_every,
        checkpoint_path=ckpt_path, checkpoint_every=args.checkpoint_every,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.name, progress=False,
    )
    net, history, sampler = train_agent(config)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    # larger final eval for the headline numbers (same metric code as in-loop)
    final_sets = build_eval_sets(
        args.world_size, args.shard_mean, args.urn_mean, n_random=args.n_random_final
    )
    final = compute_eval_metrics(
        net, final_sets, reward_fn=reward2, discount_rate=DISCOUNT_RATE,
        horizon=args.num_env_steps, device=device,
    )
    print(f"\n=== {args.name}  (wall {wall/60:.1f} min, {args.steps} steps) ===", flush=True)
    for name in final_sets:
        print(f"  {name:>8} (greedy): regret {final[f'eval/{name}/greedy/regret']:+.3f} "
              f"(max {final[f'eval/{name}/greedy/max_regret']:+.3f})  "
              f"optimal {final[f'eval/{name}/optimal']:+.3f}  "
              f"achieved {final[f'eval/{name}/greedy/achieved']:+.3f}  "
              f"break-rate {final[f'eval/{name}/greedy/break']:.2f}  "
              f"solved {final[f'eval/{name}/greedy/solved']:.2f}", flush=True)

    # buffer mean regret over filled slots (empty slots hold -inf in LevelSampler)
    filled_scores = sampler.scores[torch.isfinite(sampler.scores)]
    buffer_mean_regret = filled_scores.mean().item() if filled_scores.numel() else 0.0
    summary = dict(cfg=cfg, wall_min=round(wall / 60, 2),
                   buffer_mean_regret=round(buffer_mean_regret, 4), final=final)
    torch.save(net.state_dict(), args.save)
    with open(args.save.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved net -> {args.save}  config+metrics -> "
          f"{args.save.rsplit('.', 1)[0]}.json", flush=True)
    print("SUMMARY " + json.dumps(summary["final"]), flush=True)


if __name__ == "__main__":
    main()
