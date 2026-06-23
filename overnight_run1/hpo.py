"""
Time-budgeted hyperparameter search for `train_agent_multienv`.

For each (num_envs, num_epochs, minibatch_size) config we:
  1. calibrate per-step wall time on a few warmup steps,
  2. set num_train_steps to fill the ~budget-second training window,
  3. train a fresh net (fixed seed) and report smoothed training return plus a
     deterministic held-out eval return (1000 fixed fresh layouts).

Objective: maximise eval return within ~2 min of training; tie-break on time.
Configs come from argv as "envs:epochs:mb" triples. Results are appended to
hpo_results.jsonl (one JSON object per line) and printed as "RESULT {...}".

Usage: python hpo.py [budget_s] envs:epochs:mb [envs:epochs:mb ...]
"""

from __future__ import annotations

import functools
import json
import sys
import time

import torch

from agent import ActorCriticNetwork
from evaluation import evaluate_behaviour
from generate import generate
from potteryshop import Action
from rewards import reward2
from train import default_device, train_agent_multienv

RESULTS_FILE = "hpo_results.jsonl"

device = default_device()
gen = functools.partial(generate, world_size=4, shard_mean=1.7, urn_mean=1.3)


def build_net() -> ActorCriticNetwork:
    return ActorCriticNetwork.init(
        obs_height=4,
        obs_width=4,
        net_channels=16,
        net_width=64,
        num_conv_layers=5,
        num_dense_layers=2,
        num_actions=len(Action),
        generator=torch.Generator().manual_seed(1),
    )


# fixed held-out eval set (same layouts every trial -> comparable metric)
EVAL_N = 1000
eval_envs = gen(num_envs=EVAL_N, generator=torch.Generator().manual_seed(0))


def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def trial(num_envs: int, num_epochs: int, minibatch_size: int, budget_s: float,
          lr: float = 0.001) -> dict:
    # 1. calibrate per-step time. Run untimed warmup steps first (the first
    #    step pays CUDA init/allocator/compile costs that would otherwise
    #    inflate the estimate and make us under-fill the budget), then time.
    warmup, calib = 3, 8
    net = build_net()
    train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=warmup,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=10, regret_frac=0.0, progress=False,
    )
    _sync()
    t0 = time.time()
    train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=calib,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=10, regret_frac=0.0, progress=False,
    )
    _sync()
    per_step = (time.time() - t0) / calib
    num_train_steps = max(1, round(budget_s / per_step))

    # 2. full timed run on a fresh net
    net = build_net()
    _sync()
    t0 = time.time()
    net, hist = train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=num_train_steps,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=10, regret_frac=0.0, progress=False,
    )
    _sync()
    wall = time.time() - t0

    # 3. metrics
    tail = max(1, len(hist) // 20)
    train_ret = sum(h["return"] for h in hist[-tail:]) / tail
    eval_ret = evaluate_behaviour(
        env=eval_envs, net=net, reward_fn=reward2, num_rollouts=EVAL_N,
        generator=torch.Generator().manual_seed(0),
    ).mean().item()

    return dict(
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, num_train_steps=num_train_steps, wall=round(wall, 1),
        per_step_ms=round(per_step * 1000, 1),
        train_ret=round(train_ret, 3), eval_ret=round(eval_ret, 3),
    )


def time_to_target(
    num_envs: int, num_epochs: int, minibatch_size: int,
    target: float, max_s: float, smooth: int = 5, lr: float = 0.001,
) -> dict:
    """
    Train until the smoothed training return first reaches `target`, and report
    the wall-clock time to get there. Caps at `max_s` of training; if the
    target is never reached, reports reached=False with the best smoothed value.
    Trained in chunks so we can checkpoint the elapsed time at each step without
    per-step Python overhead dominating.
    """
    # estimate per-step time (warmup then time) to size the step cap
    warmup, calib = 3, 8
    net = build_net()
    train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=warmup,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=10, regret_frac=0.0, progress=False,
    )
    _sync(); t0 = time.time()
    train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=calib,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=10, regret_frac=0.0, progress=False,
    )
    _sync()
    per_step = (time.time() - t0) / calib
    max_steps = max(1, round(max_s / per_step))

    net = build_net()
    _sync(); t0 = time.time()
    net, hist = train_agent_multienv(
        gen=gen, net=net, reward_fn=reward2, num_train_steps=max_steps,
        num_envs=num_envs, num_epochs=num_epochs, minibatch_size=minibatch_size,
        lr=lr, device=device, seed=1, regret_every=10, regret_frac=0.0, progress=False,
    )
    _sync()
    wall = time.time() - t0
    rets = [h["return"] for h in hist]
    sm = [sum(rets[max(0, i - smooth + 1):i + 1]) / min(i + 1, smooth)
          for i in range(len(rets))]
    hit = next((i for i, v in enumerate(sm) if v >= target), None)
    per_step_actual = wall / max_steps
    return dict(
        mode="tt", num_envs=num_envs, num_epochs=num_epochs,
        minibatch_size=minibatch_size, lr=lr, target=target,
        reached=hit is not None,
        time_to_target=round((hit + 1) * per_step_actual, 1) if hit is not None else None,
        step_to_target=(hit + 1) if hit is not None else None,
        best_smoothed=round(max(sm), 3), steps_run=max_steps, wall=round(wall, 1),
        per_step_ms=round(per_step_actual * 1000, 1),
    )


def main(argv: list[str]) -> None:
    # phase-2 mode: "tt:<target>:<max_s>" switches to time-to-target trials
    budget_s = 110.0
    tt_mode = None
    specs = []
    for a in argv:
        if a.startswith("tt:"):
            _, tgt, mx = a.split(":")
            tt_mode = (float(tgt), float(mx))
            continue
        if ":" not in a:
            budget_s = float(a)
            continue
        parts = a.split(":")
        e, ep, mb = parts[:3]
        lr = float(parts[3]) if len(parts) > 3 else 0.001
        specs.append((int(e), int(ep), int(mb), lr))
    for (ne, ep, mb, lr) in specs:
        try:
            if tt_mode is not None:
                res = time_to_target(ne, ep, mb, tt_mode[0], tt_mode[1], lr=lr)
            else:
                res = trial(ne, ep, mb, budget_s, lr=lr)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res = dict(num_envs=ne, num_epochs=ep, minibatch_size=mb, lr=lr, error="OOM")
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            res = dict(num_envs=ne, num_epochs=ep, minibatch_size=mb, lr=lr, error=str(exc)[:120])
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(res) + "\n")
        print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
