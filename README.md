# Goal misgeneralisation in a pottery shop, and regret-based UED

For more details see the report on [LessWrong](https://www.lesswrong.com/posts/eNPLWgFrzZLvidBcc/).

Code for the blog post on goal misgeneralisation (GMG) in a small gridworld. A robot
cleans up shards but is penalised for smashing urns. Trained on random layouts (DR), it
learns to never break urns and fails on deployment levels where an urn wall blocks the
bin. We compare three ways of training it:

- **DR**: domain randomisation (plain PPO on random levels)
- **PLR⊥**: prioritised level replay, keyed on exact (oracle) regret
- **ACCEL**: PLR⊥ plus an editor that mutates high-regret levels, so it can build urn walls

Based on Abdel Sadek et al., [*Mitigating Goal Misgeneralization via Minimax Regret*](https://arxiv.org/abs/2507.03068).

## Contents

| Path | What it is |
|---|---|
| `potteryshop.py` | Batched environment |
| `agent.py`, `ppo.py` | Actor-critic network and PPO |
| `rewards.py` | Reward function and discount |
| `generate.py` | Random level generator (the training distribution) |
| `solver.py`, `solver_cache.py` | Exact optimal return, used to compute oracle regret |
| `level_sampler.py`, `editor.py` | PLR⊥ level buffer and ACCEL editor |
| `train.py` | One training loop for DR / PLR⊥ / ACCEL, selected by `UEDConfig` |
| `evalsuite.py`, `evaluation.py`, `utils.py` | Held-out random and urn-wall evaluation sets, regret metrics |
| `run.py` | Interactive driver (`# %%` cells): train, plot, evaluate, watch rollouts |
| `make_gifs.py`, `visualise.py` | Render rollouts as GIFs |
| `data/` | 4x4 checkpoints behind the main plot |
| `data5/` | 5x5 checkpoints (DR, PLR⊥, ACCEL) with their eval CSVs |
| `plot/` | Regret curves (CSV) and gnuplot script for the main figure |
| `gif_out/` | Rollout GIFs |
| `SESSION.md` | Summary of the 5x5 results |

## Usage

Install `torch`, `jaxtyping`, `tqdm`, `einops`, `pillow` and `ipywidgets` (`wandb` is
optional). Run everything from the repo root.

Train: open `run.py` in VS Code or Jupyter and run it cell by cell. Pick the method
(`dr`, `plr50`, `accel_walk`, ...) and grid size at the top of the file.

Make GIFs from a checkpoint:

```
python make_gifs.py data/agent_dr_5k.pt --envs walls
python make_gifs.py data5/agent_dr_5000.pt --envs random --world-size 5
```

Output goes to `gif_out/`.
