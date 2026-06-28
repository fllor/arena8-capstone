#!/usr/bin/env bash
# Render walls + random GIFs for every agent .pt in a models folder.
#
#   ./run_all_gifs.sh                 # defaults to models/, gif_out/
#   ./run_all_gifs.sh models_legacy   # explicit folder
#   ./run_all_gifs.sh models out_dir
#
# make_gifs.py tags each GIF by the model's basename + env, so nothing clobbers.
# NOTE: make_gifs.py assumes 4x4 agents (WORLD_SIZE=4); 5x5 checkpoints will fail.
set -euo pipefail

DIR="${1:-models}"
OUT="${2:-gif_out}"

shopt -s nullglob
models=("$DIR"/*.pt)
if [ ${#models[@]} -eq 0 ]; then
    echo "no .pt files found in $DIR" >&2
    exit 1
fi

echo "rendering ${#models[@]} model(s) from $DIR -> $OUT/"
for pt in "${models[@]}"; do
    for envs in walls random; do
        echo ">> $(basename "$pt")  [$envs]"
        python make_gifs.py "$pt" --envs "$envs" --out-dir "$OUT"
    done
done
echo "done -> $OUT/"
