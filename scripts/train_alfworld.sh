#!/bin/bash
# ElasticMem on ALFWorld (embodied control; memory = procedural skill cards).
# Run build_skill_cards.sh first.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u -m tasks.alfworld.train \
    --offline-data ./data/alfworld_train_offline.json \
    --skill-cards  ./data/alfworld_skill_cards.json \
    --model_name   "$MODEL" \
    --top_z        10 \
    --history_steps 5 \
    --save_dir     ./checkpoints/alfworld
