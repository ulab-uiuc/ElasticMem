#!/bin/bash
# ALFWorld evaluation in the live TextWorld environment.
#   bash eval_alfworld.sh seen   ./checkpoints/alfworld/best
#   bash eval_alfworld.sh unseen ./checkpoints/alfworld/best
set -e
PY=${PY:-python}
SPLIT=${1:-seen}
CKPT=${2:-./checkpoints/alfworld/best}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

if [ "$SPLIT" = "unseen" ]; then
    EVAL_DATA=./data/alfworld_expert_eval_out_of_distribution.json
else
    EVAL_DATA=./data/alfworld_expert_eval_in_distribution.json
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u -m alfworld_il.eval_env \
    --eval-data   "$EVAL_DATA" \
    --skill-cards ./alfworld_il/data/alfworld_skill_cards.json \
    --checkpoint  "$CKPT" \
    --model_name  "$MODEL" \
    --top_z       10 \
    --max_steps   30 \
    --per_game_timeout 240 \
    --history_steps    5 \
    --max_new_tokens   16 \
    --output      "./results/alfworld_${SPLIT}.json"
