#!/bin/bash
# ElasticMem on LoCoMo-MC10 (10-way multiple choice over long dialogues).
set -e
PY=${PY:-python}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u -m locomo_mc10.train \
    --model_name      "$MODEL" \
    --num_epochs      10 \
    --lr              2e-5 \
    --top_z           20 \
    --num_generations 4 \
    --max_new_tokens  5 \
    --eval_every      500 \
    --save_dir        ./checkpoints/locomo_mc10 \
    --cache_dir       ./cache/chunk_embeddings_locomo_mc10
