#!/bin/bash
# Eval-only on all 500 LongMemEval-MC10 questions (also used for transfer eval).
set -e
PY=${PY:-python}
CKPT=${1:-./checkpoints/lme_mc10/best}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u -m lme_mc10.train \
    --model_name      "$MODEL" \
    --split_mode      eval_only \
    --load_checkpoint "$CKPT" \
    --top_z           20 \
    --max_new_tokens  5 \
    --cache_dir       ./cache/chunk_embeddings_lme_mc10
