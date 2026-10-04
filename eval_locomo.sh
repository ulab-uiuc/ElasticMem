#!/bin/bash
# Evaluate / transfer-evaluate a checkpoint on LoCoMo-MC10.
set -e
PY=${PY:-python}
CKPT=${1:-./checkpoints/locomo_mc10/best}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u -m locomo_mc10.eval_transfer \
    --load_checkpoint "$CKPT" \
    --model_name      "$MODEL" \
    --top_z           20 \
    --max_new_tokens  5 \
    --cache_dir       ./cache/chunk_embeddings_locomo_mc10
