#!/bin/bash
# Evaluate a trained ElasticMem checkpoint on the PersonaMem test split.
#   bash eval_personamem.sh 32k ./checkpoints/personamem_32k/best
set -e
PY=${PY:-python}
SCALE=${1:-32k}
CKPT=${2:-"./checkpoints/personamem_${SCALE}/best"}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

if [ "$SCALE" = "128k" ]; then
    QUESTIONS=./data/questions_128k.csv; CONTEXTS=./data/shared_contexts_128k.jsonl; TOP_Z=21
else
    QUESTIONS=./data/questions_32k.csv;  CONTEXTS=./data/shared_contexts_32k.jsonl;  TOP_Z=9
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u eval.py \
    --checkpoint_dir "$CKPT" \
    --model_name     "$MODEL" \
    --question_path  "$QUESTIONS" \
    --context_path   "$CONTEXTS" \
    --top_z          "$TOP_Z" \
    --split          test \
    --output_path    "./results/personamem_${SCALE}.json"
