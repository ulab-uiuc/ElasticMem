#!/bin/bash
# ElasticMem on PersonaMem-32K / PersonaMem-128K.
#   bash train_personamem.sh 32k      (or 128k)
set -e
PY=${PY:-python}
SCALE=${1:-32k}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}      # or Qwen/Qwen2.5-7B-Instruct

if [ "$SCALE" = "128k" ]; then
    QUESTIONS=./data/questions_128k.csv
    CONTEXTS=./data/shared_contexts_128k.jsonl
    TOP_Z=21
else
    QUESTIONS=./data/questions_32k.csv
    CONTEXTS=./data/shared_contexts_32k.jsonl
    TOP_Z=9
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u train.py \
    --question_path   "$QUESTIONS" \
    --context_path    "$CONTEXTS" \
    --model_name      "$MODEL" \
    --num_epochs      10 \
    --num_generations 4 \
    --top_z           "$TOP_Z" \
    --lr              2e-5 \
    --temperature     1.0 \
    --seed            42 \
    --save_dir        "./checkpoints/personamem_${SCALE}"
