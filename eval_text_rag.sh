#!/bin/bash
# Text-space RAG baseline: feed the top-K retrieved chunks as raw text.
set -e
PY=${PY:-python}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -u -m baseline.eval_text_rag \
    --model_name    "$MODEL" \
    --question_path ./data/questions_32k.csv \
    --context_path  ./data/shared_contexts_32k.jsonl \
    --top_k         9 \
    --output_path   ./results/text_rag_personamem_32k.json
