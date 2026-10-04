#!/bin/bash
set -e
BASE_MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
echo "Memory cache target dir : ./cache/chunk_embeddings_personamem_128k"
echo "Base model              : ${BASE_MODEL}"
echo "Will be populated by the first train/eval invocation."
