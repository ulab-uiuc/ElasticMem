#!/bin/bash
set -e
BASE_MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
echo "Memory cache target dir : ./cache/chunk_embeddings_lme_mc10"
echo "Base model              : ${BASE_MODEL}"
echo "Note: 500 distinct haystacks → 500 distinct cache key entries."
echo "Will be populated by the first train/eval invocation."
