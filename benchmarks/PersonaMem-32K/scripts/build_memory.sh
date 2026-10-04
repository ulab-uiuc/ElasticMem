#!/bin/bash
# Build the chunk-embedding cache for PersonaMem-32K.
# Run once per (base_model, max_chunk_length, max_hidden_cache) combination.
# Subsequent train/eval runs auto-reuse the cache.
#
# Usage:  bash scripts/build_memory.sh [base_model]
set -e
BASE_MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}

# Implementation note for downstream users:
#   our reference pipeline runs encode_chunks() lazily on the first
#   train.py / eval.py call, which writes to the cache_dir below. If
#   you prefer to pre-warm, run a tiny "encode-only" pass with that
#   helper. Refer to data_utils.encode_chunks(...).
echo "Memory cache target dir : ./cache/chunk_embeddings_personamem_32k"
echo "Base model              : ${BASE_MODEL}"
echo "Will be populated by the first train/eval invocation."
