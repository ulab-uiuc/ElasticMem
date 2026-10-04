#!/bin/bash
# ALFWorld memory artifacts come in 2 layers:
#   1. expert-trajectory replay JSON v2 — produced by scripts/build_replay.sh
#      (already shipped under data/ in this repo)
#   2. LLM-generated skill cards         — produced by scripts/build_skill_cards.sh
#      (NOT shipped, you must run it once)
#
# The chunk-embedding cache for the skill cards is built lazily on
# the first train/eval invocation into:
#   ./cache/chunk_embeddings_alfworld_il/{hash}_last1.pt
#   ./cache/chunk_embeddings_alfworld_il/{hash}_lastN.pt
set -e

if [ ! -f ./data/alfworld_skill_cards.json ]; then
  echo "Skill cards file missing — run scripts/build_skill_cards.sh first."
  exit 1
fi

echo "All artifacts present:"
echo "  - Layer-1 train replay : ./data/alfworld_train_offline_v2.json"
echo "  - Layer-1 valid_seen   : ./data/alfworld_expert_eval_in_distribution_v2.json"
echo "  - Layer-1 valid_unseen : ./data/alfworld_expert_eval_out_of_distribution_v2.json"
echo "  - Layer-2 skill cards  : ./data/alfworld_skill_cards.json"
echo "Skill-card embedding cache will populate on first train/eval call."
