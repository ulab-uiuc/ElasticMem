#!/bin/bash
# Layer-2: turn each Layer-1 trajectory in the M pool (~80% of train)
# into a 200-300 word procedural "skill card" via an LLM.
# Output: data/alfworld_skill_cards.json (~4 MB, ~2,832 cards).
#
# Uses any LLM API of your choice. Concurrency via --workers.
set -e
PY=${PY:-python}

$PY -m alfworld_il.summarize \
    --offline-data ./data/alfworld_train_offline_v2.json \
    --output       ./data/alfworld_skill_cards.json \
    --memory-ratio 0.8 \
    --workers      8 \
    --resume
