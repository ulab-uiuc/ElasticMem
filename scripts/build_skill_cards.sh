#!/bin/bash
# ALFWorld Stage 0: turn offline expert trajectories into procedural skill cards
# (the memory corpus). Requires GEMINI_API_KEY (or edit llm_judge._MODEL_NAME to
# point at another provider).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY -u -m tasks.alfworld.summarize \
    --offline-data ./data/alfworld_train_offline.json \
    --output       ./data/alfworld_skill_cards.json \
    --memory-ratio 0.8 \
    --workers      8 \
    --resume
