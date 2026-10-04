#!/bin/bash
# Layer-1: build the expert-trajectory replay JSON by running AlfredExpert(PLANNER)
# on every official ALFWorld game in each split (stores admissible_commands per step).
#
# The replay collector itself comes from MemSkill (Apache-2.0):
#   https://github.com/ViktorAxelsen/MemSkill  ->  alfworld_replay.py
# Clone it next to this repo, or set REPLAY to wherever you put the script.
#
# Time: ~25 min for train (3,553 games), ~3 min each for valid_seen / unseen.
set -e
PY=${PY:-python}
REPLAY=${REPLAY:-../MemSkill/alfworld_replay.py}
OUT_DIR=${OUT_DIR:-./data}

if [ ! -f "$REPLAY" ]; then
    echo "alfworld_replay.py not found at $REPLAY"
    echo "  git clone https://github.com/ViktorAxelsen/MemSkill ../MemSkill"
    exit 1
fi

$PY "$REPLAY" --split train --batch-size 32 \
    --output "$OUT_DIR/alfworld_train_offline.json"
$PY "$REPLAY" --split eval_in_distribution --batch-size 16 \
    --output "$OUT_DIR/alfworld_expert_eval_in_distribution.json"
$PY "$REPLAY" --split eval_out_of_distribution --batch-size 16 \
    --output "$OUT_DIR/alfworld_expert_eval_out_of_distribution.json"
