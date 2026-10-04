#!/bin/bash
# Layer-0: install alfworld + textworld and download the official ALFWorld
# game files (~2.3 GB) into $ALFWORLD_DATA. After this you have
# json_2.1.1/{train,valid_seen,valid_unseen}/.
set -e
cd "$(dirname "$0")/../.."

# 1. Install
pip install --upgrade alfworld textworld

# 2. Set the data dir (default if unset)
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
mkdir -p "$ALFWORLD_DATA"
echo "ALFWORLD_DATA=$ALFWORLD_DATA"

# 3. Download the official games (idempotent)
alfworld-download

# 4. Sanity check
python - <<'PY'
import os
from alfworld.info import ALFWORLD_DATA
print("ALFWORLD_DATA:", ALFWORLD_DATA)
print("contents:", sorted(os.listdir(ALFWORLD_DATA)))
PY
