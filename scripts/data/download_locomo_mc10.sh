#!/bin/bash
# Pull LoCoMo-MC10 from HuggingFace into the local HF cache. The
# loader (tasks.locomo_mc10.data_loader) uses
# snapshot_download under the hood, so this is also a no-op if you
# just run train.sh / eval.sh directly.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY - <<'PY'
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="Percena/locomo-mc10",
                          repo_type="dataset",
                          allow_patterns=["data/locomo_mc10.json"])
print(f"LoCoMo-MC10 ready at: {local}")
PY
