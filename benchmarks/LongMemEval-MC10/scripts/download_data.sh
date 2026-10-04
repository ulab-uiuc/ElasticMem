#!/bin/bash
set -e
PY=${PY:-python}
$PY - <<'PY'
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="Percena/lme-mc10",
                          repo_type="dataset",
                          allow_patterns=["data/lme_s_mc10.json"])
print(f"LongMemEval-MC10 ready at: {local}")
PY
