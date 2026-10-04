#!/bin/bash
# Materialise the random 80/20 split (seed=42) into plain text id files
# under data/splits/. Run once after download_data.sh.
set -e
PY=${PY:-python}
$PY - <<'PY'
import json, os, random
from pathlib import Path
from huggingface_hub import snapshot_download

local = snapshot_download(repo_id="Percena/lme-mc10", repo_type="dataset",
                          allow_patterns=["data/lme_s_mc10.json"])
src = os.path.join(local, "data/lme_s_mc10.json")
items = []
with open(src) as f:
    for line in f:
        line = line.strip()
        if line:
            items.append(json.loads(line))
qids = sorted(it["question_id"] for it in items)
print(f"{len(qids)} questions in LongMemEval-MC10")

rng = random.Random(42)
shuffled = qids[:]
rng.shuffle(shuffled)
n = len(shuffled)
n_test = max(1, int(n * 0.2))
test  = sorted(shuffled[:n_test])
train = sorted(shuffled[n_test:])
print(f"train={len(train)}  test={len(test)}")

out = Path(__file__).resolve().parent.parent / "data" / "splits"
out.mkdir(parents=True, exist_ok=True)
(out / "train_ids.txt").write_text("\n".join(train) + "\n")
(out / "test_ids.txt").write_text("\n".join(test) + "\n")
print(f"wrote {out}")
PY
