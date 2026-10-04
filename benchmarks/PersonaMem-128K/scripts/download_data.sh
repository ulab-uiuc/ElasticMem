#!/bin/bash
# Alternate fetch path for the 128K context jsonl when not committed.
# Replace the URL below with whichever mirror you prefer (HF dataset
# repo, internal storage, etc).
set -e
DEST=./data
mkdir -p "$DEST"

# Example using HuggingFace Hub (uncomment and fill in repo_id):
# python - <<PY
# from huggingface_hub import snapshot_download
# local = snapshot_download(repo_id="<owner>/personamem-128k", repo_type="dataset",
#                           local_dir="$DEST")
# print(f"downloaded to {local}")
# PY

echo "Edit this script with your actual download source."
echo "Expected files in $DEST/:"
echo "  - questions_128k.csv (5.1 MB)"
echo "  - shared_contexts_128k.jsonl (71 MB)"
