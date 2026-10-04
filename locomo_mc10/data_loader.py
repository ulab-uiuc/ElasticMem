"""
LoCoMo-MC10 loader (Percena/locomo-mc10 on HuggingFace).

1,986 questions across 10 LoCoMo conversations, each reformulated into
a 10-choice MC question (answers letters 'a'..'j').

Split follows MemSkill's LoCoMo convention (6/2/2 by sorted conv index):
  train: first 6 conversations  (~1,157 QA)
  val  : middle 2 conversations (~429 QA)  — unused here by convention
  test : last 2 conversations   (~400 QA)

Sample format (matches the rest of the codebase):
    {
      row_data: {
        question_id, user_question_or_message, correct_answer (letter form like "(e)"),
        all_options (serialized "(a) ..\\n(b) ..\\n..(j) .."),
        topic (question_type), question_type
      },
      context: [],
      chunks: List[str],                    # one per user+assistant turn in haystack
      shared_context_id: str                # "conv-XX"
    }
"""
import os
import json
from typing import Dict, List, Tuple

# HF repo local cache path (set via HF_HOME or default ~/.cache/huggingface)
_HF_REPO = "Percena/locomo-mc10"
_REL_PATH = "data/locomo_mc10.json"


def _find_local_file(cache_dir: str = None) -> str:
    """Locate the downloaded locomo_mc10.json inside HF cache. Downloads if missing."""
    # Use HF API to snapshot download if not present
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        raise RuntimeError("huggingface_hub not installed") from None
    local_root = snapshot_download(
        repo_id=_HF_REPO, repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=[_REL_PATH],
    )
    path = os.path.join(local_root, _REL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found after snapshot_download")
    return path


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sessions_to_chunks(sessions: List[List[Dict]]) -> List[str]:
    """Flatten haystack_sessions (list of sessions, each a list of turns) into
    user+assistant turn-pair chunks — same format as locomo/data_loader.py."""
    chunks: List[str] = []
    for sess in sessions:
        if not sess:
            continue
        # Normalize each turn
        norm = []
        for t in sess:
            role = (t.get("role") or "unknown").strip().lower()
            text = (t.get("content") or "").strip()
            if not text or role == "system":
                continue
            norm.append({"role": role, "content": text})
        # Pair user → assistant
        i = 0
        while i < len(norm):
            cur = norm[i]
            if cur["role"] == "user":
                if i + 1 < len(norm) and norm[i + 1]["role"] == "assistant":
                    chunks.append(f"User:\n{cur['content']}\n\nAssistant:\n{norm[i + 1]['content']}")
                    i += 2
                    continue
                else:
                    chunks.append(f"User:\n{cur['content']}")
                    i += 1
            elif cur["role"] == "assistant":
                chunks.append(f"Assistant:\n{cur['content']}")
                i += 1
            else:
                i += 1
    return chunks


def _letter(i: int) -> str:
    """0 → 'a', 1 → 'b', ..., 9 → 'j'."""
    return chr(ord("a") + i)


def _item_to_sample(item: Dict, chunks: List[str]) -> Dict:
    conv_id = item["question_id"].rsplit("_q", 1)[0]
    idx = int(item["correct_choice_index"])
    choices = item["choices"]
    options_text = "\n".join(f"({_letter(i)}) {c}" for i, c in enumerate(choices))
    correct_letter_paren = f"({_letter(idx)})"
    return {
        "row_data": {
            "question_id": item["question_id"],
            "user_question_or_message": item["question"],
            "correct_answer": correct_letter_paren,
            "all_options": options_text,
            "topic": item.get("question_type", "unknown"),
            "question_type": item.get("question_type", "unknown"),
        },
        "context": [],
        "chunks": chunks,
        "shared_context_id": conv_id,
    }


def load_locomo_mc10(cache_dir: str = None
                    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Returns (train, val, test). Split by conv (6/2/2 sorted by conv-id)."""
    path = _find_local_file(cache_dir=cache_dir)
    items = _read_jsonl(path)

    # Group by conversation, keep one copy of haystack per conv (identical across Qs)
    by_conv: Dict[str, List[Dict]] = {}
    haystack_by_conv: Dict[str, List[List[Dict]]] = {}
    for it in items:
        cid = it["question_id"].rsplit("_q", 1)[0]
        by_conv.setdefault(cid, []).append(it)
        if cid not in haystack_by_conv:
            haystack_by_conv[cid] = it["haystack_sessions"]

    chunks_by_conv = {cid: _sessions_to_chunks(hs)
                      for cid, hs in haystack_by_conv.items()}

    sorted_convs = sorted(by_conv.keys())
    assert len(sorted_convs) == 10, f"expected 10 convs, got {len(sorted_convs)}"
    train_convs = set(sorted_convs[0:6])
    val_convs   = set(sorted_convs[6:8])
    test_convs  = set(sorted_convs[8:10])

    # Skip adversarial questions — their "gold" is "Not answerable", we want
    # the model to always commit to one of the factual choices.
    SKIP_TYPES = {"adversarial"}

    train, val, test = [], [], []
    skipped = 0
    for cid, qs in by_conv.items():
        chunks = chunks_by_conv[cid]
        bucket = train if cid in train_convs else val if cid in val_convs else test
        for q in qs:
            if q.get("question_type") in SKIP_TYPES:
                skipped += 1
                continue
            bucket.append(_item_to_sample(q, chunks))
    print(f"[locomo_mc10] Skipped {skipped} adversarial questions (\"Not answerable\")")
    return train, val, test
