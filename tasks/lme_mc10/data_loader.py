"""
LongMemEval-MC10 loader (Percena/lme-mc10 on HuggingFace).

500 questions, 10-choice MC (a..j). Every question has its own independent
haystack (~540 turns each, ~50 sessions). No adversarial / "Not answerable"
options — all 500 have a real factual gold.

Common usage modes:
  1. Transfer evaluation (eval-only): use all 500 as held-out test. Train
     checkpoint is loaded from another dataset (e.g., locomo_mc10).
  2. Finetune: random 80/20 split by question_id.

Returned sample format (matches the rest of the codebase):
    {
      row_data: {
        question_id, user_question_or_message, correct_answer ("(x)"),
        all_options, topic (question_type), question_type
      },
      context: [],
      chunks: List[str],
      shared_context_id: question_id          # every Q is its own context
    }
"""
import os
import json
import random
from typing import Dict, List, Tuple

from huggingface_hub import snapshot_download

_HF_REPO = "Percena/lme-mc10"
_REL_PATH = "data/lme_s_mc10.json"


def _find_local_file(cache_dir: str = None) -> str:
    local_root = snapshot_download(
        repo_id=_HF_REPO, repo_type="dataset",
        cache_dir=cache_dir, allow_patterns=[_REL_PATH],
    )
    return os.path.join(local_root, _REL_PATH)


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sessions_to_chunks(sessions: List[List[Dict]]) -> List[str]:
    """Flatten haystack_sessions → user+assistant turn-pair chunks."""
    chunks: List[str] = []
    for sess in sessions or []:
        if not sess:
            continue
        norm = []
        for t in sess:
            role = (t.get("role") or "unknown").strip().lower()
            text = (t.get("content") or "").strip()
            if not text or role == "system":
                continue
            norm.append({"role": role, "content": text})
        i = 0
        while i < len(norm):
            cur = norm[i]
            if cur["role"] == "user":
                if i + 1 < len(norm) and norm[i + 1]["role"] == "assistant":
                    chunks.append(
                        f"User:\n{cur['content']}\n\nAssistant:\n{norm[i + 1]['content']}"
                    )
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
    return chr(ord("a") + i)


def _item_to_sample(item: Dict) -> Dict:
    idx = int(item["correct_choice_index"])
    choices = item["choices"]
    options_text = "\n".join(f"({_letter(i)}) {c}" for i, c in enumerate(choices))
    qid = item["question_id"]
    return {
        "row_data": {
            "question_id": qid,
            "user_question_or_message": item["question"],
            "correct_answer": f"({_letter(idx)})",
            "all_options": options_text,
            "topic": item.get("question_type", "unknown"),
            "question_type": item.get("question_type", "unknown"),
        },
        "context": [],
        "chunks": _sessions_to_chunks(item["haystack_sessions"]),
        "shared_context_id": qid,               # unique per question
    }


def load_lme_mc10(
    cache_dir: str = None,
    split_mode: str = "random",               # "random" or "eval_only"
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Returns (train, test).

    split_mode:
      - 'eval_only'  → train = [], test = all 500 (use with pretrained ckpt)
      - 'random'     → random shuffle by question, 80/20 by default
    """
    path = _find_local_file(cache_dir=cache_dir)
    items = _read_jsonl(path)
    samples = [_item_to_sample(it) for it in items]

    if split_mode == "eval_only":
        return [], samples

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * test_ratio))
    test = samples[:n_test]
    train = samples[n_test:]
    return train, test
