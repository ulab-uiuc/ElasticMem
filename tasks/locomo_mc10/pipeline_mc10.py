"""
LoCoMo-MC10 pipeline: subclass of LatentChunkPipeline with 10-choice reward.

Parent's `_compute_reward` only handles (a)/(b)/(c)/(d). We extend to a-j.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from elasticmem.pipeline import LatentChunkPipeline


def _extract_letter_aj(text: str) -> str:
    """Extract one of a..j from model output or gold answer. Empty string on failure."""
    if not isinstance(text, str):
        return ""
    lo = text.strip().lower()
    # Prefer parenthesized form (a)..(j)
    m = re.search(r"\(([a-j])\)", lo)
    if m:
        return m.group(1)
    # Fall back to a bare a..j with word boundaries
    m = re.search(r"\b([a-j])\b", lo)
    return m.group(1) if m else ""


class MC10Pipeline(LatentChunkPipeline):
    @staticmethod
    def _compute_reward(prediction: str, correct_answer: str) -> float:
        pred = _extract_letter_aj(prediction)
        gold = _extract_letter_aj(correct_answer)
        if not gold:
            return 0.0
        return 1.0 if pred == gold else 0.0
