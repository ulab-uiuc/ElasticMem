"""
ALFWorld GRPO pipeline (v4 — snap-aligned strict reward).

Thin subclass of LatentChunkPipeline. Field mapping:

  question       = task objective
  instruction    = fixed system prompt
  all_options    = history(last 5 (action, obs)) + current_obs + admissible + "Action:"
  correct_answer = JSON-encoded {"gold": expert_action, "admissible": [...]}
                   so the reward function can replicate eval-time snap-to-admissible.

Reward (v4, strict, deployment-aligned):
  1. Snap the model's first-line generation to nearest admissible (same
     `_snap_action` used at eval).
  2. reward = 1.0 if snapped action == gold (string equality after normalise),
              else 0.0.

This eliminates the train/test gap that the tiered string reward had:
  * Eval executes the snapped action — train now scores the snapped action.
  * No partial credit for "almost matched the expert string"; the only thing
    that matters is whether the snap result == what the expert did.
"""
import os
import re
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional

from pipeline import LatentChunkPipeline


SYSTEM_INSTRUCTION = (
    "You are controlling a text-based ALFWorld environment. "
    "Choose the NEXT action as ONE admissible command string. "
    "Output only the command, copied verbatim from the admissible list."
)


def render_prompt_tail(
    history: str,
    current_obs: str,
    admissible: Optional[List[str]] = None,
) -> str:
    parts: List[str] = []
    if history:
        parts += ["Interaction history so far:", history.strip(), ""]
    parts += ["Current observation:", (current_obs or "").strip()]
    if admissible:
        parts += ["", "Admissible actions (choose exactly ONE, copy verbatim):"]
        parts += [f"- {a}" for a in admissible]
    parts += ["", "Action:"]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
#  Parsing helpers
# ──────────────────────────────────────────────────────────────────────

_ACTION_PREFIX_RE = re.compile(r'^\s*action\s*[:=\-]\s*', re.IGNORECASE)


def _first_line(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = _ACTION_PREFIX_RE.sub('', text.strip())
    if not t:
        return ""
    line = t.splitlines()[0]
    return line.strip().strip('"').strip("'").strip('`')


def _normalise_action(text: str) -> str:
    return " ".join(_first_line(text).lower().split())


def _snap_action(pred: str, admissible: List[str]) -> str:
    """Same logic as eval_env._match_action: pick the admissible command
    closest to the model's first-line output. Returns the matched admissible
    string, or the raw first line if no admissible was supplied."""
    line = _first_line(pred)
    if not admissible:
        return line
    if not line:
        return admissible[0]
    p = line.strip().lower()
    full = (pred or "").lower()
    # Tier 1: exact match against any admissible.
    for cmd in admissible:
        if str(cmd).strip().lower() == p:
            return cmd
    # Tier 2: substring match anywhere in the response.
    for cmd in admissible:
        if str(cmd).strip().lower() in full:
            return cmd
    # Tier 3: token-overlap fallback.
    p_tokens = set(p.split())
    best, best_sc = admissible[0], -1
    for cmd in admissible:
        c_tokens = set(str(cmd).strip().lower().split())
        sc = len(p_tokens & c_tokens)
        if sc > best_sc:
            best, best_sc = cmd, sc
    return best


# ──────────────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────────────

class ALFWorldGRPOPipeline(LatentChunkPipeline):
    """GRPO variant with deployment-aligned strict reward.

    `correct_answer` is expected to be a JSON string carrying both the gold
    action and the admissible list at this step:
        {"gold": "<action>", "admissible": ["...", ...]}
    Falls back to plain-string gold (no snap, exact-match only) if parsing
    fails — so the pipeline still works with old training data.
    """

    @staticmethod
    def _compute_reward(prediction: str, correct_answer: str) -> float:
        gold = ""
        admissible: List[str] = []
        try:
            payload = json.loads(correct_answer)
            if isinstance(payload, dict):
                gold = str(payload.get("gold", ""))
                ac = payload.get("admissible") or []
                if isinstance(ac, list):
                    admissible = [str(x) for x in ac if isinstance(x, str)]
        except (json.JSONDecodeError, TypeError):
            gold = correct_answer if isinstance(correct_answer, str) else ""

        g = _normalise_action(gold)
        if not g:
            return 0.0

        snapped = _snap_action(prediction, admissible)
        s = _normalise_action(snapped)
        return 1.0 if s == g else 0.0
