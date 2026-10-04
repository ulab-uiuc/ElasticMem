"""
Gemini-2.0-Flash LLM judge for free-form QA.
Follows MemSkill's 0/0.5/1.0 scoring rubric.

Usage:
    from llm_judge import GeminiJudge, JUDGE_PROMPT
    judge = GeminiJudge()                               # uses env var or default key
    score = judge.score(question, gold_answer, prediction)
    # Or batch:
    scores = judge.score_batch(items)                    # list of (q, gold, pred)
"""
import os
import re
import json
import time
import logging
from typing import List, Tuple, Union

try:
    from json_repair import repair_json as _repair_json
except Exception:
    def _repair_json(s: str) -> str:
        return s

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Set the `GEMINI_API_KEY` environment variable, or pass api_key=... explicitly.
_DEFAULT_KEY = None
_MODEL_NAME = "gemini-3.1-flash-lite-preview"

JUDGE_PROMPT = """You are an expert judge evaluating the quality of an answer for a QA task.
Your goal is to determine whether the model's answer correctly and sufficiently
answers the given question.

Read the following information carefully:

[Question]
{question}

[Ground Truth Answers]
{ground_truth}

[Model Answer]
{model_answer}

Your evaluation criteria:
1. Correctness:
   - Is the model answer factually consistent with ANY of the correct answers?
   - Does it avoid contradictions or introducing false information?

2. Relevance:
   - Does the answer address the question directly without unnecessary content?

3. Completeness:
   - Does the answer include all essential information needed to fully answer the question?
   - Partial answers are allowed but should receive lower scores.

Scoring Rules:
- Score = 1.0 if the answer is fully correct.
- Score = 0.5 if the answer is partially correct but incomplete or slightly inaccurate.
- Score = 0.0 if the answer is incorrect, irrelevant, or contradicts the ground truth.

IMPORTANT — non-answers and refusals MUST get 0.0:
- If the [Model Answer] is a refusal or "I don't know" style response (e.g. "not provided",
  "not specified", "not mentioned", "cannot determine", "unknown", "no information",
  empty, or only echoes the question) AND the [Ground Truth Answers] is a real answer,
  the score MUST be 0.0 regardless of any other content in the model answer.
- Repetitive degenerate outputs (e.g. "Short answer: Short answer: ...") MUST be 0.0.
- An answer that says "X is not specified, but you might be interested in <correct answer>"
  is still 0.0 because it does not commit to the correct answer.

Output Format (STRICT):
Return your output as a JSON dictionary with two fields:
{{
    "explanation": "<brief explanation of your reasoning>",
    "score": <0.0 | 0.5 | 1.0>
}}

Be concise and objective. Do not include anything outside the JSON.
"""


def _parse_score(response_text: str):
    """Extract the numeric score from the judge's JSON output.

    Returns a float clamped to [0.0, 1.0] on success; None on parse failure
    so callers can fall back rather than silently treat as 0.
    """
    if not response_text:
        return None
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
    cleaned = cleaned.replace("```", "").strip()
    score = None
    try:
        score = float(json.loads(_repair_json(cleaned))["score"])
    except Exception as e:
        # Fallback: regex for "score": X
        m = re.search(r'"score"\s*:\s*([0-9.]+)', cleaned)
        if m:
            try:
                score = float(m.group(1))
            except ValueError:
                pass
        if score is None:
            logger.warning(f"Judge parse failed: {e}. Response: {response_text[:200]!r}")
            return None
    # Clamp to [0, 1] — Gemini occasionally outputs 1.5 / -0.3 etc.
    return max(0.0, min(1.0, score))


def _classify_error(e: Exception) -> str:
    """Map an exception to a coarse retry category.

    Returns one of: 'rate_limit' (429), 'server' (5xx), 'transient' (network/timeout),
    'parse_empty' (empty/blocked response), 'fatal' (auth/invalid request — give up).
    """
    msg = str(e).lower()
    code = getattr(e, "code", None)
    if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if any(s in msg for s in ("500", "502", "503", "504", "internal", "unavailable")):
        return "server"
    if any(s in msg for s in ("timeout", "timed out", "deadline", "connection",
                               "broken pipe", "connection reset", "ssl")):
        return "transient"
    if any(s in msg for s in ("api_key", "permission", "unauthorized", "forbidden",
                               "invalid_argument")):
        return "fatal"
    return "transient"  # default: assume retriable


class GeminiJudge:
    """Wraps Gemini with robust retry logic.

    Retry policy (per request):
      - Up to `max_retries` attempts
      - Exponential backoff with jitter
      - Rate-limit (429) waits longer than transient/server
      - Empty / parse-failure responses get one extra retry
      - Fatal errors (bad key, invalid arg) give up immediately
    """

    def __init__(self, api_key: str = None, model_name: str = _MODEL_NAME,
                 max_retries: int = 6, base_delay: float = 1.5,
                 max_delay: float = 60.0, rate_limit_floor: float = 8.0):
        key = api_key or os.environ.get("GEMINI_API_KEY") or _DEFAULT_KEY
        if not key:
            raise ValueError("No Gemini API key found (env GEMINI_API_KEY or default).")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rate_limit_floor = rate_limit_floor

    def _backoff(self, attempt: int, kind: str) -> float:
        """Exponential backoff in seconds with full jitter; rate_limit waits longer."""
        import random as _rnd
        # exponential: base * 2^attempt
        nominal = self.base_delay * (2 ** attempt)
        if kind == "rate_limit":
            nominal = max(nominal, self.rate_limit_floor * (attempt + 1))
        nominal = min(nominal, self.max_delay)
        # full jitter
        return _rnd.uniform(nominal * 0.5, nominal)

    def score(self, question: str, ground_truth, prediction: str):
        """Returns a float in [0, 1] on success, or None on judge failure (so the
        caller can fall back to F1 instead of silently treating it as zero)."""
        if isinstance(ground_truth, (list, tuple)):
            gt_str = ", ".join(str(g) for g in ground_truth)
        else:
            gt_str = str(ground_truth)
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=gt_str, model_answer=prediction or ""
        )

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0, max_output_tokens=512,
                    ),
                )
                text = (resp.text or "") if resp is not None else ""
                score = _parse_score(text)
                if score is not None:
                    return score
                # Parse-failure: response may be empty / blocked / malformed.
                # Treat as transient and try again unless it's the last attempt.
                last_err = ValueError(f"empty/unparseable response: {text[:120]!r}")
                if attempt < self.max_retries - 1:
                    time.sleep(self._backoff(attempt, "transient"))
                    continue
                logger.warning(
                    f"Gemini judge parse failure after {self.max_retries} retries"
                )
                return None
            except Exception as e:
                last_err = e
                kind = _classify_error(e)
                if kind == "fatal":
                    logger.error(f"Gemini fatal error (no retry): {e}")
                    return None
                if attempt < self.max_retries - 1:
                    delay = self._backoff(attempt, kind)
                    if kind == "rate_limit":
                        logger.info(f"Gemini 429 backoff {delay:.1f}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                logger.warning(
                    f"Gemini judge failed after {self.max_retries} retries [{kind}]: {e}"
                )
                return None
        logger.warning(f"Gemini judge gave up. Last error: {last_err}")
        return None

    def score_batch(self, items: List[Tuple[str, Union[str, list], str]],
                    max_workers: int = 8) -> List:
        """Parallel scoring via thread pool. Each entry is float in [0, 1] on success,
        or None on judge failure — caller decides fallback (e.g. F1)."""
        from concurrent.futures import ThreadPoolExecutor
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self.score, q, g, p): i
                for i, (q, g, p) in enumerate(items)
            }
            for fut in futures:
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.warning(f"score_batch item {idx} failed: {e}")
                    results[idx] = None
        return results
