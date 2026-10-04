"""
Offline: summarise the memory-pool (M) expert trajectories into short skill cards
using Gemini. Output = JSON list of {task_type, gamefile, objective, summary}.

Run from the repo root so imports resolve:
    python -m alfworld_il.summarize \
        --offline-data ./MemSkill/data/alfworld_train_offline.json \
        --output ./alfworld_il/data/alfworld_skill_cards.json \
        --memory-ratio 0.8 \
        --workers 8
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from tqdm import tqdm

from llm_judge import _DEFAULT_KEY, _MODEL_NAME, _classify_error
from alfworld_il.data_loader import load_alfworld_il


SUMMARY_PROMPT = """You will read ONE successful ALFWorld expert trajectory.
Write a compact PROCEDURAL SKILL CARD (150-250 words) that a future agent
could retrieve to solve similar tasks.

Rules:
- Line 1: "Task type: <type>. Objective pattern: <one-line gloss>"
- Then 3-6 numbered steps describing the high-level strategy (not the
  exact action strings; generalise across object names).
- Mention the key receptacles/objects involved and WHY each step matters.
- Do NOT copy the raw trajectory verbatim.
- Output the card directly, no preamble or trailing comments.

==== TASK_TYPE ====
{task_type}

==== OBJECTIVE ====
{objective}

==== TRAJECTORY ====
{trajectory}
"""


def _summarise_one(model, entry: Tuple[str, str, Dict],
                   max_retries: int = 5, base_delay: float = 1.5) -> Dict:
    task_type, gf, data = entry
    prompt = SUMMARY_PROMPT.format(
        task_type=task_type,
        objective=data.get("objective", ""),
        trajectory=(data.get("trajectory", "") or "")[:6000],
    )
    import time, random as _rnd
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, max_output_tokens=512,
                ),
            )
            text = (resp.text or "").strip() if resp is not None else ""
            if text:
                return {
                    "task_type": task_type,
                    "gamefile": gf,
                    "objective": data.get("objective", ""),
                    "summary": text,
                }
        except Exception as e:
            kind = _classify_error(e)
            if kind == "fatal":
                logging.error(f"Gemini fatal: {e}")
                break
            delay = min(30.0, base_delay * (2 ** attempt)) + _rnd.uniform(0, 1.5)
            time.sleep(delay)
    return {
        "task_type": task_type,
        "gamefile": gf,
        "objective": data.get("objective", ""),
        "summary": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-data", required=True,
                    help="Path to alfworld_train_offline.json")
    ap.add_argument("--output", required=True,
                    help="Path for the skill-card JSON")
    ap.add_argument("--memory-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="Skip games whose gamefile already has a non-empty summary")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only summarise first N (debug)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    memory_entries, _ = load_alfworld_il(
        args.offline_data, memory_ratio=args.memory_ratio, seed=args.seed
    )
    if args.limit > 0:
        memory_entries = memory_entries[: args.limit]

    existing: Dict[str, Dict] = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for card in json.load(f):
                if isinstance(card, dict) and card.get("summary"):
                    existing[card["gamefile"]] = card

    todo = [e for e in memory_entries if e[1] not in existing]
    print(f"[summarize] memory pool total={len(memory_entries)}  "
          f"already done={len(existing)}  remaining={len(todo)}")

    api_key = os.environ.get("GEMINI_API_KEY") or _DEFAULT_KEY
    if not api_key:
        raise SystemExit("Set the GEMINI_API_KEY environment variable to build skill cards.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_MODEL_NAME)

    results: List[Dict] = list(existing.values())

    def _save():
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_summarise_one, model, e): e for e in todo}
            for i, fut in enumerate(tqdm(as_completed(futs),
                                         total=len(futs),
                                         desc="Summarising")):
                card = fut.result()
                results.append(card)
                if (i + 1) % 50 == 0:
                    _save()
    _save()

    ok = sum(1 for r in results if r.get("summary"))
    print(f"[summarize] wrote {len(results)} cards ({ok} non-empty) → {args.output}")


if __name__ == "__main__":
    main()
