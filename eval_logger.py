"""
Shared per-sample evaluation logger for all ElasticMem datasets.

Every per-question entry follows a standard schema so results across
datasets / methods / baselines are directly comparable. See README §9
"Persistence & Logging" in each dataset subfolder for the schema.

Usage:

    from common.eval_logger import EvalLogger

    logger = EvalLogger(
        dataset="LoCoMo-MC10",
        method="ours-ckpt-v4_epoch3",
        output_path="results/eval_test.json",
    )

    for sample in dataset_iter:
        prompt_text = build_prompt(sample)
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.size(1)

        gen_ids = model.generate(...)
        completion_tokens = gen_ids.size(1) - prompt_tokens
        raw_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        parsed = parse_answer(raw_output)
        score = compute_score(parsed, sample.gold)

        logger.log_sample(
            qid=sample["question_id"],
            input={
                "objective": sample["question"],
                "history": sample.get("history", ""),
                "current_obs": sample.get("current_obs", ""),
                "admissible": sample.get("admissible", []),
                "retrieved_chunk_ids": retrieved_idx.tolist(),
                "prompt_full": prompt_text,
            },
            output={
                "raw_text": raw_output,
                "parsed_answer": parsed,
                "snapped_action": maybe_snapped,
            },
            score=score,
            tokens={"prompt": prompt_tokens, "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens},
            extra={"task_type": sample.get("task_type"),
                   "shared_context_id": sample.get("shared_context_id")},
        )

    logger.dump()
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional


class EvalLogger:
    """Append per-sample eval rows; flush a structured JSON at the end."""

    def __init__(
        self,
        dataset: str,
        method: str,
        output_path: str,
        store_full_prompt: bool = True,
    ) -> None:
        self.dataset = dataset
        self.method = method
        self.output_path = output_path
        self.store_full_prompt = store_full_prompt
        self.results: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def log_sample(
        self,
        qid: str,
        input: Dict[str, Any],
        output: Dict[str, Any],
        score: float,
        tokens: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.store_full_prompt:
            input = {k: v for k, v in input.items() if k != "prompt_full"}
        self.results.append({
            "question_id": qid,
            "input": input,
            "output": output,
            "score": float(score),
            "tokens": {
                "prompt":     int(tokens.get("prompt", 0)),
                "completion": int(tokens.get("completion", 0)),
                "total":      int(tokens.get("total",
                                  tokens.get("prompt", 0) + tokens.get("completion", 0))),
            },
            "extra": extra or {},
        })

    def dump(self, also_print_summary: bool = True) -> Dict[str, Any]:
        """Flush results + computed summary (overall + per-task type)."""
        scores = [r["score"] for r in self.results]
        prompt_t = [r["tokens"]["prompt"] for r in self.results]
        compl_t = [r["tokens"]["completion"] for r in self.results]

        per_type: Dict[str, List[float]] = defaultdict(list)
        for r in self.results:
            tt = r.get("extra", {}).get("task_type")
            if tt:
                per_type[tt].append(r["score"])

        summary = {
            "dataset": self.dataset,
            "method":  self.method,
            "num_samples": len(self.results),
            "overall_score": (sum(scores) / len(scores)) if scores else 0.0,
            "per_task_type": {k: sum(v) / len(v) for k, v in per_type.items()},
            "tokens": {
                "prompt_total":     sum(prompt_t),
                "completion_total": sum(compl_t),
                "prompt_avg":       (statistics.mean(prompt_t) if prompt_t else 0.0),
                "completion_avg":   (statistics.mean(compl_t) if compl_t else 0.0),
                "total_total":      sum(prompt_t) + sum(compl_t),
            },
        }
        payload = {"summary": summary, "results": self.results}
        with open(self.output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if also_print_summary:
            print(json.dumps(summary, indent=2))
        return summary
