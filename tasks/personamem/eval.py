"""
Evaluation: load trained projector + LoRA, run inference, report metrics.
"""
import os
import re
import json
import argparse
import logging
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from elasticmem.config import Config
from elasticmem.projector import LatentProjector
from elasticmem.pipeline import LatentChunkPipeline
from elasticmem.data_utils import load_all_data, encode_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION = (
    "You MUST respond with exactly one option: (a), (b), (c), or (d). "
    "Do not include any explanation or extra text."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--top_z", type=int, default=None)
    parser.add_argument("--n_tokens_fixed", type=int, default=None)
    parser.add_argument("--question_path", type=str, default=None)
    parser.add_argument("--context_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./results/eval_results.json")
    parser.add_argument("--split", type=str, default="test", choices=["all", "train", "val", "test"],
                        help="Which split to evaluate on. Uses same seed=42 / 80/10/10 split as training.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    if args.model_name:
        cfg.model.model_name = args.model_name
    if args.top_z:
        cfg.retrieval.top_z = args.top_z
    if args.question_path:
        cfg.data.question_path = args.question_path
    if args.context_path:
        cfg.data.context_path = args.context_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reasoner = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Load LoRA
    lora_path = os.path.join(args.checkpoint_dir, "lora_adapter")
    if os.path.exists(lora_path):
        reasoner = PeftModel.from_pretrained(reasoner, lora_path)
        logger.info(f"Loaded LoRA from {lora_path}")

    # Load projector
    projector = LatentProjector(
        hidden_size=cfg.projector.hidden_size,
        intermediate_size=cfg.projector.intermediate_size,
        max_queries=cfg.retrieval.n_tokens_max,
        num_heads=8,
        num_layers=2,
        ffn_mult=2,
        dropout=cfg.projector.dropout,
    ).to(device).to(torch.bfloat16)

    proj_path = os.path.join(args.checkpoint_dir, "projector.pt")
    projector.load_state_dict(torch.load(proj_path, map_location=device))
    projector.eval()
    logger.info(f"Loaded projector from {proj_path}")

    # Only force a fixed n_tokens if user explicitly passed --n_tokens_fixed.
    # Otherwise let the pipeline pick: policy (if loaded) > random fallback.
    n_fixed = args.n_tokens_fixed  # may be None

    # Optional: load BudgetPolicy if checkpoint contains policy.pt
    policy = None
    policy_path = os.path.join(args.checkpoint_dir, "policy.pt")
    if os.path.exists(policy_path):
        from elasticmem.policy import BudgetPolicy
        policy = BudgetPolicy(
            input_dim=cfg.projector.hidden_size,
            hidden_size=256,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            n_choices=cfg.retrieval.n_tokens_max,
            max_chunks=cfg.retrieval.top_z,
        ).to(device).to(torch.float32)
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy.eval()
        logger.info(f"Loaded policy from {policy_path}")

    pipeline = LatentChunkPipeline(
        reasoner=reasoner,
        tokenizer=tokenizer,
        projector=projector,
        top_z=cfg.retrieval.top_z,
        n_tokens_min=cfg.retrieval.n_tokens_min,
        n_tokens_max=cfg.retrieval.n_tokens_max,
        max_hidden_cache=cfg.retrieval.max_hidden_cache,
        policy=policy,
    )

    # Load data
    all_samples, groups = load_all_data(cfg.data.question_path, cfg.data.context_path)
    logger.info(f"Total samples: {len(all_samples)}")

    # Match the training-time split (seed=42, 80/10/10 by shared_context_id).
    # Defaults to the held-out test split to prevent train/val leakage.
    if args.split != "all":
        import random as _random
        rng = _random.Random(args.seed)
        sids = sorted(set(s["shared_context_id"] for s in all_samples))
        rng.shuffle(sids)
        n_total = len(sids)
        n_eval = max(1, int(n_total * 0.1))
        n_test = max(1, int(n_total * 0.1))
        test_sids = set(sids[:n_test])
        eval_sids = set(sids[n_test:n_test + n_eval])
        train_sids = set(sids[n_test + n_eval:])
        split_sids = {"train": train_sids, "val": eval_sids, "test": test_sids}[args.split]
        all_samples = [s for s in all_samples if s["shared_context_id"] in split_sids]
        logger.info(f"Filtered to '{args.split}' split: {len(all_samples)} samples from {len(split_sids)} contexts")

    # Precompute embeddings
    emb_cache = {}
    for sample in tqdm(all_samples, desc="Encoding"):
        sid = sample["shared_context_id"]
        if sid in emb_cache:
            continue
        emb_cache[sid] = encode_chunks(
            reasoner, tokenizer, sample["chunks"],
            max_length=cfg.data.max_chunk_length,
            max_hidden_cache=cfg.retrieval.max_hidden_cache,
            cache_dir=cfg.data.cache_dir,
            device=device,
        )

    # Evaluate
    results = []
    correct = 0
    total = 0
    topic_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    qtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in tqdm(all_samples, desc="Evaluating"):
        sid = sample["shared_context_id"]
        row = sample["row_data"]
        question = row["user_question_or_message"]
        correct_answer = row["correct_answer"]
        all_options = row["all_options"]
        topic = row.get("topic", "unknown")
        qtype = row.get("question_type", "unknown")

        cached = emb_cache[sid]
        chunk_last1 = cached["last1"].to(device)
        chunk_lastN = cached["lastN"].to(device)

        answer = pipeline.generate(
            question=question,
            instruction=INSTRUCTION,
            all_options=all_options,
            chunk_embs=chunk_last1,
            chunk_hiddens=chunk_lastN,
            n_tokens_fixed=n_fixed,
        )

        reward = pipeline._compute_reward(answer, correct_answer)
        is_correct = reward > 0

        if is_correct:
            correct += 1
            topic_stats[topic]["correct"] += 1
            qtype_stats[qtype]["correct"] += 1
        total += 1
        topic_stats[topic]["total"] += 1
        qtype_stats[qtype]["total"] += 1

        results.append({
            "question_id": row.get("question_id", ""),
            "question": question,
            "correct_answer": correct_answer,
            "predicted": answer.strip(),
            "is_correct": is_correct,
            "topic": topic,
            "question_type": qtype,
        })

    accuracy = correct / max(total, 1)
    logger.info(f"\n{'='*50}")
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")

    logger.info("\nBy Topic:")
    for topic, stats in sorted(topic_stats.items()):
        acc = stats["correct"] / max(stats["total"], 1)
        logger.info(f"  {topic:30s}  {acc:.4f}  ({stats['correct']}/{stats['total']})")

    logger.info("\nBy Question Type:")
    for qtype, stats in sorted(qtype_stats.items()):
        acc = stats["correct"] / max(stats["total"], 1)
        logger.info(f"  {qtype:40s}  {acc:.4f}  ({stats['correct']}/{stats['total']})")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": accuracy,
            "total": total,
            "correct": correct,
            "n_tokens_fixed": n_fixed,  # None means policy/random was used
            "used_policy": policy is not None and n_fixed is None,
            "topic_breakdown": {k: {**v, "accuracy": v["correct"] / max(v["total"], 1)} for k, v in topic_stats.items()},
            "qtype_breakdown": {k: {**v, "accuracy": v["correct"] / max(v["total"], 1)} for k, v in qtype_stats.items()},
            "predictions": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
