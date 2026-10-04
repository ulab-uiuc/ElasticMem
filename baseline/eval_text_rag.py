"""
Baseline: Qwen2.5-1.5B + SentenceTransformer retrieval + full text RAG.
No latent tokens, no projector — just retrieve top-k chunks as text and let LLM answer.
"""
import os
import sys
import re
import json
import argparse
import logging
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_all_data, parse_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION = (
    "You MUST respond with exactly one option: (a), (b), (c), or (d). "
    "Do not include any explanation or extra text."
)


def extract_option(text: str) -> str:
    text = text.strip().lower()
    m = re.search(r"\(([a-d])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([a-d])\b", text)
    return m.group(1) if m else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--retriever_name", type=str, default="multi-qa-MiniLM-L6-cos-v1")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--question_path", type=str,
                        default="../ICLR2026_RF-Mem/RF_mem/personamem_data/data/questions_32k.csv")
    parser.add_argument("--context_path", type=str,
                        default="../ICLR2026_RF-Mem/RF_mem/personamem_data/data/shared_contexts_32k.jsonl")
    parser.add_argument("--output_path", type=str, default="./baseline/results_text_rag.json")
    parser.add_argument("--split", type=str, default="all", choices=["all", "test"],
                        help="all=entire dataset, test=same 10%% test split as training (seed=42, 80/10/10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load LLM
    logger.info(f"Loading LLM: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load retriever
    logger.info(f"Loading retriever: {args.retriever_name}")
    retriever = SentenceTransformer(args.retriever_name)

    # Load data
    logger.info("Loading data...")
    all_samples, groups = load_all_data(args.question_path, args.context_path)
    logger.info(f"Total samples: {len(all_samples)}")

    # Optional: restrict to same test split as training (seed=42, 80/10/10)
    if args.split == "test":
        import random as _random
        rng = _random.Random(args.seed)
        sids = sorted(set(s["shared_context_id"] for s in all_samples))
        rng.shuffle(sids)
        n_total = len(sids)
        n_eval = max(1, int(n_total * 0.1))
        n_test = max(1, int(n_total * 0.1))
        test_sids = set(sids[:n_test])
        all_samples = [s for s in all_samples if s["shared_context_id"] in test_sids]
        logger.info(f"Filtered to test split: {len(all_samples)} samples from {len(test_sids)} contexts")

    # Precompute chunk embeddings per context
    logger.info("Encoding chunks with retriever...")
    chunk_emb_cache = {}
    for sample in tqdm(all_samples, desc="Encoding"):
        sid = sample["shared_context_id"]
        if sid in chunk_emb_cache:
            continue
        chunks = sample["chunks"]
        embs = retriever.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        chunk_emb_cache[sid] = {"chunks": chunks, "embs": embs}

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

        cached = chunk_emb_cache[sid]
        chunks = cached["chunks"]
        chunk_embs = cached["embs"]

        # Retrieve top-k
        q_emb = retriever.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores = (chunk_embs @ q_emb.T).squeeze(-1)
        top_indices = np.argsort(scores)[::-1][:args.top_k]
        retrieved_texts = [chunks[i] for i in top_indices]

        # Build prompt with retrieved text context
        context_str = "\n\n---\n\n".join(retrieved_texts)
        prompt = (
            f"Here are some relevant conversation excerpts:\n\n"
            f"{context_str}\n\n"
            f"Question: {question}\n"
            f"{INSTRUCTION}\n"
            f"{all_options}"
        )

        # Generate
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=30000).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].size(1):]
        answer = tokenizer.decode(generated, skip_special_tokens=True)

        pred = extract_option(answer)
        gt = extract_option(correct_answer)
        is_correct = pred == gt and gt != ""

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
            "pred_option": pred,
            "gt_option": gt,
            "is_correct": is_correct,
            "topic": topic,
            "question_type": qtype,
        })

    # Print results
    accuracy = correct / max(total, 1)
    logger.info(f"\n{'='*50}")
    logger.info(f"Baseline: {args.model_name} + {args.retriever_name} + top-{args.top_k} text RAG")
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")

    logger.info("\nBy Topic:")
    for topic, stats in sorted(topic_stats.items()):
        acc = stats["correct"] / max(stats["total"], 1)
        logger.info(f"  {topic:30s}  {acc:.4f}  ({stats['correct']}/{stats['total']})")

    logger.info("\nBy Question Type:")
    for qtype, stats in sorted(qtype_stats.items()):
        acc = stats["correct"] / max(stats["total"], 1)
        logger.info(f"  {qtype:40s}  {acc:.4f}  ({stats['correct']}/{stats['total']})")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_name,
            "retriever": args.retriever_name,
            "top_k": args.top_k,
            "overall_accuracy": accuracy,
            "total": total,
            "correct": correct,
            "topic_breakdown": {k: {**v, "accuracy": v["correct"] / max(v["total"], 1)} for k, v in topic_stats.items()},
            "qtype_breakdown": {k: {**v, "accuracy": v["correct"] / max(v["total"], 1)} for k, v in qtype_stats.items()},
            "predictions": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
