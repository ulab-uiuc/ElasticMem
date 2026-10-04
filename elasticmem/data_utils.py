"""
PersonaMem data loading + offline chunk encoding utilities.
"""
import os
import csv
import json
import hashlib
from typing import List, Dict, Tuple, Generator

import torch
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
#  1. Load PersonaMem raw data
# ──────────────────────────────────────────────

def build_jsonl_index(jsonl_path: str) -> Dict[str, int]:
    """Scan JSONL once → {shared_context_id: file_offset}."""
    index = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def load_context_by_id(jsonl_path: str, offset: int):
    """Seek to offset, load one JSONL line, return its value."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


def load_all_data(question_path: str, context_path: str) -> List[Dict]:
    """
    Load all questions with their conversation contexts.
    Groups by shared_context_id to avoid redundant context loading.

    Returns: list of dicts, each containing:
        - row_data: all CSV fields for this question
        - context: the conversation history [{role, content}, ...]
        - chunks: parsed user+assistant pair chunks (list of str)
    """
    jsonl_index = build_jsonl_index(context_path)

    # Group questions by shared_context_id
    groups = {}  # shared_context_id -> [row_data, ...]
    with open(question_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["shared_context_id"]
            if sid not in groups:
                groups[sid] = []
            groups[sid].append(row)

    all_samples = []
    for sid, rows in groups.items():
        context = load_context_by_id(context_path, jsonl_index[sid])
        chunks = parse_chunks(context)
        for row in rows:
            all_samples.append({
                "row_data": row,
                "context": context,
                "chunks": chunks,
                "shared_context_id": sid,
            })

    return all_samples, groups


def parse_chunks(context: List[Dict[str, str]]) -> List[str]:
    """
    Parse conversation history into chunks.
    Each chunk = "User:\\n{user_text}\\n\\nAssistant:\\n{assistant_text}"
    Skips system messages, merges consecutive same-role messages.
    """
    # Filter and normalize
    norm = []
    for i, msg in enumerate(context):
        role = (msg.get("role") or "unknown").strip().lower()
        text = (msg.get("content") or "").strip()
        if not text or role == "system":
            continue
        norm.append({"role": role, "content": text})

    # Merge consecutive same-role messages
    merged = []
    for item in norm:
        if not merged:
            merged.append({"role": item["role"], "content": item["content"]})
            continue
        if item["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + item["content"]
        else:
            merged.append({"role": item["role"], "content": item["content"]})

    # Pair user + assistant
    docs = []
    i = 0
    while i < len(merged):
        cur = merged[i]
        if cur["role"] == "user":
            if i + 1 < len(merged) and merged[i + 1]["role"] == "assistant":
                text = f"User:\n{cur['content']}\n\nAssistant:\n{merged[i + 1]['content']}"
                docs.append(text)
                i += 2
                continue
            else:
                i += 1
                continue
        elif cur["role"] == "assistant":
            text = f"Assistant:\n{cur['content']}"
            docs.append(text)
            i += 1
        else:
            i += 1

    return docs


# ──────────────────────────────────────────────
#  2. Offline chunk encoding with Reasoner
# ──────────────────────────────────────────────

def _context_hash(chunks: List[str], extra: dict = None) -> str:
    """Hash chunk contents + optional signature (model name, max_length, max_hidden_cache).

    Embeddings depend on the encoder model AND on how many hidden states we cache,
    so we bind those into the key to avoid cross-config reuse.
    """
    payload = {"chunks": chunks, "extra": extra or {}}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@torch.no_grad()
def encode_chunks(
    model,
    tokenizer,
    chunks: List[str],
    max_length: int = 512,
    max_hidden_cache: int = 10,
    cache_dir: str = "./cache/chunk_embeddings",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Encode all chunks for one conversation history.

    For each chunk:
        1. Tokenize (truncate to max_length)
        2. Forward through model with output_hidden_states=True
        3. Take last `max_hidden_cache` hidden states from last layer

    Returns dict:
        "last1":   [num_chunks, 1, hidden_size]   → for retrieval
        "lastN":   [num_chunks, max_hidden_cache, hidden_size] → for latent injection

    Results are cached to disk by content hash.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Bind cache key to model config (tokenizer + hidden_size + max_length + max_hidden_cache)
    # so we never silently reuse embeddings produced under a different setup.
    sig = {
        "model": getattr(getattr(model, "config", None), "name_or_path", type(model).__name__),
        "max_length": max_length,
        "max_hidden_cache": max_hidden_cache,
        "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
    }
    cache_key = _context_hash(chunks, extra=sig)
    last1_path = os.path.join(cache_dir, f"{cache_key}_last1.pt")
    lastN_path = os.path.join(cache_dir, f"{cache_key}_lastN.pt")

    if os.path.exists(last1_path) and os.path.exists(lastN_path):
        last1 = torch.load(last1_path, map_location="cpu")
        lastN = torch.load(lastN_path, map_location="cpu")
        return {"last1": last1, "lastN": lastN}

    all_last1 = []
    all_lastN = []

    for chunk_text in chunks:
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].squeeze(0)  # [seq_len, hidden_size]

        # last 1 token for retrieval
        last1 = hidden[-1:].cpu()  # [1, hidden_size]

        # last N tokens for latent injection
        n = min(max_hidden_cache, hidden.size(0))
        lastN = hidden[-n:].cpu()  # [n, hidden_size]
        # Pad to max_hidden_cache if chunk is shorter
        if lastN.size(0) < max_hidden_cache:
            pad = torch.zeros(max_hidden_cache - lastN.size(0), hidden.size(-1))
            lastN = torch.cat([pad, lastN], dim=0)

        all_last1.append(last1)
        all_lastN.append(lastN)

    last1_tensor = torch.stack(all_last1)  # [num_chunks, 1, hidden_size]
    lastN_tensor = torch.stack(all_lastN)  # [num_chunks, max_hidden_cache, hidden_size]

    torch.save(last1_tensor, last1_path)
    torch.save(lastN_tensor, lastN_path)

    return {"last1": last1_tensor, "lastN": lastN_tensor}


@torch.no_grad()
def precompute_dense_scores(
    all_samples: list,
    cache_path: str = "./cache/dense_scores.pt",
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
) -> dict:
    """
    Precompute dense retrieval scores for each (question, chunks) pair using
    a frozen SentenceTransformer. Returns {question_id: scores_tensor}.
    """
    import os
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    dense_model = SentenceTransformer(model_name, device="cpu")

    # Group by context so we encode each context's chunks only once
    context_chunks = {}
    for s in all_samples:
        sid = s["shared_context_id"]
        if sid not in context_chunks:
            context_chunks[sid] = s["chunks"]

    chunk_embs_cache = {}
    for sid, chunks in tqdm(context_chunks.items(), desc="Dense-encode chunks"):
        chunk_embs_cache[sid] = dense_model.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        )

    dense_scores = {}
    for s in tqdm(all_samples, desc="Dense-score questions"):
        qid = s["row_data"]["question_id"]
        question = s["row_data"]["user_question_or_message"]
        sid = s["shared_context_id"]
        q_emb = dense_model.encode(
            [question], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        ).squeeze(0)
        c_embs = chunk_embs_cache[sid]
        scores = (c_embs @ q_emb).cpu()       # [num_chunks]
        dense_scores[qid] = scores

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(dense_scores, cache_path)
    del dense_model
    return dense_scores


@torch.no_grad()
def encode_query(model, tokenizer, question: str, max_length: int = 128, device: str = "cuda") -> torch.Tensor:
    """
    Encode a question → last token hidden state [1, hidden_size].
    """
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)

    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].squeeze(0)  # [seq_len, hidden_size]
    return hidden[-1:]  # [1, hidden_size]
