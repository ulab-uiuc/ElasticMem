# LongMemEval-MC10

Long-memory QA benchmark, recast to 10-choice MC. Each question has its
own independent haystack of conversation sessions (no per-question
shared context with others).

## 1. Dataset Source

- HuggingFace dataset repo: **`Percena/lme-mc10`** —
  `data/lme_s_mc10.json`.
- Auto-downloaded by `huggingface_hub.snapshot_download` on first run
  (see [`scripts/download_data.sh`](./scripts/download_data.sh)).
- Derived from **LongMemEval-S**.
- License: follows the upstream LongMemEval license.

## 2. Format & Schema

JSONL, one line per question:

```python
{
  "question_id": "<unique id>",
  "question": "<user question>",
  "choices": ["...", ..., "..."],         // 10 strings
  "correct_choice_index": 0..9,
  "question_type": "<category>",
  "haystack_sessions": [                   // ~50 sessions, ~540 turns total
      [ {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}, ... ],
      [ ... ],
      ...
  ]
}
```

The 10 choices are mapped to letters `(a)..(j)`. The gold answer is
written as `"({letter})"`. Note that **every question has its own
unique haystack** — there is no `conversation_id` to group questions.

**Aggregate stats**:

- 500 questions
- 500 unique haystacks (one per question)
- avg ~540 turns per haystack across ~50 sessions
- All 500 questions have a real factual gold (no "Not answerable"
  adversarial questions in this MC10 release).

## 3. Train / Val / Test Split

Two operating modes (selected via `split_mode`):

| Mode | train | test | Use case |
|------|-------|------|----------|
| `eval_only` | `[]` | all 500 | Transfer eval — load a checkpoint trained on a different dataset (typically LoCoMo-MC10) and evaluate here. Default for transfer experiments. |
| `random`    | random 80% by `question_id` | remaining 20% | In-domain finetuning on LongMemEval-MC10 itself (`seed=42`). |

In both modes the splitting unit is **`question_id`** (each question
already has its own haystack).

**Frozen split files** are produced by
[`scripts/freeze_split.sh`](./scripts/freeze_split.sh) (run once after
`download_data.sh`); output goes to
[`data/splits/{train,test}_ids.txt`](./data/splits/) — one
`question_id` per line. Cannot be shipped pre-built because the HF
snapshot is not redistributed.

## 4. Retrieval Settings

- **Chunking**: each question's `haystack_sessions` are flattened into
  user+assistant turn-pair chunks; per-question independent (no
  cross-question reuse). Each question's `shared_context_id`
  defaults to its `question_id`.
- **Chunk encoder**: base reasoner's last-token hidden.
- **Query**: question text → reasoner forward → next-token-step
  hidden.
- **Score**: cosine similarity.
- **top-K (`top_z`)**: default **20**.

## 5. Prompt Template

Identical structure to LoCoMo-MC10 §5 (10-choice MC, same
instruction):

```
[ user_question_or_message ]
[ <retrieval gate token, learned> ]
[ <retrieved-chunk representations injected here, top-20> ]
[ <instruction>\n<all_options> ]
[ <answer>  ← model writes this ]
```

`<instruction>` (literal):

```
You MUST pick exactly one option from (a) to (j) — one of them is
guaranteed to be correct. Do NOT say "not answerable", "I don't know",
or refuse. If uncertain, make your best guess. Output exactly one
token in the form (a), (b), (c), (d), (e), (f), (g), (h), (i), or (j).
```

`<all_options>` is the question's 10 choices serialized as
`(a)..(j) <text>`. The expected `<answer>` is one of `(a)..(j)`.

## 6. Generation / Sampling Hyperparams

Same as LoCoMo-MC10 §6:

| Param | Train rollout | Eval (greedy) |
|-------|---------------|---------------|
| `max_new_tokens`   | 5    | 5    |
| `temperature`      | 1.0  | 0.0  |
| `top_p`            | 1.0  | 1.0  |
| `num_generations`  | 4 or 8 (memory-bound; this dataset has long haystacks) | 1 |

> Memory note: each LongMemEval haystack is ~10× longer than a single
> LoCoMo conversation. With Qwen-3B, `num_generations=8` may OOM on
> 48GB GPUs once `top_z=20` is folded in. Drop to `num_generations=4`
> if needed.

## 7. Training Hyperparams

| Param | Value |
|-------|-------|
| optimizer            | AdamW |
| learning rate        | 2e-5  |
| weight_decay         | 0.0   |
| scheduler            | CosineAnnealingLR |
| num_epochs           | 5     |
| batch_size           | 1     |
| GRPO `num_generations` | 4 (memory-safe) or 8 |
| GRPO `epsilon`       | 0.2   |
| `policy_entropy_coef` | 0.001 |
| `top_z`              | 20    |
| `n_tokens_min / max` | 1 / 20 |
| `max_hidden_cache`   | 20    |
| LoRA `r / alpha / dropout` | 32 / 64 / 0.1 |
| LoRA `target_modules` | `q_proj,k_proj,v_proj,o_proj` |

## 8. Reward Function

Same letter match (a–j) as LoCoMo-MC10 §8. Binary 0 / 1.

## 9. Persistence & Logging

**Memory artifact** (auto-built):

- Path: `cache/chunk_embeddings_lme_mc10/{hash}_last1.pt`,
  `{hash}_lastN.pt`
- Cache key: SHA256 of `(chunks_text, model_name, max_chunk_length,
  max_hidden_cache, tokenizer_name)`.
- Build: `bash scripts/build_memory.sh`.
- **Note**: 500 distinct haystacks → 500 distinct cache key entries
  (each haystack is encoded once and reused for its question only).

**Per-question test log**: same `EvalLogger` schema as defined in
[`common/eval_logger.py`](../common/eval_logger.py), with
`dataset = "LongMemEval-MC10"` in the summary header.
