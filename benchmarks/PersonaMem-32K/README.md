# PersonaMem-32K

Long-persona-context multi-choice QA at the 32K-token scale.

## 1. Dataset Source

- Origin: ICLR2026 RF-Mem release; locally cached at
  `latten_mem/ICLR2026_RF-Mem/RF_mem/personamem_data/data/`.
- License: MIT.
- Files (shipped under [`data/`](./data)):
  - `questions_32k.csv` — 1.3 MB
  - `shared_contexts_32k.jsonl` — 5.4 MB

## 2. Format & Schema

**`questions_32k.csv`** — 15 columns, one row per question:

```
persona_id, question_id, question_type, topic,
context_length_in_tokens, context_length_in_letters,
distance_to_ref_in_blocks, distance_to_ref_in_tokens,
num_irrelevant_tokens, distance_to_ref_proportion_in_context,
user_question_or_message, correct_answer, all_options,
shared_context_id, end_index_in_shared_context
```

- `correct_answer` is a letter form like `(c)` (4-choice MC: a/b/c/d).
- `all_options` is a serialized text:
  `(a) <choice text>\n(b) ...\n(c) ...\n(d) ...`
- `shared_context_id` links a question back to its conversation.

**`shared_contexts_32k.jsonl`** — one line per context, each line:

```json
{ "<shared_context_id>": [
    {"role": "system", "content": "Current user persona: ..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
] }
```

A `system` turn fixes the persona; user/assistant turns alternate.

**Aggregate stats**:

- 589 questions
- 37 unique `shared_context_id` (conversations)
- 20 unique `persona_id`
- avg context length ≈ 27K tokens

## 3. Train / Val / Test Split

- Splitter: `split_train_eval` (random shuffle of `shared_context_id`,
  fixed `seed=42`).
- Splitting unit: **shared_context_id** (NOT question) — prevents
  leakage of the same conversation across splits.
- Default ratios: `train : val : test = 80 : 10 : 10` →
  31 / 3 / 3 contexts (out of 37 total).
- **Frozen split index files** live under
  [`data/splits/`](./data/splits/): `train_ids.txt`, `val_ids.txt`,
  `test_ids.txt` — one `shared_context_id` per line. Use these to
  guarantee bit-identical splits regardless of Python / random
  implementation.

```python
splits = split_train_eval(samples,
                          eval_ratio=0.1, test_ratio=0.1, seed=42)
train_samples, eval_samples, test_samples = splits
```

## 4. Retrieval Settings

- **Chunking**: each context's full message list is parsed into
  user+assistant turn-pair chunks (see `parse_chunks(context)`).
- **Chunk encoder**: the base reasoner model itself
  (`encode_chunks(...)` runs each chunk through one forward and stores
  (a) the last-token hidden as the retrieval key and
  (b) the last-N hidden states as the content cache).
- **Query**: the user question forwarded through the reasoner; the
  hidden state after one extra step is used as the query vector.
- **Score**: cosine similarity (key vs query, both L2-normalised).
- **top-K (`top_z`)**: default 9; ablations have used 5 / 10 / 20 / 40.

## 5. Prompt Template

Four fields are concatenated to form the model input
(see `pipeline.build_full_sequence`):

```
[ user_question_or_message ]
[ <retrieval gate token, learned> ]
[ <retrieved-chunk representations injected here, top-K> ]
[ <instruction>\n<all_options> ]
[ <answer>  ← the model writes this ]
```

`<instruction>` (literal):

```
Answer with exactly one of the four options below, formatted as a
single token like "(a)", "(b)", "(c)", or "(d)". Do not output any
other text.
```

`<all_options>` is the question's `all_options` field verbatim.
The expected `<answer>` is one of `(a)/(b)/(c)/(d)`.

## 6. Generation / Sampling Hyperparams

| Param | Train rollout | Eval (greedy) |
|-------|---------------|---------------|
| `max_new_tokens`   | 5    | 5    |
| `temperature`      | 1.0  | (greedy, T=0) |
| `top_p`            | 1.0  | 1.0  |
| `num_generations`  | 8    | 1    |
| sampling           | multinomial | argmax |

## 7. Training Hyperparams

| Param | Value |
|-------|-------|
| optimizer            | AdamW |
| learning rate        | 1e-4  |
| weight_decay         | 0.0   |
| scheduler            | CosineAnnealingLR |
| num_epochs           | 10    |
| batch_size (samples per optimizer step) | 1 |
| GRPO `num_generations` | 8 |
| GRPO `epsilon` (PPO clip) | 0.2 |
| `policy_entropy_coef` | 0.001 |
| `top_z`              | 9     |
| `n_tokens_min / max` | 1 / 20 |
| `max_hidden_cache`   | 20    |
| LoRA `r / alpha / dropout` | 32 / 64 / 0.1 |
| LoRA `target_modules` | `q_proj,k_proj,v_proj,o_proj` |

## 8. Reward Function

Letter-match (a/b/c/d):

```python
def _compute_reward(prediction, correct_answer):
    p = extract_letter_abcd(prediction.strip().lower())
    g = extract_letter_abcd(correct_answer.strip().lower())
    return 1.0 if (g and p == g) else 0.0
```

`extract_letter_abcd` greps for `(a|b|c|d)` first, then a bare a-d
word boundary. Binary 0/1.

## 9. Persistence & Logging

**Memory artifact** (auto-built on first run, then reused):

- Path: `cache/chunk_embeddings_personamem_32k/{hash}_last1.pt`
  and `{hash}_lastN.pt`
- Cache key: SHA256 of `(chunks_text, model_name, max_chunk_length,
  max_hidden_cache, tokenizer_name)`
- Rebuild via `bash scripts/build_memory.sh`

**Per-question test log** (every eval run produces one JSON):

- Written by `EvalLogger.dump()` from `common/eval_logger.py`
- Schema:

```json
{
  "summary": {
    "dataset": "PersonaMem-32K",
    "method":  "<your method name>",
    "num_samples": 59,
    "overall_score": 0.71,
    "per_task_type": {"recall_user_shared_facts": 0.78, ...},
    "tokens": {"prompt_total": ..., "completion_total": ...,
               "prompt_avg": ..., "completion_avg": ..., "total_total": ...}
  },
  "results": [
    {
      "question_id": "<uuid>",
      "input": {
        "objective":           "<user_question_or_message>",
        "history":             "",                       // not used here
        "current_obs":         "",
        "admissible":          [],
        "retrieved_chunk_ids": [12, 47, 88, ...],
        "prompt_full":         "<full prompt text>"
      },
      "output": {
        "raw_text":      "(c)",
        "parsed_answer": "(c)"
      },
      "score": 1.0,
      "tokens": {"prompt": 4128, "completion": 3, "total": 4131},
      "extra":  {"task_type": "recall_user_shared_facts",
                 "shared_context_id": "..."}
    },
    ...
  ]
}
```

Default output path: `results/eval_test.json` (one file per
`(method, split)` pair). Methods MUST follow this schema for
cross-baseline comparison.
