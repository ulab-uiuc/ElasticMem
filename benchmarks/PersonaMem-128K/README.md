# PersonaMem-128K

Same task as PersonaMem-32K but at the **128K-token context scale**.
Per-context conversations are 4-5× longer; questions test recall over a
much larger persona memory.

## 1. Dataset Source

- Origin: ICLR2026 RF-Mem release (128K variant); locally cached at
  `latten_mem/personamem_128k/`.
- License: MIT.
- Files (shipped under [`data/`](./data)):
  - `questions_128k.csv` — 5.1 MB
  - `shared_contexts_128k.jsonl` — 71 MB *(over GitHub's 50 MB
    soft-warn threshold; if you do not want to commit it, see
    [`scripts/download_data.sh`](./scripts/download_data.sh) for an
    alternate fetch path.)*

## 2. Format & Schema

Identical column / line schema to PersonaMem-32K (15 CSV columns;
JSONL of `{shared_context_id: [{role, content}, ...]}`). The only
difference is the context length distribution.

**Aggregate stats**:

- 2,727 questions
- 60 unique `shared_context_id` (conversations)
- 20 unique `persona_id`
- avg context length ≈ 100K-128K tokens (~4× the 32K variant)

## 3. Train / Val / Test Split

Same recipe as 32K:

- Splitter: `split_train_eval` (random shuffle of `shared_context_id`,
  `seed=42`).
- Splitting unit: **shared_context_id**.
- Default ratios: `train : val : test = 80 : 10 : 10` →
  48 / 6 / 6 contexts (out of 60 total).
- **Frozen split index files** under
  [`data/splits/`](./data/splits/): `train_ids.txt`, `val_ids.txt`,
  `test_ids.txt` — one `shared_context_id` per line.

## 4. Retrieval Settings

Same chunking, encoder, query, and scoring as 32K (see PersonaMem-32K
README §4 for full description). Differences:

- `cache_dir`: `cache/chunk_embeddings_personamem_128k/`
- `top_z`: default 9, but 128K's longer contexts mean more chunks per
  conversation; ablations have used `top_z = 21 / 27 / 40` (see
  `baseline/` for the text-RAG comparison).
- `max_chunk_length`: 2048 (unchanged — chunk granularity is per
  user+assistant pair, not per fixed token window).

## 5. Prompt Template

Identical to PersonaMem-32K §5 (4-choice MC, instruction asks for one
of `(a)/(b)/(c)/(d)`).

## 6. Generation / Sampling Hyperparams

Identical to PersonaMem-32K §6:

| Param | Train rollout | Eval (greedy) |
|-------|---------------|---------------|
| `max_new_tokens`   | 5    | 5    |
| `temperature`      | 1.0  | 0.0  |
| `top_p`            | 1.0  | 1.0  |
| `num_generations`  | 8    | 1    |

## 7. Training Hyperparams

Same as 32K (see PersonaMem-32K §7) **with one exception** — for
memory reasons it is recommended to either:

- use `Qwen/Qwen2.5-1.5B-Instruct` (hidden_size 1536), or
- raise `top_z` and shrink `n_tokens_max` to keep the input sequence
  within the GPU budget when training Qwen-3B.

## 8. Reward Function

Identical letter-match (a/b/c/d), binary 0 / 1.
See PersonaMem-32K README §8.

## 9. Persistence & Logging

Identical schema as PersonaMem-32K (see its README §9). Different
artifact paths:

**Memory artifact**:
- Path: `cache/chunk_embeddings_personamem_128k/{hash}_last1.pt`,
  `{hash}_lastN.pt`
- Cache key: SHA256 of `(chunks_text, model_name, max_chunk_length,
  max_hidden_cache, tokenizer_name)`.
- Build: `bash scripts/build_memory.sh`

**Per-question test log**:
- Same `EvalLogger` schema, set `dataset: "PersonaMem-128K"` in the
  summary header.
- Default output: `results/eval_test.json`.
