# ElasticMem — Datasets & Settings

This repo collects everything needed to reproduce experiments on five
memory-augmented benchmarks under a single, comparable protocol. Each
subfolder is a self-contained recipe: data source / format, train/test
splits, prompt template, retrieval setting, generation hyperparams,
training hyperparams, reward function, and persistence schema.

## Datasets

| # | Subfolder | Task | # Questions / Games | Memory unit |
|---|-----------|------|---------------------|-------------|
| 1 | [PersonaMem-32K](./PersonaMem-32K)   | 4-choice MC, long persona context  | 589 Q / 37 contexts | conversation turn-pair chunks |
| 2 | [PersonaMem-128K](./PersonaMem-128K) | 4-choice MC, very-long context     | 2,727 Q / 60 contexts | conversation turn-pair chunks |
| 3 | [LoCoMo-MC10](./LoCoMo-MC10)         | 10-choice MC, long dialogue        | 1,986 Q / 10 conversations | session turn-pair chunks |
| 4 | [LongMemEval-MC10](./LongMemEval-MC10) | 10-choice MC, very-long memory   | 500 Q / 500 contexts (per-Q haystack) | session turn-pair chunks |
| 5 | [ALFWorld](./ALFWorld)               | interactive text adventure (env)   | 140 valid_seen + 134 valid_unseen | LLM-generated procedural skill cards |

Each subfolder's `README.md` details:

1. Dataset source
2. Format & schema
3. Train / val / test split
4. Retrieval settings (top-K)
5. Prompt template
6. Generation / sampling hyperparams
7. Training hyperparams
8. Reward function
9. Persistence & logging schema

## Base Models

All experiments default to one of two open Qwen instruct checkpoints:

| Model | hidden_size | Notes |
|-------|------------:|-------|
| `Qwen/Qwen2.5-3B-Instruct`   | 2048 | default for all 5 datasets |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1536 | smaller fallback / ablation |

The choice changes only `cfg.model.model_name`. The projector,
BudgetPolicy, and LoRA shapes auto-adjust to the model's hidden size.

## Common Tooling

- [`common/eval_logger.py`](./common/eval_logger.py) — shared per-sample
  logger with the canonical result JSON schema (used by every dataset's
  eval script, see §9 of each subfolder).

## Frozen Splits

Every dataset ships **plain-text index files** under
`<dataset>/data/splits/` so that the train / val / test partitions
are bit-identical regardless of Python version, `random` library
internals, or shuffle implementation. Each file is one id per line:

| Dataset | Frozen files | Unit |
|---------|--------------|------|
| PersonaMem-32K   | `train_ids.txt` (31), `val_ids.txt` (3), `test_ids.txt` (3)              | shared_context_id |
| PersonaMem-128K  | `train_ids.txt` (48), `val_ids.txt` (6), `test_ids.txt` (6)              | shared_context_id |
| LoCoMo-MC10      | `train_conv_ids.txt` (6), `val_conv_ids.txt` (2), `test_conv_ids.txt` (2)| conversation_id   |
| LongMemEval-MC10 | `train_ids.txt` (400), `test_ids.txt` (100) — produced by `freeze_split.sh` | question_id |
| ALFWorld         | `M_pool_gamefiles.txt` (2,832), `T_pool_gamefiles.txt` (709), `valid_seen_gamefiles.txt` (140), `valid_unseen_gamefiles.txt` (134) | trajectory (gamefile path) |

Baselines MUST consume these files instead of re-running the seeded
shuffle locally — different RNG implementations can otherwise produce
divergent splits even with the same seed.

## Persistence Convention

For every experiment two artifact families must be saved:

- **Memory artifact** — the processed retrieval bank (e.g.,
  `cache/chunk_embeddings_<dataset>/...`). Generated once per
  `(dataset, base_model, max_chunk_length, max_hidden_cache)`
  combination. Reused by every subsequent train / eval run.

- **Per-question test log** — one JSON per eval run, with one entry per
  question. Schema includes the full prompt, raw model output, parsed
  answer, score, and prompt / completion token counts. Aggregate
  summary (overall + per-task-type accuracy + average token usage) is
  computed automatically by `EvalLogger.dump()`.

This makes results across baselines apples-to-apples comparable
without re-loading models.

## Baselines to Run

For each of the 5 datasets above, the following memory-based methods
should be evaluated under the same prompt template, top-K, generation
hyperparams, and per-question logging schema as defined in each
subfolder's README:

| Baseline   | Type                       | Notes |
|------------|----------------------------|-------|
| MemoryBank | rolling memory bank        | -     |
| A-MEM      | agentic memory             | -     |
| LightMem   | lightweight retrieval      | -     |
| Mem0       | structured memory          | -     |
| MemoryOS   | OS-style memory layers     | -     |
| MemP       | -                          | -     |
| LangMem    | -                          | -     |

Implementation links / responsible parties to be filled in.
