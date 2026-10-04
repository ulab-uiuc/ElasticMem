# LoCoMo-MC10

Multi-choice memory QA derived from LoCoMo (LOng COnversation MEmory).
Each long dialogue spawns multiple 10-choice questions.

## 1. Dataset Source

- HuggingFace dataset repo: **`Percena/locomo-mc10`** —
  `data/locomo_mc10.json`.
- Auto-downloaded by `huggingface_hub.snapshot_download` on first run
  (see [`scripts/download_data.sh`](./scripts/download_data.sh)).
- Derived from the original LoCoMo dataset (Maharana et al.) by
  reformulating the free-form QA into 10-way MC.
- License: follows the upstream LoCoMo license.

## 2. Format & Schema

JSONL, one line per question:

```python
{
  "question_id": "<unique id>",
  "conversation_id": "conv-XX",      // 0..9
  "question": "<user question>",
  "choices": ["...", "...", ..., "..."],   // exactly 10 strings
  "correct_choice_index": 0..9,
  "question_type": "<category>",
  "haystack_sessions": [               // multi-session conversation
      [ {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}, ... ],
      [ ... ],   // session 2
      ...
  ]
}
```

The 10 choices are mapped to letters `(a)..(j)`. The gold answer is
written as `"({letter})"` after our loader maps `correct_choice_index`.

**Aggregate stats**:

- 1,986 questions
- 10 conversations (`conv-00` … `conv-09`)
- avg ~199 questions per conversation; total dialogue across all
  haystack sessions per conversation can exceed 30K tokens.

## 3. Train / Val / Test Split

Uses the **MemSkill convention** (split by sorted `conversation_id`):

| Split | Conversations         | # Questions |
|-------|-----------------------|-------------|
| train | conv-00 … conv-05  (first 6) | ~1,157 |
| val   | conv-06, conv-07   (middle 2) | ~429 (typically unused) |
| test  | conv-08, conv-09   (last 2)   | ~400 |

Conversation-level splitting guarantees the test conversations'
haystack sessions are **never seen** during training. Frozen split
files under [`data/splits/`](./data/splits/):
`train_conv_ids.txt`, `val_conv_ids.txt`, `test_conv_ids.txt`.

## 4. Retrieval Settings

- **Chunking**: each conversation's `haystack_sessions` are flattened
  to a single sequence of user+assistant turn-pair chunks (one chunk
  per consecutive user→assistant exchange). All chunks for a given
  conversation share the same `shared_context_id = "conv-XX"`.
- **Chunk encoder**: base reasoner's last-token hidden (same as
  PersonaMem).
- **Query**: question text → reasoner forward → next-token-step
  hidden as the retrieval vector.
- **Score**: cosine similarity.
- **top-K (`top_z`)**: default **20** (LoCoMo conversations have many
  more chunks than PersonaMem — 9 is too narrow).

## 5. Prompt Template

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

`<all_options>` is the question's 10 options serialized:
```
(a) <choice 0 text>
(b) <choice 1 text>
...
(j) <choice 9 text>
```

The expected `<answer>` is one of `(a)/(b)/.../(j)`.

## 6. Generation / Sampling Hyperparams

| Param | Train rollout | Eval (greedy) |
|-------|---------------|---------------|
| `max_new_tokens`   | 5    | 5    |
| `temperature`      | 1.0  | 0.0  |
| `top_p`            | 1.0  | 1.0  |
| `num_generations`  | 8    | 1    |

## 7. Training Hyperparams

| Param | Value |
|-------|-------|
| optimizer            | AdamW |
| learning rate        | 2e-5  |
| weight_decay         | 0.0   |
| scheduler            | CosineAnnealingLR |
| num_epochs           | 5     |
| batch_size           | 1     |
| GRPO `num_generations` | 8 |
| GRPO `epsilon`       | 0.2   |
| `policy_entropy_coef` | 0.001 |
| `top_z`              | 20    |
| `n_tokens_min / max` | 1 / 20 |
| `max_hidden_cache`   | 20    |
| LoRA `r / alpha / dropout` | 32 / 64 / 0.1 |
| LoRA `target_modules` | `q_proj,k_proj,v_proj,o_proj` |

## 8. Reward Function

Letter match over a–j:

```python
def _compute_reward(prediction, correct_answer):
    p = extract_letter_aj(prediction.strip().lower())
    g = extract_letter_aj(correct_answer.strip().lower())
    return 1.0 if (g and p == g) else 0.0
```

`extract_letter_aj` greps for `(a..j)` first, then a bare a-j with
word boundaries. Binary 0 / 1.

## 9. Persistence & Logging

**Memory artifact** (auto-built):

- Path: `cache/chunk_embeddings_locomo_mc10/{hash}_last1.pt`,
  `{hash}_lastN.pt`
- Cache key: SHA256 of `(chunks_text, model_name, max_chunk_length,
  max_hidden_cache, tokenizer_name)`.
- Build: `bash scripts/build_memory.sh`

**Per-question test log**: same `EvalLogger` schema as defined in
[`common/eval_logger.py`](../common/eval_logger.py), with
`dataset = "LoCoMo-MC10"` in the summary header. Default output:
`results/eval_test.json`.
