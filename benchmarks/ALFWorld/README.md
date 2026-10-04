# ALFWorld

Interactive text-adventure benchmark — agent must read observations
and emit action commands until the task is solved or `max_steps`
is exhausted. We treat ALFWorld as both an **imitation-learning
training source** (per-step (state, expert_action) supervision derived
from expert trajectories) and an **environment-success deployment
target** (full episode rollout in the live `textworld.gym` env).

## 1. Dataset Source

Three layers — only Layer 0 must come from the official ALFWorld
release; Layers 1 and 2 are produced by scripts shipped here.

### Layer 0 — Raw ALFWorld game files
- Source: official `alfworld` Python package; downloaded into
  `$ALFWORLD_DATA = ~/.cache/alfworld/json_2.1.1/`.
- ~2.3 GB total (NOT redistributed in this repo). Three splits:

  | Split | Game count |
  |-------|-----------:|
  | `train/`        | ~6,374 (post-filtering: ~3,553 solvable + 6 task types) |
  | `valid_seen/`   | 140    (same room layouts as train, new tasks) |
  | `valid_unseen/` | 134    (entirely new room layouts)             |

- 6 task types:
  `pick_and_place_simple`,
  `pick_clean_then_place_in_recep`,
  `pick_heat_then_place_in_recep`,
  `pick_cool_then_place_in_recep`,
  `look_at_obj_in_light`,
  `pick_two_obj_and_place`.
- Setup: follow the official ALFWorld instructions, set
  `$ALFWORLD_DATA`, then `pip install alfworld textworld`. See
  [`scripts/download_data.sh`](./scripts/download_data.sh).

### Layer 1 — Expert trajectory pool (replay JSON v2)
- Three files (shipped under [`data/`](./data)):
  - `alfworld_train_offline_v2.json`           — 42 MB, ~3,553 train games
  - `alfworld_expert_eval_in_distribution_v2.json`     — 1.7 MB, 140 valid_seen games
  - `alfworld_expert_eval_out_of_distribution_v2.json` — 1.5 MB, 134 valid_unseen games
- Produced by replaying each Layer-0 game with `AlfredExpert(PLANNER)`
  inside `textworld.gym`, recording every step. See
  [`scripts/build_replay.sh`](./scripts/build_replay.sh).
- Each `_v2.json` is a dict `{task_type: {gamefile_path: entry}}`
  where `entry` includes:
  ```python
  {
    "first_observation": "...You are in a room... Your task is to: ...",
    "objective":         "<one-line task goal>",
    "total_reward":      1.0,
    "steps": [
      {"step": 0, "action": null, "observation": <first_obs>,
       "reward": 0, "done": false,
       "admissible_commands": ["go to bed 1", "go to drawer 1", ...],
       "expert_plan": ["go to ...", ...]},
      {"step": 1, "action": "go to toilet 1", "observation": "...",
       "reward": 0, "done": false,
       "admissible_commands": [...]},
      ...
    ],
    "trajectory": "<pre-joined human-readable text>"
  }
  ```
- Layer 1 is **the IL ground truth source**, not the retrieval bank.

### Layer 2 — Skill cards (NOT shipped — regenerate locally)
- Each card is a 200-300 word *procedural summary* of one Layer-1
  trajectory from the M pool (the 80% reserved for memory). Reads like
  a recipe:
  ```
  Task type: pick_clean_then_place_in_recep
  Objective pattern: clean an object then place it on a receptacle
  1. Locate the target object (search common containers if not visible).
  2. Pick it up.
  3. Walk to the sink and turn it on.
  4. Wash the object, then walk to the target receptacle.
  5. Place the object.
  ```
- ~2,832 cards total (one per M-pool trajectory).
- Generated offline by an LLM via [`scripts/build_skill_cards.sh`](./scripts/build_skill_cards.sh)
  — the script reads each Layer-1 trajectory and prompts the LLM with
  a procedural-summary template; output written to
  `data/alfworld_skill_cards.json`.
- **This is the retrieval bank** at training and deployment.

## 2. Format & Schema

(Full per-entry schema given inline above for Layers 1 and 2.)

For Layer 2 (`alfworld_skill_cards.json`):

```python
[
  {
    "task_type":  "pick_and_place_simple",
    "gamefile":   "/.../game.tw-pddl",
    "objective":  "put a soapbottle in toilet",
    "summary":    "<200-300 word procedural summary>"
  },
  ...
]
```

Skill-card retrieval text = `f"[{task_type}] {summary}"`.

## 3. Train / Val / Test Split

Three layers of splits, all by **trajectory** (i.e., game), never by
step:

### Layer 1 split (in-house, deterministic)
`load_alfworld_il(memory_ratio=0.8, seed=42)` shuffles the
`train_offline_v2.json` games and splits:

| Pool | exact count | Purpose |
|------|---:|---------|
| **M** (memory) | **2,832** | Layer-2 LLM skill-card source |
| **T** (IL train) | **709**  | per-step IL supervision (each trajectory is later expanded into multiple `(state, expert_action)` samples) |

**Frozen pool index files** under [`data/splits/`](./data/splits/):
`M_pool_gamefiles.txt`, `T_pool_gamefiles.txt`,
`valid_seen_gamefiles.txt`, `valid_unseen_gamefiles.txt` — one
gamefile path per line.

### Layer 2 split (training-time held-out for EM monitoring)
After expanding T into per-step samples (`expand_il_samples(...)`),
the script reserves the first 5% as a held-out eval set used only for
periodic string-EM checks during training; the remaining 95% are the
GRPO training samples.

### Layer 3 split (env-success deployment eval)
Two **completely independent** game pools:

| Pool | # games | Source |
|------|--------:|--------|
| valid_seen   | 140 | `alfworld_expert_eval_in_distribution_v2.json`     |
| valid_unseen | 134 | `alfworld_expert_eval_out_of_distribution_v2.json` |

The valid_seen game files were *not* in train, but their **room
layouts** were. valid_unseen games are in **brand new rooms**. Neither
pool overlaps with M, T, or the held-out EM samples.

## 4. Retrieval Settings

- **Chunk = one Layer-2 skill card** (`f"[{task_type}] {summary}"`).
- **Chunk encoder**: base reasoner's last-token hidden (same protocol
  as the QA datasets).
- **Query**: the task's `objective` string (e.g., "put a soapbottle in
  toilet") forwarded through the reasoner once per episode.
- **Score**: cosine similarity against all ~2,832 cached card
  embeddings.
- **top-K (`top_z`)**: default **10**.
- **Frequency**: **per-episode** (compute once at `env.reset()`,
  reuse for every step in that episode). This avoids redundant
  encoder calls — the objective never changes mid-episode.

## 5. Prompt Template

Two prompt structures: **training-time prompt** (uses Layer-1 expert
history per sample) and **deployment-time prompt** (uses the agent's
own history accumulated by `env.step`).

Both share the same scaffolding:

```
[ <objective>  ← question field ]
[ <retrieval gate token, learned> ]
[ <retrieved-skill-card representations injected here, top-10> ]
[ <SYSTEM_INSTRUCTION>\n<all_options> ]
[ <action>  ← model writes this ]
```

`<SYSTEM_INSTRUCTION>` (literal):

```
You are controlling a text-based ALFWorld environment. Choose the
NEXT action as ONE admissible command string. Output only the
command, copied verbatim from the admissible list.
```

`<all_options>` is built by `render_prompt_tail(history, current_obs,
admissible)`:

```
Interaction history so far:
<last N (action, obs) pairs concatenated as
 ACTION: <a>\nOBSERVATION: <obs> on alternating lines>

Current observation:
<current obs string>

Admissible actions (choose exactly ONE, copy verbatim):
- <admissible cmd 1>
- <admissible cmd 2>
- ...

Action:
```

The expected `<action>` is the next text command (a single line).

**History truncation parameter `history_steps`**:
- Training data loader uses `expand_il_samples(history_steps=N)` to
  keep only the last `N` (action, obs) pairs of the expert trajectory
  in `<history>`.
- Deployment script (`eval_env.py`) keeps the last `N` (action, obs)
  pairs of the agent's own trajectory.
- `N` MUST be the same at train and eval to avoid distribution shift.
  Values used: **5** (default) and **10**.

## 6. Generation / Sampling Hyperparams

| Param | Train rollout | Deployment (env eval, greedy) |
|-------|---------------|-------------------------------|
| `max_new_tokens`   | 16   | 16   |
| `temperature`      | 1.0  | 0.0  |
| `top_p`            | 1.0  | 1.0  |
| `num_generations`  | 4 (memory-safe) or 8 | 1 |
| sampling           | multinomial | greedy |

Deployment-only knobs (live env interaction):

| Param | Default |
|-------|--------:|
| `max_steps` (per game)        | 30  |
| `per_game_timeout` (seconds)  | 240 (SIGALRM hard-aborts a game) |

## 7. Training Hyperparams

| Param | Value |
|-------|-------|
| optimizer            | AdamW |
| learning rate        | 2e-5  |
| weight_decay         | 0.0   |
| scheduler            | CosineAnnealingLR |
| num_epochs           | 3     |
| batch_size           | 1     |
| GRPO `num_generations` | 4   |
| GRPO `epsilon`       | 0.2   |
| `policy_entropy_coef` | 0.001 |
| `top_z`              | 10    |
| `n_tokens_min / max` | 1 / 20 |
| `max_hidden_cache`   | 20    |
| `history_steps`      | 5 (must match deployment) |
| LoRA `r / alpha / dropout` | 32 / 64 / 0.1 |
| LoRA `target_modules` | `q_proj,k_proj,v_proj,o_proj` |

## 8. Reward Function

Two related rewards, depending on the training recipe:

### Recipe A — string-tiered match (no admissible-snap)
For each generated action vs the expert action:
```
1.0 — first-line equals expert (case/whitespace normalised)
0.7 — expert action is a token-prefix of the first line
0.5 * ratio — token-LCP ratio of first-line vs expert
0.0 — otherwise
```

### Recipe B — snap-to-admissible + strict (deployment-aligned)
At reward time the model's first-line generation is first snapped to
the nearest admissible command (exact / substring / token-overlap
fallback, identical to deployment-time `_match_action`). Then:
```
1.0 — snap result equals expert action (after normalise)
0.0 — otherwise
```
Recipe B aligns the training reward with what actually gets executed
at deployment (`env.step([snapped_action])`). The `correct_answer`
field is JSON-encoded `{"gold": expert_action, "admissible": [...]}`
so the reward function can replay the snap step.

For **deployment evaluation** the metric is per-game **task success**
(the env reports `total_reward >= 1.0` at termination), aggregated as
the success rate per task type and overall — independent of either
recipe above.

## 9. Persistence & Logging

**Memory artifacts** (must be saved):

| Artifact | Path | Producer |
|----------|------|----------|
| Layer-1 replay JSON v2 (3 splits)  | `data/alfworld_*_v2.json`           | `scripts/build_replay.sh` (one-time) |
| Layer-2 skill cards                | `data/alfworld_skill_cards.json`    | `scripts/build_skill_cards.sh` (one-time) |
| Skill-card embedding cache         | `cache/chunk_embeddings_alfworld_il/...` | auto on first train/eval |

**Per-question (per-game) test log** — written by `EvalLogger.dump()`
(`common/eval_logger.py`):

```json
{
  "summary": {
    "dataset": "ALFWorld",
    "method":  "<your method>",
    "num_samples": 134,
    "overall_score": 0.41,
    "per_task_type": {
      "pick_and_place_simple": 0.375,
      "pick_clean_then_place_in_recep": 0.323,
      ...
    },
    "tokens": {"prompt_total": ..., "completion_total": ...,
               "prompt_avg": ..., "completion_avg": ...,
               "total_total": ...}
  },
  "results": [
    {
      "question_id":  "<gamefile_path>",
      "input": {
        "objective":           "put a soapbottle in toilet",
        "history":             "ACTION: ... OBSERVATION: ... ...",
        "current_obs":         "<current observation>",
        "admissible":          ["go to bed 1", ...],
        "retrieved_chunk_ids": [12, 47, 88, ...],
        "prompt_full":         "<full prompt at the LAST step>"
      },
      "output": {
        "raw_text":       "go to toilet 1\nACTION: ...",   // model raw
        "parsed_answer":  "go to toilet 1",                 // first line
        "snapped_action": "go to toilet 1"                  // executed
      },
      "score": 1.0,                            // 1 = task success, 0 = fail
      "tokens": {"prompt": 4128, "completion": 9, "total": 4137},
      "extra":  {"task_type": "pick_and_place_simple",
                 "steps_taken": 4, "total_reward": 1.0}
    },
    ...
  ]
}
```

For ALFWorld the `tokens` field aggregates **across all steps of the
episode** (sum of every per-step generate's prompt + completion
tokens). The `input.prompt_full` field stores the prompt at the last
step (for inspection); intermediate-step prompts can be saved to a
`steps[]` list inside `extra` if you want fully detailed traces.

Default output: `results/eval_seen.json` and
`results/eval_unseen.json`.
