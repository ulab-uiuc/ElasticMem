# data/

Put downloaded datasets here. Expected files:

| File | Used by |
|---|---|
| `questions_32k.csv`, `shared_contexts_32k.jsonl` | PersonaMem-32K |
| `questions_128k.csv`, `shared_contexts_128k.jsonl` | PersonaMem-128K |
| `alfworld_train_offline.json` | ALFWorld (expert trajectories) |
| `alfworld_expert_eval_in_distribution.json` | ALFWorld seen |
| `alfworld_expert_eval_out_of_distribution.json` | ALFWorld unseen |
| `alfworld_skill_cards.json` | ALFWorld memory bank (from `scripts/build_skill_cards.sh`) |

LoCoMo-MC10 and LongMemEval-MC10 are loaded from the Hugging Face cache.

`splits/` holds the frozen train / val / test partitions.
