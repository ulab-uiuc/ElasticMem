"""
ALFWorld IL data loader.

Reads the offline expert-trajectory JSON produced by
`MemSkill/alfworld_replay.py` and produces two pools:

  - memory_entries  (M):  used to build the retrieval memory bank
                          (will be summarised by an LLM separately).
  - il_entries      (T):  used as IL training material; each trajectory
                          is later expanded into per-step samples.

Input JSON schema (from replay):
  { task_type: {
      gamefile_path: {
        "traj_path":       str,
        "first_observation": str,
        "objective":       str,
        "total_reward":    float,
        "steps": [
          {step, action, observation, reward, done, expert_plan}, ...
        ],
        "trajectory":      str       # pre-joined text
      }, ...
    }, ...
  }
"""
import os
import json
import random
from typing import Dict, List, Tuple


def _flatten_successful_entries(data: Dict) -> List[Tuple[str, str, Dict]]:
    out = []
    for task_type, games in data.items():
        if not isinstance(games, dict):
            continue
        for gf, entry in games.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("total_reward", 0.0) != 1.0:
                continue
            steps = entry.get("steps")
            if not isinstance(steps, list) or len(steps) < 2:
                continue
            if entry.get("error"):
                continue
            out.append((task_type, gf, entry))
    return out


def load_alfworld_il(
    offline_path: str,
    memory_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, Dict]], List[Tuple[str, str, Dict]]]:
    """Split successful expert trajectories into memory pool (M) and IL pool (T)."""
    with open(offline_path, "r") as f:
        data = json.load(f)
    entries = _flatten_successful_entries(data)
    rng = random.Random(seed)
    rng.shuffle(entries)
    cut = int(len(entries) * memory_ratio)
    return entries[:cut], entries[cut:]


def expand_il_samples(
    il_entries: List[Tuple[str, str, Dict]],
    history_steps: int = 5,
    max_history_chars: int = 4000,
) -> List[Dict]:
    """
    Turn each expert trajectory into per-step IL samples:

      step_index t  →  predict action taken at t given
                       (objective, last-`history_steps` (action, obs) pairs,
                       observation at t-1, admissible at t-1).

    Match deployment: eval_env keeps last `history_steps` (action, obs) pairs.
    """
    samples: List[Dict] = []
    for task_type, gf, entry in il_entries:
        steps = entry["steps"]
        objective = entry.get("objective") or ""
        first_obs = entry.get("first_observation") or ""

        for t in range(1, len(steps)):
            cur = steps[t]
            action = cur.get("action")
            if not isinstance(action, str) or not action.strip():
                continue

            # Take only the last `history_steps` (action, obs) pairs,
            # mirroring the eval-time history truncation.
            start = max(1, t - history_steps)
            lines: List[str] = []
            for prev in steps[start:t]:
                pa = prev.get("action")
                po = prev.get("observation") or ""
                if isinstance(pa, str) and pa.strip():
                    lines.append(f"ACTION: {pa.strip()}")
                if isinstance(po, str) and po.strip():
                    lines.append(f"OBSERVATION: {po.strip()}")
            history = "\n".join(lines)
            if len(history) > max_history_chars:
                history = history[-max_history_chars:]

            prev_step = steps[t - 1]
            current_obs = str(prev_step.get("observation") or first_obs).strip()
            # Admissible at the state we're deciding FROM = admissible stored
            # on step t-1 (v2 replay). v1 replays without this field get [].
            ac = prev_step.get("admissible_commands") or []
            if not isinstance(ac, list):
                ac = []
            # Drop admissible entries that are just "inventory" / empty.
            admissible = [str(a).strip() for a in ac if isinstance(a, str) and a.strip()]

            samples.append({
                "task_type": task_type,
                "gamefile": gf,
                "objective": objective,
                "first_observation": first_obs,
                "history": history,
                "current_obs": current_obs,
                "admissible": admissible,
                "expert_action": action.strip(),
                "step_idx": t,
            })
    return samples


def load_summaries(summary_path: str) -> List[Dict]:
    """Load the skill-card summaries produced by alfworld_il/summarize.py.

    Each entry: {task_type, gamefile, objective, summary}
    """
    with open(summary_path, "r") as f:
        return json.load(f)


def summary_chunks(summaries: List[Dict]) -> List[str]:
    """Flatten skill cards into retrievable chunk texts (with task_type prefix)."""
    return [
        f"[{s.get('task_type','unknown')}] {s.get('summary','').strip()}"
        for s in summaries
        if isinstance(s, dict) and s.get("summary")
    ]
