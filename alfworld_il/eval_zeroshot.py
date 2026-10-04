"""
Zero-shot ALFWorld baseline: pure Qwen2.5-3B-Instruct generation given the
same prompt structure used by our trained pipeline (objective, last-N history,
current obs, admissible, "Action:"), but WITHOUT projector/LoRA/skill cards.

Goal: isolate how much of our 32%/22% (seen/unseen) success rate is
attributable to IL training vs simply Qwen reading admissible+obs.
"""
import os
import re
import sys
import json
import signal
import uuid
import argparse
import logging
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SYSTEM_INSTRUCTION = (
    "You are controlling a text-based ALFWorld environment. "
    "Choose the NEXT action as ONE admissible command string. "
    "Output only the command, copied verbatim from the admissible list."
)


def render_prompt_tail(history, current_obs, admissible):
    parts = []
    if history:
        parts += ["Interaction history so far:", history.strip(), ""]
    parts += ["Current observation:", (current_obs or "").strip()]
    if admissible:
        parts += ["", "Admissible actions (choose exactly ONE, copy verbatim):"]
        parts += [f"- {a}" for a in admissible]
    parts += ["", "Action:"]
    return "\n".join(parts)


def build_prompt(objective: str, prompt_tail: str) -> str:
    """Render as a single instruction-style prompt for greedy decoding."""
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Goal: {objective}\n\n"
        f"{prompt_tail}"
    )


_ACTION_PREFIX_RE = re.compile(r'^\s*action\s*[:=\-]\s*', re.IGNORECASE)


def _first_line(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = _ACTION_PREFIX_RE.sub('', text.strip())
    if not t:
        return ""
    line = t.splitlines()[0]
    return line.strip().strip('"').strip("'").strip('`')


def _match_action(pred: str, admissible: List[str]) -> str:
    """MemSkill-style snap to nearest admissible command."""
    line = _first_line(pred)
    if not line and admissible:
        return admissible[0]
    p = line.lower()
    if admissible:
        for cmd in admissible:
            if str(cmd).strip().lower() == p:
                return cmd
        for cmd in admissible:
            if str(cmd).strip().lower() in pred.lower():
                return cmd
        return admissible[0]
    return line or "look"


def _unwrap(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    if isinstance(x, list) and x and isinstance(x[0], list):
        return x[0]
    return x


def _extract_objective(first_obs: str) -> str:
    m = re.search(r"Your task is to:\s*(.+)", first_obs or "",
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    first_line = m.group(1).splitlines()[0].strip()
    return first_line.rstrip(".")


# ─── Env helpers ────────────────────────────────────────────────────

def _make_env(gamefile: str, max_steps: int):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos,
    )
    req = textworld.EnvInfos(
        feedback=True, description=True, inventory=True,
        command_templates=True, intermediate_reward=True, location=True,
        objective=True, admissible_commands=True,
        extras=["gamefile", "expert_plan"],
    )
    wrappers = [
        AlfredDemangler(),
        AlfredInfos,
        AlfredExpert(expert_type=AlfredExpertType.PLANNER),
    ]
    env_id = textworld.gym.register_games(
        [gamefile], req, batch_size=1, auto_reset=False,
        max_episode_steps=max_steps, asynchronous=False,
        name=f"alfworld-zs-{uuid.uuid4().hex}", wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


# ─── Timeout ────────────────────────────────────────────────────────

class GameTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise GameTimeout("game wall-clock timeout")


# ─── Run one game ───────────────────────────────────────────────────

@torch.no_grad()
def run_one_game(model, tokenizer, gamefile, objective_hint,
                 max_steps=30, max_new_tokens=16, history_steps=5,
                 device="cuda"):
    env = _make_env(gamefile, max_steps)
    try:
        obs, info = env.reset()
        obs = _unwrap(obs)
        info = info if isinstance(info, dict) else _unwrap(info)
        first_obs = obs or ""
        objective = (objective_hint or _extract_objective(first_obs)
                     or "complete the task")

        history_lines: List[str] = []
        done, total_reward = False, 0.0
        steps_taken = 0
        for t in range(max_steps):
            ac = _unwrap(info.get("admissible_commands")) or []
            keep = max(0, 2 * history_steps)
            recent = history_lines[-keep:] if keep else []
            tail = render_prompt_tail(
                history="\n".join(recent),
                current_obs=obs or "",
                admissible=ac,
            )
            prompt = build_prompt(objective, tail)

            ids = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0, ids.input_ids.size(1):], skip_special_tokens=True
            )
            action = _match_action(gen, ac)

            obs_next, score, done_b, info = env.step([action])
            obs_next = _unwrap(obs_next); info = _unwrap(info)
            score = _unwrap(score); done_flag = bool(_unwrap(done_b))
            total_reward += float(score or 0.0)

            history_lines.append(f"ACTION: {action}")
            history_lines.append(f"OBSERVATION: {str(obs_next or '').strip()}")
            obs = obs_next
            steps_taken = t + 1
            if done_flag:
                done = True
                break

        success = bool(done and total_reward >= 1.0)
        return {
            "gamefile": gamefile,
            "success": success,
            "steps_taken": steps_taken,
            "total_reward": total_reward,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


# ─── Main ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--hf_cache_dir",
                    default=None)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--per_game_timeout", type=int, default=240)
    ap.add_argument("--history_steps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        cache_dir=args.hf_cache_dir,
    ).to(device)
    model.eval()

    with open(args.eval_data) as f:
        data = json.load(f)
    games: List[Dict] = []
    for task_type, by_gf in data.items():
        if not isinstance(by_gf, dict):
            continue
        for gf, entry in by_gf.items():
            if not isinstance(entry, dict) or entry.get("error"):
                continue
            games.append({
                "task_type": task_type, "gamefile": gf,
                "objective": entry.get("objective") or "",
            })
    if args.limit:
        games = games[: args.limit]
    logger.info(f"ZeroShot {args.model_name} on {len(games)} games "
                f"(history_steps={args.history_steps})")

    signal.signal(signal.SIGALRM, _alarm_handler)
    per_game_timeout = max(60, int(args.per_game_timeout))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results, per_type = [], {}
    for g in tqdm(games, desc="ZS Eval"):
        signal.alarm(per_game_timeout)
        try:
            r = run_one_game(
                model=model, tokenizer=tokenizer,
                gamefile=g["gamefile"], objective_hint=g["objective"],
                max_steps=args.max_steps, max_new_tokens=args.max_new_tokens,
                history_steps=args.history_steps, device=device,
            )
        except GameTimeout:
            logger.warning(f"timeout {g['gamefile']}")
            r = {"gamefile": g["gamefile"], "success": False, "error": "timeout"}
        except Exception as e:
            logger.warning(f"errored {g['gamefile']}: {e}")
            r = {"gamefile": g["gamefile"], "success": False, "error": str(e)}
        finally:
            signal.alarm(0)
        r["task_type"] = g["task_type"]
        r["objective"] = g["objective"]
        results.append(r)
        per_type.setdefault(g["task_type"], []).append(
            1.0 if r.get("success") else 0.0
        )
        if (len(results) % 10) == 0:
            tmp = {
                "summary": {
                    "overall_success": sum(1 for r in results if r.get("success")) / len(results),
                    "per_type_success": {k: sum(v) / len(v) for k, v in per_type.items()},
                    "num_games": len(results),
                    "in_progress": True,
                },
                "results": results,
            }
            with open(args.output, "w") as f:
                json.dump(tmp, f, indent=2)

    overall = sum(1 for r in results if r.get("success")) / max(len(results), 1)
    summary = {
        "overall_success": overall,
        "per_type_success": {k: sum(v) / len(v) for k, v in per_type.items()},
        "num_games": len(results),
    }
    logger.info(json.dumps(summary, indent=2))
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
