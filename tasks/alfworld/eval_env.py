"""
Stage-2 eval: run the trained GRPO pipeline in real textworld + alfworld envs
on valid_seen / valid_unseen games. Reports success rate per split.

Needs the `memskill` conda env for textworld/alfworld imports.
Run from repo root:
    python -m tasks.alfworld.eval_env \
        --eval-data    ./data/alfworld_expert_eval_in_distribution.json \
        --skill-cards  ./data/alfworld_skill_cards.json \
        --checkpoint   ./checkpoints/alfworld/best

The eval-data JSON just gives the list of gamefiles + objectives; the
environment is re-launched per game via textworld.gym. Each step we call
the pipeline's `generate()` — the soft-token prefix and BudgetPolicy come
along for free.
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


class GameTimeout(Exception):
    """Raised by SIGALRM when one game exceeds its wall-clock budget."""


def _alarm_handler(signum, frame):
    raise GameTimeout("game wall-clock timeout")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elasticmem.config import Config
from elasticmem.projector import LatentProjector
from elasticmem.data_utils import encode_chunks
from elasticmem.policy import BudgetPolicy

from tasks.alfworld.data_loader import load_summaries, summary_chunks
from tasks.alfworld.pipeline_alfworld import (
    ALFWorldGRPOPipeline, render_prompt_tail, SYSTEM_INSTRUCTION,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Env helpers
# ----------------------------------------------------------------------

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
        name=f"alfworld-grpo-eval-{uuid.uuid4().hex}", wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


def _unwrap(x):
    """Strip a single-batch wrapper from textworld returns (batch_size=1).

    textworld.gym with batch_size=1 returns each field as a length-1 list/tuple
    (e.g. obs = ('text',), info = [{'admissible_commands': [...]}]). We pull
    the inner element out so downstream code can treat scalars as scalars.
    """
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


def _match_action(pred: str, admissible: List[str]) -> str:
    """Snap the model's free-form output to the nearest admissible command."""
    if not admissible:
        return pred.strip()
    p = pred.strip().lower()
    if not p:
        return admissible[0]
    for cmd in admissible:
        if cmd.strip().lower() == p:
            return cmd
    for cmd in admissible:
        cl = cmd.strip().lower()
        if cl in p or p in cl:
            return cmd
    p_tokens = set(p.split())
    best, best_sc = admissible[0], -1
    for cmd in admissible:
        c_tokens = set(cmd.strip().lower().split())
        sc = len(p_tokens & c_tokens)
        if sc > best_sc:
            best, best_sc = cmd, sc
    return best


# ----------------------------------------------------------------------
# Run one game
# ----------------------------------------------------------------------

@torch.no_grad()
def run_one_game(pipeline, gamefile, objective_hint,
                 chunk_last1, chunk_lastN, max_steps=50, max_new_tokens=16,
                 history_steps=10):
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
            # Truncate history to the last `history_steps` (action, observation)
            # pairs to match the IL training distribution (expert succeeds in
            # ~5-15 steps, so training rarely sees long messy histories).
            keep_lines = max(0, 2 * history_steps)
            recent_history = history_lines[-keep_lines:] if keep_lines else []
            all_options = render_prompt_tail(
                history="\n".join(recent_history),
                current_obs=obs or "",
                admissible=ac,
            )
            pred = pipeline.generate(
                question=objective,
                instruction=SYSTEM_INSTRUCTION,
                all_options=all_options,
                chunk_embs=chunk_last1,
                chunk_hiddens=chunk_lastN,
                max_new_tokens=max_new_tokens,
            )
            action = _match_action(pred, ac)

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


# ----------------------------------------------------------------------
# Checkpoint load
# ----------------------------------------------------------------------

def load_pipeline(args, device):
    cfg = Config()
    cfg.model.model_name = args.model_name
    cfg.retrieval.top_z = args.top_z

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reasoner = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        cache_dir=args.hf_cache_dir,
    ).to(device)
    h = reasoner.config.hidden_size
    cfg.projector.hidden_size = h
    cfg.model.hidden_size = h

    projector = LatentProjector(
        hidden_size=h,
        intermediate_size=cfg.projector.intermediate_size,
        max_queries=cfg.retrieval.n_tokens_max,
        num_heads=8, num_layers=2, ffn_mult=2,
        dropout=cfg.projector.dropout,
    ).to(device).to(torch.bfloat16)
    policy = BudgetPolicy(
        input_dim=h, hidden_size=256, num_heads=4, num_layers=2, dropout=0.1,
        n_choices=cfg.retrieval.n_tokens_max, max_chunks=cfg.retrieval.top_z,
    ).to(device).to(torch.float32)

    lora_cfg = {
        "r": cfg.lora.r, "lora_alpha": cfg.lora.lora_alpha,
        "target_modules": cfg.lora.target_modules,
        "lora_dropout": cfg.lora.lora_dropout,
    }
    pipeline = ALFWorldGRPOPipeline(
        reasoner=reasoner, tokenizer=tokenizer, projector=projector,
        top_z=cfg.retrieval.top_z, n_tokens_min=cfg.retrieval.n_tokens_min,
        n_tokens_max=cfg.retrieval.n_tokens_max,
        max_hidden_cache=cfg.retrieval.max_hidden_cache,
        temperature=cfg.train.temperature, lora_config=lora_cfg, policy=policy,
    )

    ckpt = args.checkpoint
    if ckpt:
        proj_path = os.path.join(ckpt, "projector.pt")
        lora_path = os.path.join(ckpt, "lora_adapter")
        pol_path = os.path.join(ckpt, "policy.pt")
        if os.path.exists(proj_path):
            pipeline.projector.load_state_dict(
                torch.load(proj_path, map_location=device))
            logger.info(f"loaded projector {proj_path}")
        if os.path.isdir(lora_path):
            pipeline.reasoner.load_adapter(lora_path, adapter_name="default")
            logger.info(f"loaded LoRA {lora_path}")
        if os.path.exists(pol_path):
            pipeline.policy.load_state_dict(
                torch.load(pol_path, map_location=device))
            logger.info(f"loaded policy {pol_path}")
    pipeline.eval()
    return pipeline, tokenizer


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--skill-cards", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--hf_cache_dir",
                    default=None)
    ap.add_argument("--cache_dir", default="./cache/chunk_embeddings_alfworld_il")
    ap.add_argument("--top_z", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--per_game_timeout", type=int, default=300,
                    help="Seconds before SIGALRM aborts a single game (default 5min)")
    ap.add_argument("--history_steps", type=int, default=10,
                    help="Keep last N (action, obs) pairs in prompt history")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="./results/eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, tokenizer = load_pipeline(args, device)

    summaries = load_summaries(args.skill_cards)
    chunks_text = summary_chunks(summaries)
    logger.info(f"Memory: {len(chunks_text)} skill cards")
    emb = encode_chunks(
        pipeline.reasoner, tokenizer, chunks_text,
        max_length=Config().data.max_chunk_length,
        max_hidden_cache=Config().retrieval.max_hidden_cache,
        cache_dir=args.cache_dir, device=device,
    )
    chunk_last1 = emb["last1"].to(device)
    chunk_lastN = emb["lastN"].to(device)

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
    logger.info(f"Running on {len(games)} games from {args.eval_data}")

    # Per-game wall-clock timeout via SIGALRM. textworld occasionally hangs
    # in env.reset() / env.step() on certain games; without this the whole
    # eval stalls forever. Resolution is in whole seconds.
    signal.signal(signal.SIGALRM, _alarm_handler)
    per_game_timeout = max(60, int(args.per_game_timeout))

    results, per_type = [], {}
    for g in tqdm(games, desc="Eval envs"):
        signal.alarm(per_game_timeout)
        try:
            r = run_one_game(
                pipeline=pipeline,
                gamefile=g["gamefile"],
                objective_hint=g["objective"],
                chunk_last1=chunk_last1,
                chunk_lastN=chunk_lastN,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                history_steps=args.history_steps,
            )
        except GameTimeout:
            logger.warning(f"game {g['gamefile']} timed out after {per_game_timeout}s")
            r = {"gamefile": g["gamefile"], "success": False, "error": "timeout"}
        except Exception as e:
            logger.warning(f"game {g['gamefile']} errored: {e}")
            r = {"gamefile": g["gamefile"], "success": False, "error": str(e)}
        finally:
            signal.alarm(0)
        r["task_type"] = g["task_type"]
        r["objective"] = g["objective"]
        results.append(r)
        per_type.setdefault(g["task_type"], []).append(
            1.0 if r.get("success") else 0.0
        )

        # Incremental dump every 10 games so partial results are inspectable.
        if (len(results) % 10) == 0:
            tmp = {
                "summary": {
                    "overall_success": sum(1 for r in results if r.get("success")) / max(len(results), 1),
                    "per_type_success": {k: sum(v) / max(len(v), 1) for k, v in per_type.items()},
                    "num_games": len(results),
                    "in_progress": True,
                },
                "results": results,
            }
            with open(args.output, "w") as f:
                json.dump(tmp, f, indent=2)

    overall = sum(1 for r in results if r.get("success")) / max(len(results), 1)
    per_type_rates = {k: sum(v) / max(len(v), 1) for k, v in per_type.items()}
    summary = {
        "overall_success": overall,
        "per_type_success": per_type_rates,
        "num_games": len(results),
    }
    logger.info(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
