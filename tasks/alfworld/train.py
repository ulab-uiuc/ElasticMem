"""
ALFWorld GRPO training.

Per-step (state, expert_action) samples from expert replays serve as the
'question / correct_answer' of the GRPO rollout. The parent pipeline handles
rollout (num_generations trajectories), PPO clipping, BudgetPolicy updates,
and soft-token injection. We only plug in our prompt renderer and a binary
exact-match reward.

Run from repo root:
    python -u -m tasks.alfworld.train \
        --offline-data ./data/alfworld_train_offline.json \
        --skill-cards  ./data/alfworld_skill_cards.json \
        --num_epochs 3 --num_generations 8 --top_z 10 --lr 2e-5
"""
import os
import sys
import json
import random
import argparse
import logging

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elasticmem.config import Config
from elasticmem.projector import LatentProjector
from elasticmem.data_utils import encode_chunks
from elasticmem.policy import BudgetPolicy

from tasks.alfworld.data_loader import (
    load_alfworld_il, expand_il_samples, load_summaries, summary_chunks,
)
from tasks.alfworld.pipeline_alfworld import (
    ALFWorldGRPOPipeline, render_prompt_tail, SYSTEM_INSTRUCTION,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Sample → four-field prompt
# ----------------------------------------------------------------------

def sample_to_prompt(s):
    """Build (question, instruction, all_options, correct_answer) for GRPO.

    correct_answer is JSON-encoded so the reward function can apply the same
    snap-to-admissible logic that eval_env uses at deployment time.
    """
    return {
        "question":       s["objective"],
        "instruction":    SYSTEM_INSTRUCTION,
        "all_options":    render_prompt_tail(
                              history=s["history"],
                              current_obs=s["current_obs"],
                              admissible=s.get("admissible") or None,
                          ),
        "correct_answer": json.dumps({
            "gold":       s["expert_action"],
            "admissible": s.get("admissible") or [],
        }),
    }


# ----------------------------------------------------------------------
# Epoch loop
# ----------------------------------------------------------------------

def train_one_epoch(pipeline, samples, chunk_last1, chunk_lastN, optimizer,
                    cfg, device, epoch, use_wandb,
                    eval_samples, best_ref, eval_every=500,
                    max_new_tokens=16, max_steps=None):
    pipeline.train()
    indices = list(range(len(samples)))
    random.shuffle(indices)
    if max_steps is not None:
        indices = indices[:max_steps]

    total_loss, total_reward, n_steps = 0.0, 0.0, 0
    optimizer.zero_grad()

    for step_i, idx in enumerate(tqdm(indices, desc=f"GRPO epoch {epoch}")):
        s = samples[idx]
        p = sample_to_prompt(s)

        trajectories = pipeline.grpo_rollout(
            question=p["question"],
            correct_answer=p["correct_answer"],
            instruction=p["instruction"],
            all_options=p["all_options"],
            chunk_embs=chunk_last1,
            chunk_hiddens=chunk_lastN,
            num_generations=cfg.train.num_generations,
            max_new_tokens=max_new_tokens,
        )
        old_logprobs = pipeline.compute_old_logprobs(
            question=p["question"],
            instruction=p["instruction"],
            all_options=p["all_options"],
            chunk_hiddens_all=chunk_lastN,
            trajectories=trajectories,
        )

        step_loss, last_mean_reward = 0.0, 0.0
        for _ in range(cfg.train.num_grpo_iters):
            result = pipeline.grpo_loss(
                question=p["question"],
                instruction=p["instruction"],
                all_options=p["all_options"],
                chunk_hiddens_all=chunk_lastN,
                trajectories=trajectories,
                old_logprobs=old_logprobs,
                epsilon=cfg.train.epsilon,
                policy_entropy_coef=0.001,
                aux_lambda=0.0,
                aux_temp=1.0,
            )
            loss = result["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf at step {step_i}, skipping")
                optimizer.zero_grad()
                break
            loss.backward()
            step_loss += loss.item()
            last_mean_reward = result["mean_reward"]
            torch.nn.utils.clip_grad_norm_(
                pipeline.get_trainable_parameters(), 1.0
            )
            for pr in pipeline.get_trainable_parameters():
                if pr.grad is not None and (
                    torch.isnan(pr.grad).any() or torch.isinf(pr.grad).any()
                ):
                    pr.grad.zero_()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += step_loss / max(cfg.train.num_grpo_iters, 1)
        total_reward += last_mean_reward
        n_steps += 1
        if use_wandb:
            wandb.log({
                "train/loss": step_loss / max(cfg.train.num_grpo_iters, 1),
                "train/mean_reward": last_mean_reward,
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        if eval_every and (step_i + 1) % eval_every == 0 and eval_samples:
            em = eval_string_em(pipeline, eval_samples, chunk_last1, chunk_lastN,
                                max_new_tokens=max_new_tokens, limit=200)
            logger.info(f"  [step {step_i+1}] string-EM: {em:.4f}")
            if use_wandb:
                wandb.log({"eval/string_em": em})
            if em > best_ref[0]:
                best_ref[0] = em
                save_checkpoint(pipeline, cfg.train.save_dir, "best")
            pipeline.train()

    return total_loss / max(n_steps, 1), total_reward / max(n_steps, 1)


@torch.no_grad()
def eval_string_em(pipeline, samples, chunk_last1, chunk_lastN,
                   max_new_tokens=16, limit=200):
    pipeline.eval()
    hits, total = 0, 0
    for s in tqdm(samples[:limit], desc="IL-EM"):
        p = sample_to_prompt(s)
        pred = pipeline.generate(
            question=p["question"],
            instruction=p["instruction"],
            all_options=p["all_options"],
            chunk_embs=chunk_last1,
            chunk_hiddens=chunk_lastN,
            max_new_tokens=max_new_tokens,
        )
        if pipeline._compute_reward(pred, p["correct_answer"]) > 0:
            hits += 1
        total += 1
    return hits / max(total, 1)


def save_checkpoint(pipeline, save_dir, name):
    path = os.path.join(save_dir, name)
    os.makedirs(path, exist_ok=True)
    torch.save(pipeline.projector.state_dict(),
               os.path.join(path, "projector.pt"))
    pipeline.reasoner.save_pretrained(os.path.join(path, "lora_adapter"))
    if pipeline.policy is not None:
        torch.save(pipeline.policy.state_dict(),
                   os.path.join(path, "policy.pt"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-data", required=True)
    ap.add_argument("--skill-cards", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--hf_cache_dir",
                    default=None)

    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--top_z", type=int, default=10)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--history_steps", type=int, default=5,
                    help="Last N (action, obs) pairs in training history "
                         "(must match eval_env --history_steps for distribution alignment)")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--memory_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_dir", default="./checkpoints/alfworld")
    ap.add_argument("--cache_dir", default="./cache/chunk_embeddings_alfworld_il")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device={device}  model={args.model_name}")

    cfg = Config()
    cfg.model.model_name = args.model_name
    cfg.train.lr = args.lr
    cfg.train.num_epochs = args.num_epochs
    cfg.train.num_generations = args.num_generations
    cfg.retrieval.top_z = args.top_z
    cfg.data.cache_dir = args.cache_dir
    cfg.train.save_dir = args.save_dir

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

    # ---- data ----
    _, il_entries = load_alfworld_il(
        args.offline_data, memory_ratio=args.memory_ratio, seed=args.seed
    )
    il_samples = expand_il_samples(il_entries, history_steps=args.history_steps)
    random.shuffle(il_samples)
    cut = max(1, int(len(il_samples) * 0.05))
    eval_samples = il_samples[:cut]
    train_samples = il_samples[cut:]
    logger.info(
        f"IL samples: train={len(train_samples)} eval={len(eval_samples)} "
        f"(from {len(il_entries)} trajectories)"
    )

    summaries = load_summaries(args.skill_cards)
    chunks_text = summary_chunks(summaries)
    logger.info(f"Memory bank: {len(chunks_text)} skill cards")
    emb = encode_chunks(
        reasoner, tokenizer, chunks_text,
        max_length=cfg.data.max_chunk_length,
        max_hidden_cache=cfg.retrieval.max_hidden_cache,
        cache_dir=cfg.data.cache_dir, device=device,
    )
    chunk_last1 = emb["last1"].to(device)
    chunk_lastN = emb["lastN"].to(device)

    # ---- model + BudgetPolicy (same shape as locomo_mc10) ----
    projector = LatentProjector(
        hidden_size=cfg.projector.hidden_size,
        intermediate_size=cfg.projector.intermediate_size,
        max_queries=cfg.retrieval.n_tokens_max,
        num_heads=8, num_layers=2, ffn_mult=2,
        dropout=cfg.projector.dropout,
    ).to(device).to(torch.bfloat16)
    policy = BudgetPolicy(
        input_dim=cfg.projector.hidden_size, hidden_size=256, num_heads=4,
        num_layers=2, dropout=0.1,
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

    optimizer = AdamW(pipeline.get_trainable_parameters(), lr=args.lr,
                      weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    try:
        wandb.init(
            project="latent-chunk-mem",
            name=f"alfworld_grpo_z{args.top_z}_G{args.num_generations}_lr{args.lr}",
            config={"dataset": "alfworld_il",
                    "top_z": args.top_z, "num_generations": args.num_generations,
                    "lr": args.lr, "model": args.model_name},
        )
        use_wandb = True
    except Exception as e:
        logger.warning(f"wandb init failed: {e}")
        use_wandb = False

    os.makedirs(args.save_dir, exist_ok=True)
    best_ref = [0.0]
    for epoch in range(1, args.num_epochs + 1):
        avg_loss, avg_reward = train_one_epoch(
            pipeline, train_samples, chunk_last1, chunk_lastN,
            optimizer, cfg, device, epoch, use_wandb,
            eval_samples=eval_samples, best_ref=best_ref,
            eval_every=args.eval_every,
            max_new_tokens=args.max_new_tokens,
            max_steps=args.max_steps_per_epoch or None,
        )
        scheduler.step()
        em = eval_string_em(pipeline, eval_samples, chunk_last1, chunk_lastN,
                            max_new_tokens=args.max_new_tokens, limit=200)
        logger.info(
            f"Epoch {epoch} | loss={avg_loss:.4f} reward={avg_reward:.4f} EM={em:.4f}"
        )
        if use_wandb:
            wandb.log({"epoch": epoch, "epoch/loss": avg_loss,
                       "epoch/mean_reward": avg_reward, "epoch/em": em})
        if em > best_ref[0]:
            best_ref[0] = em
            save_checkpoint(pipeline, args.save_dir, "best")
        save_checkpoint(pipeline, args.save_dir, f"epoch{epoch}")

    logger.info(f"Done. Best string-EM: {best_ref[0]:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
