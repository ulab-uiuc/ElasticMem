"""
LongMemEval-MC10 training entry (or eval-only with --split_mode eval_only).
- 500 Qs total; random 80/20 split by default (400 train / 100 test).
- 10-choice MC, letter-match reward, no Gemini dependency.
- Each question has its own haystack (~540 turns); encoding cost is the bulk
  of wall time (500 × ~540 chunks) — run on a dedicated GPU.

Run:
  bash lme_mc10/run.sh
"""
import os
import sys
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

from tasks.lme_mc10.data_loader import load_lme_mc10
from tasks.lme_mc10.pipeline_lme import LMEMC10Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION = (
    "You MUST pick exactly one option from (a) to (j) — one of them is guaranteed "
    "to be correct. Do NOT say \"not answerable\", \"I don't know\", or refuse. "
    "If uncertain, make your best guess. "
    "Output exactly one token in the form (a), (b), (c), (d), (e), (f), (g), (h), (i), or (j)."
)
MAX_NEW_TOKENS = 5


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def precompute_all_embeddings(model, tokenizer, samples, cfg, device):
    seen = {}
    for sample in tqdm(samples, desc="Encoding chunks"):
        sid = sample["shared_context_id"]
        if sid in seen:
            continue
        seen[sid] = encode_chunks(
            model, tokenizer, sample["chunks"],
            max_length=cfg.data.max_chunk_length,
            max_hidden_cache=cfg.retrieval.max_hidden_cache,
            cache_dir=cfg.data.cache_dir, device=device,
        )
    return seen


@torch.no_grad()
def evaluate(pipeline, samples, emb_cache, device, max_new_tokens=MAX_NEW_TOKENS):
    pipeline.eval()
    correct, total = 0, 0
    for sample in tqdm(samples, desc="Evaluating"):
        sid = sample["shared_context_id"]
        row = sample["row_data"]
        cached = emb_cache[sid]
        answer = pipeline.generate(
            question=row["user_question_or_message"],
            instruction=INSTRUCTION,
            all_options=row["all_options"],
            chunk_embs=cached["last1"].to(device),
            chunk_hiddens=cached["lastN"].to(device),
            max_new_tokens=max_new_tokens,
        )
        if pipeline._compute_reward(answer, row["correct_answer"]) > 0:
            correct += 1
        total += 1
    return correct / max(total, 1)


def train_one_epoch(pipeline, samples, emb_cache, optimizer, cfg, device, epoch,
                    test_samples, best_ref, use_wandb,
                    eval_every=500, max_new_tokens=MAX_NEW_TOKENS):
    pipeline.train()
    total_loss, total_reward, n_steps = 0.0, 0.0, 0
    indices = list(range(len(samples)))
    random.shuffle(indices)
    optimizer.zero_grad()

    for step_i, idx in enumerate(tqdm(indices, desc=f"Train epoch {epoch}")):
        sample = samples[idx]
        sid = sample["shared_context_id"]
        row = sample["row_data"]
        cached = emb_cache[sid]
        chunk_last1 = cached["last1"].to(device)
        chunk_lastN = cached["lastN"].to(device)

        trajectories = pipeline.grpo_rollout(
            question=row["user_question_or_message"],
            correct_answer=row["correct_answer"],
            instruction=INSTRUCTION,
            all_options=row["all_options"],
            chunk_embs=chunk_last1, chunk_hiddens=chunk_lastN,
            num_generations=cfg.train.num_generations, max_new_tokens=max_new_tokens,
        )
        old_logprobs = pipeline.compute_old_logprobs(
            question=row["user_question_or_message"],
            instruction=INSTRUCTION, all_options=row["all_options"],
            chunk_hiddens_all=chunk_lastN, trajectories=trajectories,
        )
        step_loss = 0.0
        for _ in range(cfg.train.num_grpo_iters):
            result = pipeline.grpo_loss(
                question=row["user_question_or_message"],
                instruction=INSTRUCTION, all_options=row["all_options"],
                chunk_hiddens_all=chunk_lastN, trajectories=trajectories,
                old_logprobs=old_logprobs, epsilon=cfg.train.epsilon,
                policy_entropy_coef=0.001, aux_lambda=0.0, aux_temp=1.0,
            )
            loss = result["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); break
            loss.backward()
            step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(pipeline.get_trainable_parameters(), 1.0)
            for p in pipeline.get_trainable_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    p.grad.zero_()
            optimizer.step(); optimizer.zero_grad()

        total_loss += step_loss / cfg.train.num_grpo_iters
        total_reward += result["mean_reward"]
        n_steps += 1
        global_step = (epoch - 1) * len(indices) + step_i
        if use_wandb:
            wandb.log({
                "train/loss": result["loss"].item(),
                "train/mean_reward": result["mean_reward"],
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)
        # Free per-step tensors and release unused CUDA blocks back to allocator.
        # LME per-question haystacks have different shapes (~540 chunks each),
        # which fragments PyTorch's caching allocator without this cleanup.
        del trajectories, old_logprobs, result, chunk_last1, chunk_lastN
        if (step_i + 1) % 5 == 0:
            torch.cuda.empty_cache()

        if (step_i + 1) % eval_every == 0:
            test_acc = evaluate(pipeline, test_samples, emb_cache, device,
                                max_new_tokens=max_new_tokens)
            logger.info(f"  [step {step_i+1}] Test acc: {test_acc:.4f}")
            if use_wandb:
                wandb.log({"test/accuracy": test_acc,
                           "test/best_accuracy": max(best_ref[0], test_acc)},
                          step=global_step)
            if test_acc > best_ref[0]:
                best_ref[0] = test_acc
                save_checkpoint(pipeline, cfg.train.save_dir, "best")
            pipeline.train()

    return total_loss / max(n_steps, 1), total_reward / max(n_steps, 1)


def save_checkpoint(pipeline, save_dir, name):
    path = os.path.join(save_dir, name)
    os.makedirs(path, exist_ok=True)
    torch.save(pipeline.projector.state_dict(), os.path.join(path, "projector.pt"))
    pipeline.reasoner.save_pretrained(os.path.join(path, "lora_adapter"))
    if pipeline.policy is not None:
        torch.save(pipeline.policy.state_dict(), os.path.join(path, "policy.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--top_z", type=int, default=20)
    p.add_argument("--num_generations", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--save_dir", type=str, default="./checkpoints/lme_mc10")
    p.add_argument("--cache_dir", type=str, default="./cache/chunk_embeddings_lme_mc10")
    p.add_argument("--hf_cache_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--split_mode", type=str, default="random",
                   choices=["random", "eval_only"],
                   help="random=80/20 train/test, eval_only=no train (500 test)")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="Optional path to LoRA+projector ckpt (for transfer eval)")
    p.add_argument("--n_tokens_max", type=int, default=None,
                   help="Override max soft tokens per chunk (default: cfg=20).")
    p.add_argument("--max_hidden_cache", type=int, default=None,
                   help="Override #hidden states cached per chunk (default: cfg=20). Must be >= n_tokens_max.")
    args = p.parse_args()

    cfg = Config()
    if args.model_name: cfg.model.model_name = args.model_name
    if args.num_epochs: cfg.train.num_epochs = args.num_epochs
    if args.lr: cfg.train.lr = args.lr
    if args.top_z: cfg.retrieval.top_z = args.top_z
    if args.num_generations: cfg.train.num_generations = args.num_generations
    if args.n_tokens_max: cfg.retrieval.n_tokens_max = args.n_tokens_max
    if args.max_hidden_cache: cfg.retrieval.max_hidden_cache = args.max_hidden_cache
    cfg.train.save_dir = args.save_dir
    cfg.data.cache_dir = args.cache_dir
    cfg.train.seed = args.seed
    set_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device={device}, Model={cfg.model.model_name}, split={args.split_mode}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # Single-GPU (model-parallel broke pipeline due to manual .to(device) calls).
    reasoner = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    h = reasoner.config.hidden_size
    cfg.projector.hidden_size = h
    cfg.model.hidden_size = h

    # Optional: load pretrained LoRA (e.g. from locomo_mc10 checkpoint) for transfer eval
    if args.load_checkpoint:
        from peft import PeftModel
        lora_dir = os.path.join(args.load_checkpoint, "lora_adapter")
        reasoner = PeftModel.from_pretrained(reasoner, lora_dir,
                                             is_trainable=(args.split_mode != "eval_only"))
        logger.info(f"Loaded LoRA from {lora_dir}")

    train_samples, test_samples = load_lme_mc10(
        cache_dir=args.hf_cache_dir, split_mode=args.split_mode,
        test_ratio=args.test_ratio, seed=args.seed,
    )
    logger.info(f"LME-MC10 splits: train={len(train_samples)}, test={len(test_samples)}")

    all_for_encode = train_samples + test_samples
    emb_cache = precompute_all_embeddings(reasoner, tokenizer, all_for_encode, cfg, device)

    projector = LatentProjector(
        hidden_size=cfg.projector.hidden_size,
        intermediate_size=cfg.projector.intermediate_size,
        max_queries=cfg.retrieval.n_tokens_max,
        num_heads=8, num_layers=2, ffn_mult=2,
        dropout=cfg.projector.dropout,
    ).to(device).to(torch.bfloat16)
    if args.load_checkpoint:
        proj_path = os.path.join(args.load_checkpoint, "projector.pt")
        if os.path.exists(proj_path):
            projector.load_state_dict(torch.load(proj_path, map_location=device))
            logger.info(f"Loaded projector from {proj_path}")

    policy = BudgetPolicy(
        input_dim=cfg.projector.hidden_size, hidden_size=256, num_heads=4,
        num_layers=2, dropout=0.1,
        n_choices=cfg.retrieval.n_tokens_max, max_chunks=cfg.retrieval.top_z,
    ).to(device).to(torch.float32)
    if args.load_checkpoint:
        pol_path = os.path.join(args.load_checkpoint, "policy.pt")
        if os.path.exists(pol_path):
            policy.load_state_dict(torch.load(pol_path, map_location=device))
            logger.info(f"Loaded policy from {pol_path}")

    lora_cfg = None if args.load_checkpoint else {
        "r": cfg.lora.r, "lora_alpha": cfg.lora.lora_alpha,
        "target_modules": cfg.lora.target_modules, "lora_dropout": cfg.lora.lora_dropout,
    }

    pipeline = LMEMC10Pipeline(
        reasoner=reasoner, tokenizer=tokenizer, projector=projector,
        top_z=cfg.retrieval.top_z, n_tokens_min=cfg.retrieval.n_tokens_min,
        n_tokens_max=cfg.retrieval.n_tokens_max,
        max_hidden_cache=cfg.retrieval.max_hidden_cache,
        temperature=cfg.train.temperature, lora_config=lora_cfg, policy=policy,
    )

    if args.split_mode == "eval_only":
        logger.info("eval_only mode: running evaluation once over all 500 Qs")
        acc = evaluate(pipeline, test_samples, emb_cache, device, args.max_new_tokens)
        logger.info(f"LME-MC10 Test acc: {acc:.4f}")
        return

    optimizer = AdamW(pipeline.get_trainable_parameters(), lr=cfg.train.lr,
                      weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)

    try:
        wandb.init(project="latent-chunk-mem",
                   name=f"lme_mc10_z{cfg.retrieval.top_z}_lr{cfg.train.lr}",
                   config={"dataset": "lme_mc10", "top_z": cfg.retrieval.top_z,
                           "lr": cfg.train.lr, "model": cfg.model.model_name})
        use_wandb = True
    except Exception:
        use_wandb = False

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    best_ref = [0.0]
    for epoch in range(1, cfg.train.num_epochs + 1):
        avg_loss, avg_reward = train_one_epoch(
            pipeline, train_samples, emb_cache, optimizer, cfg, device, epoch,
            test_samples=test_samples, best_ref=best_ref, use_wandb=use_wandb,
            eval_every=args.eval_every, max_new_tokens=args.max_new_tokens,
        )
        scheduler.step()
        test_acc = evaluate(pipeline, test_samples, emb_cache, device, args.max_new_tokens)
        logger.info(f"Epoch {epoch} | loss={avg_loss:.4f} reward={avg_reward:.4f} test_acc={test_acc:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch, "epoch/loss": avg_loss,
                       "epoch/reward": avg_reward, "epoch/test_acc": test_acc})
        if test_acc > best_ref[0]:
            best_ref[0] = test_acc
            save_checkpoint(pipeline, cfg.train.save_dir, "best")
        save_checkpoint(pipeline, cfg.train.save_dir, f"epoch{epoch}")

    logger.info(f"Done. Best test acc: {best_ref[0]:.4f}")
    if use_wandb: wandb.finish()


if __name__ == "__main__":
    main()
