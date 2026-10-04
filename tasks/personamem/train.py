"""
GRPO training with sample-based retrieval.
Trainable: Projector + Reasoner LoRA
"""
import os
import random
import argparse
import logging

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from elasticmem.config import Config
from elasticmem.projector import LatentProjector
from elasticmem.pipeline import LatentChunkPipeline
from elasticmem.data_utils import load_all_data, encode_chunks, precompute_dense_scores
from elasticmem.policy import BudgetPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION = (
    "You MUST respond with exactly one option: (a), (b), (c), or (d). "
    "Do not include any explanation or extra text."
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_eval(samples, eval_ratio=0.1, seed=42, test_ratio=0.0):
    """Split by shared_context_id to avoid data leakage.

    If test_ratio > 0, returns (train, val, test). Otherwise (train, eval).
    """
    rng = random.Random(seed)
    sids = sorted(set(s["shared_context_id"] for s in samples))
    rng.shuffle(sids)
    n_total = len(sids)
    n_eval = max(1, int(n_total * eval_ratio))
    n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0

    test_sids = set(sids[:n_test]) if n_test > 0 else set()
    eval_sids = set(sids[n_test:n_test + n_eval])
    train_sids = set(sids[n_test + n_eval:])

    train_samples = [s for s in samples if s["shared_context_id"] in train_sids]
    eval_samples = [s for s in samples if s["shared_context_id"] in eval_sids]

    if test_ratio > 0:
        test_samples = [s for s in samples if s["shared_context_id"] in test_sids]
        return train_samples, eval_samples, test_samples
    return train_samples, eval_samples


def precompute_all_embeddings(model, tokenizer, samples, cfg, device):
    logger.info("Precomputing chunk embeddings (base model)...")
    seen = {}
    for sample in tqdm(samples, desc="Encoding chunks"):
        sid = sample["shared_context_id"]
        if sid in seen:
            continue
        seen[sid] = encode_chunks(
            model, tokenizer, sample["chunks"],
            max_length=cfg.data.max_chunk_length,
            max_hidden_cache=cfg.retrieval.max_hidden_cache,
            cache_dir=cfg.data.cache_dir,
            device=device,
        )
    logger.info(f"Encoded {len(seen)} unique conversation histories.")
    return seen


def train_one_epoch(pipeline, samples, emb_cache, optimizer, cfg, device, epoch,
                     use_wandb=False, eval_samples=None, best_acc_ref=None,
                     test_samples=None, dense_scores_cache=None, aux_lambda=0.0):
    pipeline.train()
    total_loss = 0.0
    total_reward = 0.0
    n_steps = 0

    indices = list(range(len(samples)))
    random.shuffle(indices)

    optimizer.zero_grad()
    for step_i, idx in enumerate(tqdm(indices, desc=f"Train epoch {epoch}")):
        sample = samples[idx]
        sid = sample["shared_context_id"]
        row = sample["row_data"]
        question = row["user_question_or_message"]
        correct_answer = row["correct_answer"]
        all_options = row["all_options"]

        cached = emb_cache[sid]
        chunk_last1 = cached["last1"].to(device)
        chunk_lastN = cached["lastN"].to(device)

        # GRPO rollout: G trajectories with sampled retrieval + random n_tokens
        trajectories = pipeline.grpo_rollout(
            question=question,
            correct_answer=correct_answer,
            instruction=INSTRUCTION,
            all_options=all_options,
            chunk_embs=chunk_last1,
            chunk_hiddens=chunk_lastN,
            num_generations=cfg.train.num_generations,
        )

        # Compute old logprobs ONCE (frozen snapshot)
        old_logprobs = pipeline.compute_old_logprobs(
            question=question,
            instruction=INSTRUCTION,
            all_options=all_options,
            chunk_hiddens_all=chunk_lastN,
            trajectories=trajectories,
        )

        # Multiple GRPO iterations on the same batch of trajectories
        # old_logprobs stay fixed, current logprobs change as weights update.
        # Each inner iter takes a FULL gradient step (no /num_iters scaling).
        num_iters = cfg.train.num_grpo_iters
        step_loss = 0.0
        for grpo_iter in range(num_iters):
            qid = sample["row_data"]["question_id"]
            dense_scores_q = dense_scores_cache.get(qid) if dense_scores_cache is not None else None
            result = pipeline.grpo_loss(
                question=question,
                instruction=INSTRUCTION,
                all_options=all_options,
                chunk_hiddens_all=chunk_lastN,
                trajectories=trajectories,
                old_logprobs=old_logprobs,
                epsilon=cfg.train.epsilon,
                policy_entropy_coef=0.001,
                chunk_embs_all=chunk_last1 if dense_scores_q is not None else None,
                dense_scores=dense_scores_q,
                aux_lambda=aux_lambda,
                aux_temp=1.0,
            )

            loss = result["loss"]
            loss.backward()
            step_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(pipeline.get_trainable_parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += step_loss / num_iters
        total_reward += result["mean_reward"]
        n_steps += 1

        # Log to wandb every step
        global_step = (epoch - 1) * len(indices) + step_i
        if use_wandb:
            wandb.log({
                "train/loss": result["loss"].item(),
                "train/mean_reward": result["mean_reward"],
                "train/std_reward": result["std_reward"],
                "train/llm_entropy": result.get("llm_entropy", 0.0),
                "train/policy_entropy": result.get("policy_entropy", 0.0),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        if (step_i + 1) % 50 == 0:
            logger.info(
                f"  step {step_i+1}/{len(indices)} | "
                f"loss={total_loss/n_steps:.4f} | "
                f"mean_reward={total_reward/n_steps:.4f} | "
                f"llm_ent={result.get('llm_entropy', 0.0):.3f} | "
                f"policy_ent={result.get('policy_entropy', 0.0):.3f}"
            )

            # Print 3 sample outputs so we can eyeball what the LLM is producing
            if eval_samples is not None:
                _peek_outputs(pipeline, eval_samples, emb_cache, device, k=3, tag=f"step {step_i+1}")

            # Evaluate every 50 steps (val + test)
            if eval_samples is not None and best_acc_ref is not None:
                cur_acc = evaluate(pipeline, eval_samples, emb_cache, cfg, device)
                test_msg = ""
                test_acc = None
                if test_samples is not None:
                    test_acc = evaluate(pipeline, test_samples, emb_cache, cfg, device)
                    test_msg = f" | Test: {test_acc:.4f}"
                logger.info(f"  [step {step_i+1}] Val: {cur_acc:.4f}{test_msg}")
                if use_wandb:
                    log_dict = {
                        "eval/accuracy": cur_acc,
                        "eval/best_accuracy": max(best_acc_ref[0], cur_acc),
                    }
                    if test_acc is not None:
                        log_dict["test/accuracy"] = test_acc
                    wandb.log(log_dict, step=global_step)
                if cur_acc > best_acc_ref[0]:
                    best_acc_ref[0] = cur_acc
                    save_checkpoint(pipeline, cfg.train.save_dir, "best")
                    logger.info(f"  [step {step_i+1}] New best val: {cur_acc:.4f}")
                pipeline.train()

    return total_loss / max(n_steps, 1), total_reward / max(n_steps, 1)


@torch.no_grad()
def _peek_outputs(pipeline, samples, emb_cache, device, k=3, tag=""):
    """Print k sample (question, gt, model_answer, reward) so we can eyeball the LLM."""
    pipeline.eval()
    picks = random.sample(samples, min(k, len(samples)))
    logger.info(f"  ── Sample outputs [{tag}] ──")
    for j, sample in enumerate(picks):
        sid = sample["shared_context_id"]
        row = sample["row_data"]
        question = row["user_question_or_message"]
        correct_answer = row["correct_answer"]
        all_options = row["all_options"]
        cached = emb_cache[sid]
        answer = pipeline.generate(
            question=question,
            instruction=INSTRUCTION,
            all_options=all_options,
            chunk_embs=cached["last1"].to(device),
            chunk_hiddens=cached["lastN"].to(device),
        )
        reward = pipeline._compute_reward(answer, correct_answer)
        q_short = question[:80].replace("\n", " ")
        logger.info(f"    [{j+1}] Q: {q_short}")
        logger.info(f"        GT: {correct_answer.strip()[:60]} | Pred: {answer.strip()[:60]} | r={reward:.0f}")
    pipeline.train()


@torch.no_grad()
def evaluate(pipeline, samples, emb_cache, cfg, device):
    pipeline.eval()
    correct = 0
    total = 0

    for sample in tqdm(samples, desc="Evaluating"):
        sid = sample["shared_context_id"]
        row = sample["row_data"]
        question = row["user_question_or_message"]
        correct_answer = row["correct_answer"]
        all_options = row["all_options"]

        cached = emb_cache[sid]
        chunk_last1 = cached["last1"].to(device)
        chunk_lastN = cached["lastN"].to(device)

        answer = pipeline.generate(
            question=question,
            instruction=INSTRUCTION,
            all_options=all_options,
            chunk_embs=chunk_last1,
            chunk_hiddens=chunk_lastN,
        )

        reward = pipeline._compute_reward(answer, correct_answer)
        correct += int(reward)
        total += 1

    return correct / max(total, 1)


def save_checkpoint(pipeline, save_dir, name):
    path = os.path.join(save_dir, name)
    os.makedirs(path, exist_ok=True)
    torch.save(pipeline.projector.state_dict(), os.path.join(path, "projector.pt"))
    pipeline.reasoner.save_pretrained(os.path.join(path, "lora_adapter"))
    if pipeline.policy is not None:
        torch.save(pipeline.policy.state_dict(), os.path.join(path, "policy.pt"))
    logger.info(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--top_z", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--question_path", type=str, default=None)
    parser.add_argument("--context_path", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to checkpoint dir (with projector.pt + lora_adapter/) to initialize from")
    parser.add_argument("--freeze_llm", action="store_true",
                        help="Freeze the LLM entirely (no LoRA). Only projector + policy are trained.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override checkpoint save dir (default: ./checkpoints).")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Enable gradient checkpointing on the LLM to reduce VRAM at the cost of speed.")
    args = parser.parse_args()

    cfg = Config()
    if args.model_name:
        cfg.model.model_name = args.model_name
    if args.num_epochs:
        cfg.train.num_epochs = args.num_epochs
    if args.lr:
        cfg.train.lr = args.lr
    if args.top_z:
        cfg.retrieval.top_z = args.top_z
    if args.num_generations:
        cfg.train.num_generations = args.num_generations
    if args.temperature:
        cfg.train.temperature = args.temperature
    if args.seed:
        cfg.train.seed = args.seed
    if args.question_path:
        cfg.data.question_path = args.question_path
    if args.context_path:
        cfg.data.context_path = args.context_path
    if args.save_dir:
        cfg.train.save_dir = args.save_dir

    set_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Model: {cfg.model.model_name}")
    logger.info(f"GRPO: G={cfg.train.num_generations}, temp={cfg.train.temperature}, eps={cfg.train.epsilon}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reasoner = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Sync projector dim to actual model hidden size (handles 1.5B/3B/7B).
    h = reasoner.config.hidden_size
    cfg.projector.hidden_size = h
    cfg.model.hidden_size = h
    logger.info(f"Detected hidden_size={h}")

    # Optionally load pre-trained LoRA from init_from
    init_lora_path = None
    init_proj_path = None
    if args.init_from:
        from peft import PeftModel
        init_lora_path = os.path.join(args.init_from, "lora_adapter")
        init_proj_path = os.path.join(args.init_from, "projector.pt")
        if os.path.exists(init_lora_path):
            reasoner = PeftModel.from_pretrained(reasoner, init_lora_path, is_trainable=True)
            logger.info(f"Initialized LoRA from {init_lora_path}")

    # Load data
    all_samples, groups = load_all_data(cfg.data.question_path, cfg.data.context_path)
    logger.info(f"Total samples: {len(all_samples)}, contexts: {len(groups)}")

    # 3-way split: 80/10/10 (train/val/test)
    splits = split_train_eval(all_samples, eval_ratio=0.1, seed=cfg.train.seed, test_ratio=0.1)
    train_samples, eval_samples, test_samples = splits
    logger.info(f"Train: {len(train_samples)}, Val: {len(eval_samples)}, Test: {len(test_samples)}")

    # Precompute embeddings with base model (before LoRA)
    emb_cache = precompute_all_embeddings(reasoner, tokenizer, all_samples, cfg, device)

    # Precompute dense retriever scores (MiniLM) for aux supervision
    dense_cache_path = os.path.join(cfg.data.cache_dir, "..", "dense_scores.pt")
    dense_scores_cache = precompute_dense_scores(all_samples, cache_path=dense_cache_path)
    logger.info(f"Dense scores precomputed for {len(dense_scores_cache)} questions")

    # Build pipeline (adds LoRA)
    projector = LatentProjector(
        hidden_size=cfg.projector.hidden_size,
        intermediate_size=cfg.projector.intermediate_size,
        max_queries=cfg.retrieval.n_tokens_max,
        num_heads=8,
        num_layers=2,
        ffn_mult=2,
        dropout=cfg.projector.dropout,
    ).to(device).to(torch.bfloat16)
    if init_proj_path and os.path.exists(init_proj_path):
        projector.load_state_dict(torch.load(init_proj_path, map_location=device))
        logger.info(f"Initialized projector from {init_proj_path}")
    logger.info(f"Projector params: {projector.param_count():,}")

    # Skip LoRA wrap if already loaded; otherwise create new LoRA.
    # With --freeze_llm, we never add LoRA and freeze the base model entirely.
    if args.freeze_llm:
        lora_cfg = None
        for p in reasoner.parameters():
            p.requires_grad = False
        logger.info("LLM fully frozen (--freeze_llm). Only projector + policy will be trained.")
    elif args.init_from and init_lora_path and os.path.exists(init_lora_path):
        lora_cfg = None
        # Make pre-loaded LoRA trainable; freeze base
        for n, p in reasoner.named_parameters():
            p.requires_grad = ("lora" in n.lower())
    else:
        lora_cfg = {
            "r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "target_modules": cfg.lora.target_modules,
            "lora_dropout": cfg.lora.lora_dropout,
        }

    # Budget policy for joint training
    policy = BudgetPolicy(
        input_dim=cfg.projector.hidden_size,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        n_choices=cfg.retrieval.n_tokens_max,
        max_chunks=cfg.retrieval.top_z,
    ).to(device).to(torch.float32)
    if args.init_from:
        init_policy_path = os.path.join(args.init_from, "policy.pt")
        if os.path.exists(init_policy_path):
            policy.load_state_dict(torch.load(init_policy_path, map_location=device))
            logger.info(f"Initialized policy from {init_policy_path}")
    logger.info(f"Policy params: {policy.param_count():,}")

    pipeline = LatentChunkPipeline(
        reasoner=reasoner,
        tokenizer=tokenizer,
        projector=projector,
        top_z=cfg.retrieval.top_z,
        n_tokens_min=cfg.retrieval.n_tokens_min,
        n_tokens_max=cfg.retrieval.n_tokens_max,
        max_hidden_cache=cfg.retrieval.max_hidden_cache,
        temperature=cfg.train.temperature,
        lora_config=lora_cfg,
        policy=policy,
    )

    if args.grad_ckpt:
        # Required when using LoRA + gradient checkpointing (frozen base needs input grads).
        if hasattr(pipeline.reasoner, "enable_input_require_grads"):
            pipeline.reasoner.enable_input_require_grads()
        pipeline.reasoner.gradient_checkpointing_enable()
        pipeline.reasoner.config.use_cache = False
        logger.info("Gradient checkpointing enabled on the LLM.")

    # Optimizer
    trainable_params = pipeline.get_trainable_parameters()
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Total trainable params (Projector + LoRA + Policy): {total_trainable:,}")

    optimizer = AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)

    # Wandb init
    try:
        wandb.init(
            project="latent-chunk-mem",
            name=f"grpo_z{cfg.retrieval.top_z}_G{cfg.train.num_generations}_lr{cfg.train.lr}",
            config={
            "model": cfg.model.model_name,
            "top_z": cfg.retrieval.top_z,
            "n_tokens_min": cfg.retrieval.n_tokens_min,
            "n_tokens_max": cfg.retrieval.n_tokens_max,
            "num_generations": cfg.train.num_generations,
            "lr": cfg.train.lr,
            "temperature": cfg.train.temperature,
            "epsilon": cfg.train.epsilon,
            "lora_r": cfg.lora.r,
            "projector_intermediate": cfg.projector.intermediate_size,
            "num_epochs": cfg.train.num_epochs,
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "total_trainable_params": total_trainable,
        },
        )
        use_wandb = True
    except Exception as e:
        logger.warning(f"wandb init failed: {e}, continuing without wandb")
        use_wandb = False

    # Training
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    best_acc = 0.0

    best_acc_ref = [best_acc]  # mutable ref so train_one_epoch can update
    for epoch in range(1, cfg.train.num_epochs + 1):
        avg_loss, avg_reward = train_one_epoch(
            pipeline, train_samples, emb_cache, optimizer, cfg, device, epoch, use_wandb,
            eval_samples=eval_samples, best_acc_ref=best_acc_ref,
            test_samples=test_samples,
            dense_scores_cache=dense_scores_cache,
            aux_lambda=0.0,
        )
        best_acc = best_acc_ref[0]
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{cfg.train.num_epochs} | "
            f"Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        accuracy = evaluate(pipeline, eval_samples, emb_cache, cfg, device)
        logger.info(f"Epoch {epoch} | Eval Accuracy: {accuracy:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch/avg_loss": avg_loss,
                "epoch/avg_reward": avg_reward,
                "epoch/eval_accuracy": accuracy,
                "epoch/best_accuracy": max(best_acc, accuracy),
                "epoch/lr": scheduler.get_last_lr()[0],
            })

        if accuracy > best_acc:
            best_acc = accuracy
            save_checkpoint(pipeline, cfg.train.save_dir, "best")
            logger.info(f"New best accuracy: {accuracy:.4f}")

        save_checkpoint(pipeline, cfg.train.save_dir, f"epoch{epoch}")

    logger.info(f"Training done. Best eval accuracy: {best_acc:.4f}")

    # Final test on held-out test set using best checkpoint
    logger.info("Loading best checkpoint and running final test...")
    best_ckpt = os.path.join(cfg.train.save_dir, "best")
    # reload projector + lora + policy
    projector.load_state_dict(torch.load(os.path.join(best_ckpt, "projector.pt"), map_location=device))
    # reload LoRA adapter weights into the pipeline's PeftModel
    best_lora_dir = os.path.join(best_ckpt, "lora_adapter")
    if os.path.exists(best_lora_dir):
        try:
            from peft.utils.save_and_load import set_peft_model_state_dict
            adapter_bin = os.path.join(best_lora_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_bin):
                from safetensors.torch import load_file
                lora_state = load_file(adapter_bin)
            else:
                lora_state = torch.load(os.path.join(best_lora_dir, "adapter_model.bin"), map_location=device)
            set_peft_model_state_dict(pipeline.reasoner, lora_state)
            logger.info(f"Reloaded best LoRA from {best_lora_dir}")
        except Exception as e:
            logger.warning(f"Failed to reload best LoRA: {e}")
    if pipeline.policy is not None and os.path.exists(os.path.join(best_ckpt, "policy.pt")):
        pipeline.policy.load_state_dict(torch.load(os.path.join(best_ckpt, "policy.pt"), map_location=device))
    # encode test samples' contexts (re-uses cache by hash)
    test_emb_cache = {}
    for sample in tqdm(test_samples, desc="Encoding test"):
        sid = sample["shared_context_id"]
        if sid in emb_cache:
            test_emb_cache[sid] = emb_cache[sid]
            continue
        if sid in test_emb_cache:
            continue
        test_emb_cache[sid] = encode_chunks(
            reasoner, tokenizer, sample["chunks"],
            max_length=cfg.data.max_chunk_length,
            max_hidden_cache=cfg.retrieval.max_hidden_cache,
            cache_dir=cfg.data.cache_dir,
            device=device,
        )
    test_acc = evaluate(pipeline, test_samples, test_emb_cache, cfg, device)
    logger.info(f"Final TEST accuracy: {test_acc:.4f}")
    if use_wandb:
        wandb.log({"test/final_accuracy": test_acc})
        wandb.finish()


if __name__ == "__main__":
    main()
