"""
Zero-shot transfer eval: load a LoRA+projector+policy ckpt trained on a
different task (e.g. PersonaMem-32K) and evaluate on LoCoMo-MC10 test split.
"""
import argparse, os, logging, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from elasticmem.config import Config
from elasticmem.projector import LatentProjector
from elasticmem.policy import BudgetPolicy

from tasks.locomo_mc10.train import (
    precompute_all_embeddings, evaluate, MC10Pipeline,
)
from tasks.locomo_mc10.data_loader import load_locomo_mc10

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--load_checkpoint", type=str, required=True)
    p.add_argument("--top_z", type=int, default=9,
                   help="Must match the source ckpt's max_chunks (PersonaMem default=9).")
    p.add_argument("--cache_dir", type=str, default="./cache/chunk_embeddings_locomo_mc10")
    p.add_argument("--hf_cache_dir", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=5)
    args = p.parse_args()

    cfg = Config()
    cfg.model.model_name = args.model_name
    cfg.retrieval.top_z = args.top_z
    cfg.data.cache_dir = args.cache_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device={device}, Model={cfg.model.model_name}, ckpt={args.load_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reasoner = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    h = reasoner.config.hidden_size
    cfg.projector.hidden_size = h
    cfg.model.hidden_size = h
    logger.info(f"Detected hidden_size={h}")

    # Load LoRA from ckpt
    lora_dir = os.path.join(args.load_checkpoint, "lora_adapter")
    reasoner = PeftModel.from_pretrained(reasoner, lora_dir, is_trainable=False)
    logger.info(f"Loaded LoRA from {lora_dir}")

    train_s, val_s, test_s = load_locomo_mc10(cache_dir=args.hf_cache_dir)
    logger.info(f"MC10 splits: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    all_for_encode = train_s + val_s + test_s
    emb_cache = precompute_all_embeddings(reasoner, tokenizer, all_for_encode, cfg, device)

    projector = LatentProjector(
        hidden_size=cfg.projector.hidden_size,
        intermediate_size=cfg.projector.intermediate_size,
        max_queries=cfg.retrieval.n_tokens_max,
        num_heads=8, num_layers=2, ffn_mult=2,
        dropout=cfg.projector.dropout,
    ).to(device).to(torch.bfloat16)
    proj_path = os.path.join(args.load_checkpoint, "projector.pt")
    projector.load_state_dict(torch.load(proj_path, map_location=device))
    logger.info(f"Loaded projector from {proj_path}")

    policy = BudgetPolicy(
        input_dim=cfg.projector.hidden_size, hidden_size=256, num_heads=4,
        num_layers=2, dropout=0.1,
        n_choices=cfg.retrieval.n_tokens_max, max_chunks=cfg.retrieval.top_z,
    ).to(device).to(torch.float32)
    pol_path = os.path.join(args.load_checkpoint, "policy.pt")
    policy.load_state_dict(torch.load(pol_path, map_location=device))
    logger.info(f"Loaded policy from {pol_path}")

    pipeline = MC10Pipeline(
        reasoner=reasoner, tokenizer=tokenizer, projector=projector,
        top_z=cfg.retrieval.top_z, n_tokens_min=cfg.retrieval.n_tokens_min,
        n_tokens_max=cfg.retrieval.n_tokens_max,
        max_hidden_cache=cfg.retrieval.max_hidden_cache,
        temperature=cfg.train.temperature, lora_config=None, policy=policy,
    )

    acc = evaluate(pipeline, test_s, emb_cache, device, args.max_new_tokens)
    logger.info(f"LoCoMo-MC10 transfer test acc: {acc:.4f}")


if __name__ == "__main__":
    main()
