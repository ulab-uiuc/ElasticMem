from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    hidden_size: int = 2048
    torch_dtype: str = "bfloat16"


@dataclass
class ProjectorConfig:
    hidden_size: int = 2048
    intermediate_size: int = 2560  # MLP 中间层
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    # Cover all 4 attention projections (drops MLP to fit in memory)
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    lora_dropout: float = 0.1


@dataclass
class RetrievalConfig:
    top_z: int = 9                  # 检索 candidate 数量
    n_tokens_min: int = 1           # 每个 chunk 最少取几个 latent token
    n_tokens_max: int = 20          # 每个 chunk 最多取几个 latent token
    max_hidden_cache: int = 20      # 离线缓存每个 chunk 最后多少个 hidden states


@dataclass
class DataConfig:
    question_path: str = "../ICLR2026_RF-Mem/RF_mem/personamem_data/data/questions_32k.csv"
    context_path: str = "../ICLR2026_RF-Mem/RF_mem/personamem_data/data/shared_contexts_32k.jsonl"
    cache_dir: str = "./cache/chunk_embeddings"
    max_chunk_length: int = 2048    # 足够大，确保不截断任何chunk


@dataclass
class TrainConfig:
    num_epochs: int = 10
    batch_size: int = 1             # 每次处理一个 question
    num_generations: int = 8        # GRPO: 每个 question 采样几组 trajectory
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epsilon: float = 0.05           # GRPO clipping (tightened from 0.2)
    num_grpo_iters: int = 2         # 每个 step 内 GRPO 迭代次数
    temperature: float = 1.0        # 采样 retrieval token 的温度
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    eval_ratio: float = 0.1        # 验证集比例
    seed: int = 42
    gradient_accumulation_steps: int = 4


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
