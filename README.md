<div align="center">
  <img src="assets/logo.svg" alt="ElasticMem Logo" width="420">
</div>



<h1 align="center">ElasticMem: Latent Memory as a Learnable Resource for LLM Agents</h1>




<div align="center">
  <p>
    <a href='https://arxiv.org/abs/2605.30690'><img src='https://img.shields.io/badge/arXiv-2605.30690-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white'></a>
    <a href="https://huggingface.co/papers/2605.30690"><img src="https://img.shields.io/badge/HuggingFace-Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="HuggingFace Paper"></a>
    <br>
    <a href="https://github.com/ulab-uiuc/ElasticMem/stargazers"><img src='https://img.shields.io/github/stars/ulab-uiuc/ElasticMem?color=f1e05a&style=for-the-badge&logo=star&logoColor=white' /></a>
    <a href="https://github.com/ulab-uiuc/ElasticMem/forks"><img src='https://img.shields.io/github/forks/ulab-uiuc/ElasticMem?color=2ea44f&style=for-the-badge&logo=git&logoColor=white' /></a>
    <a href="https://github.com/ulab-uiuc/ElasticMem/issues"><img src='https://img.shields.io/github/issues/ulab-uiuc/ElasticMem?color=d73a49&style=for-the-badge&logo=github&logoColor=white' /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache--2.0-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</div>




## 🧩 Overview

**ElasticMem** is a memory-augmented LLM framework that learns to use long-term memory as an **elastic latent resource**. Existing approaches treat memory as a *fixed* resource: text-space methods concatenate retrieved memories into the context window, paying a large token cost and inheriting sensitivity to noisy evidence, while latent-space methods cut the textual cost but still rely on rigid retrieval or a fixed-capacity memory interface. Either way, a query that needs one decisive memory and a query that needs ten are served the same allocation.

ElasticMem closes that gap. It builds an offline latent memory bank of retrieval keys and content caches, **retrieves from the reasoner's own hidden state** rather than a detached encoder, gives each retrieved memory a **variable latent budget** chosen by a learned policy, and injects the selected latent states as soft memory tokens. The whole memory-use process — retrieval, allocation, and generation — is optimized end-to-end with downstream task reward via **group-relative policy optimization (GRPO)**.

❗The memory ElasticMem learns to manage is **not** a prompt-engineering artifact. Nothing is written back into the context as text: retrieved memories enter the reasoner as *latent states*, and what the policy learns is **how much latent capacity each memory deserves for this particular query**.

**Highlights**

- **Latent memory bank, built once**: Every memory chunk — dialogue turn-pairs, passages, or procedural skill cards — is encoded once by a frozen LLM encoder into a retrieval key plus a content cache, then frozen. No re-encoding during training.

- **Query-conditioned retrieval**: The retrieval query is the reasoner's hidden state after a learned control token, so retrieval is trained jointly with the task instead of being fixed to off-the-shelf cosine similarity.

- **Elastic budget allocation**: A small transformer policy reads the query state, keys, similarities, and ranks, then assigns each retrieved chunk a budget of latent tokens sequentially — spending capacity where evidence is actually useful.

- **Soft-token injection**: Allocated latent states pass through a projector into the reasoner's embedding space, so memory costs *latent tokens*, not context tokens.

- **One reward, end-to-end**: Retrieval, allocation, and answer generation are optimized together by GRPO against task reward (exact match for QA, expert-action agreement for ALFWorld).

- **Five benchmarks, one protocol**: PersonaMem-32K/128K, LoCoMo-MC10, LongMemEval-MC10, and ALFWorld under a single comparable setup, with frozen split files for bit-identical reproduction.



<div align="center">
  <img src="./assets/model.png" width="900" alt="ElasticMem">
</div>



## 📰 News

- 🚀 **[2026-10]**: **ElasticMem** is officially released — long-term memory as an *elastic* latent resource: retrieved adaptively from the reasoner's own hidden state, budgeted per query by a learned policy, and optimized end-to-end with task reward 🧠. Paper, code, and the five-benchmark MemorySuite protocol are all available.




## 🔗 Links

- [Overview](#-overview)
- [News](#-news)
- [Get Started](#-get-started)
- [Installation](#installation)
- [Preparing Data](#-preparing-data)
- [Experiments](#-experiments)
- [Repository Layout](#-repository-layout)
- [Extending to New Datasets](#-extending-to-new-datasets)
- [Commonly Used Configs](#️-commonly-used-configs)
- [Citation](#-citation)






## 🚀 Get Started

### Installation

```bash
# Clone the repository
git clone https://github.com/ulab-uiuc/ElasticMem
cd ElasticMem

# Create and activate virtual environment
conda create -n elasticmem python=3.10
conda activate elasticmem

# PyTorch (match your CUDA version)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Others
pip install -r requirements.txt

# ALFWorld only (embodied experiments)
pip install "alfworld[full]" && alfworld-download
```

ElasticMem runs on a single GPU. A 3B backbone fits comfortably on 24 GB; the 7B backbone needs ~40 GB for training. All experiments in the paper use one NVIDIA RTX A6000 (48 GB).




### 📊 Preparing Data

ElasticMem is evaluated on **five** benchmarks. Per-dataset recipes — source, schema, splits, retrieval settings, prompt template, hyperparameters, and reward — live under [`benchmarks/`](./benchmarks). Frozen split index files ship with this repo so partitions are bit-identical regardless of Python or shuffle implementation.

| # | Benchmark | Task | Size | Memory unit |
|---|-----------|------|------|-------------|
| 1 | [PersonaMem-32K](./benchmarks/PersonaMem-32K)     | 4-choice MC, long persona context  | 589 Q / 37 contexts     | conversation turn-pair |
| 2 | [PersonaMem-128K](./benchmarks/PersonaMem-128K)   | 4-choice MC, very-long context     | 2,727 Q / 60 contexts   | conversation turn-pair |
| 3 | [LoCoMo-MC10](./benchmarks/LoCoMo-MC10)           | 10-choice MC, long dialogue        | 1,986 Q / 10 conversations | session turn-pair |
| 4 | [LongMemEval-MC10](./benchmarks/LongMemEval-MC10) | 10-choice MC, per-question haystack | 500 Q / 500 contexts   | session turn-pair |
| 5 | [ALFWorld](./benchmarks/ALFWorld)                 | interactive text adventure (env)   | 140 valid_seen + 134 valid_unseen | LLM-generated skill card |

#### 1) PersonaMem-32K / PersonaMem-128K
- Source: [PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem)
- Place `questions_{32k,128k}.csv` and `shared_contexts_{32k,128k}.jsonl` under `data/`.
- Splits: `benchmarks/PersonaMem-*/data/splits/{train,val,test}_ids.txt`, by `shared_context_id`.

#### 2) LoCoMo-MC10
- Source: [LoCoMo](https://github.com/snap-research/locomo), converted to a 10-way multiple-choice protocol.
- `bash benchmarks/LoCoMo-MC10/scripts/download_data.sh` pulls it into the HF cache; the loader also downloads on demand.
- Splits: 6 / 2 / 2 conversations (train / val / test).

#### 3) LongMemEval-MC10
- Source: **LongMemEval-S** from [LongMemEval](https://github.com/xiaowu0162/LongMemEval), converted to 10-way MC.
- Each question carries its **own** haystack, so the memory bank is per question — this is the most expensive benchmark to encode (see [Commonly Used Configs](#️-commonly-used-configs)).
- `--split_mode random` gives an 80/20 train/test split; `--split_mode eval_only` evaluates all 500 questions (used for transfer).

#### 4) ALFWorld
- Follow the official instructions to install the environment and assets: [ALFWorld](https://github.com/alfworld/alfworld).
- Collect offline expert trajectories (the raw material for the memory corpus). We reuse the replay collector from [MemSkill](https://github.com/ViktorAxelsen/MemSkill) rather than vendoring a copy:

```bash
git clone https://github.com/ViktorAxelsen/MemSkill ../MemSkill
bash benchmarks/ALFWorld/scripts/build_replay.sh      # writes the three JSONs into ./data
```

- Then turn the memory-pool trajectories into **procedural skill cards** (150–250 words each), which become the memory chunks:

```bash
export GEMINI_API_KEY=...        # any summarizer LLM works; see llm_judge.py
bash build_skill_cards.sh
```

The memory pool is the 80% of expert games reserved by `--memory-ratio`; evaluation games never contribute cards to their own bank.




## 🧪 Experiments

> [!IMPORTANT]
>
> Before running, open the `.sh` script you intend to use and check `MODEL`, data paths, `--top_z`, and `--save_dir`. Defaults reproduce the paper's main table for the 3B backbone; export `MODEL="Qwen/Qwen2.5-7B-Instruct"` for the 7B rows.

Memory encoding is cached on disk by content hash (`cache/`), so the first run of any script pays the one-time offline encoding cost and later runs start immediately.

### 🖥️ Training

```bash
bash train_personamem.sh 32k      # or: bash train_personamem.sh 128k
bash train_locomo.sh
bash train_longmemeval.sh
bash train_alfworld.sh            # run build_skill_cards.sh first
```

### 🧭 Evaluation

```bash
bash eval_personamem.sh 32k ./checkpoints/personamem_32k/best
bash eval_locomo.sh              ./checkpoints/locomo_mc10/best
bash eval_longmemeval.sh         ./checkpoints/lme_mc10/best
bash eval_alfworld.sh seen       ./checkpoints/alfworld/best
bash eval_alfworld.sh unseen     ./checkpoints/alfworld/best
```

### 📏 Baseline

```bash
bash eval_text_rag.sh            # text-space RAG: same retrieved chunks, raw text
```




## 📁 Repository Layout

```
ElasticMem/
├── pipeline.py              # the method: retrieval, budget allocation, injection, GRPO rollout
├── projector.py             # latent projector (hidden states → soft memory tokens)
├── stage2/policy.py         # BudgetPolicy: sequential per-chunk latent budget
├── data_utils.py            # chunking + offline memory-bank encoding (content-hash cached)
├── config.py                # dataclass config shared by all entry points
├── train.py / eval.py       # PersonaMem entry points
├── locomo_mc10/             # LoCoMo-MC10 loader, pipeline, train/eval
├── lme_mc10/                # LongMemEval-MC10 loader, pipeline, train
├── alfworld_il/             # skill-card construction, training, live-env evaluation
├── baseline/                # text-space RAG baseline
├── benchmarks/              # per-dataset recipes: settings.yaml, splits, download scripts
└── *.sh                     # one train/eval script per benchmark
```




## 🔧 Extending to New Datasets

ElasticMem only needs three things from a dataset: how to **segment** history into memory chunks, how to **format** the query and answer, and how to **score** an output. To add one:

- Add a loader in `<your_dataset>/data_loader.py` that returns samples of the form `{"shared_context_id", "chunks": [str], "row_data": {...}}`.
- Reuse `data_utils.encode_chunks` to build the latent memory bank — no changes needed, caching is by content hash.
- Add a thin pipeline module if the prompt differs (see `locomo_mc10/pipeline_mc10.py` for a 10-way MC example, `alfworld_il/pipeline_alfworld.py` for an interactive-environment example).
- Define the reward in your train entry point (`_compute_reward`), then add a `train_<dataset>.sh`.

`pipeline.py` itself normally needs no changes: the retrieval → budget → injection → GRPO loop is dataset-agnostic. Interactive environments are the exception, since they need a step loop around `generate()`.




## ⚙️ Commonly Used Configs

**Core**
- `--model_name`: backbone (`Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`)
- `--question_path` / `--context_path`: dataset files (PersonaMem); MC10 datasets resolve through the HF cache
- `--cache_dir`: latent memory bank cache — **must differ per (dataset, backbone, `max_hidden_cache`)**, since the cache key binds to all three
- `--save_dir`: checkpoint directory (`projector.pt`, `policy.pt`, `lora_adapter/`)
- `--seed`: default 42, used by the split helpers as well

**Retrieval & memory**
- `--top_z`: number of retrieved chunks per query (9 for PersonaMem-32K, 21 for 128K, 20 for the MC10 datasets, 10 for ALFWorld)
- `--n_tokens_max`: maximum latent tokens a single chunk may receive — this is the policy's action space
- `--max_hidden_cache`: how many trailing hidden states to cache per chunk; must be ≥ `--n_tokens_max`, and it drives the on-disk cache size
- `--max_chunk_length`: truncation length when encoding a chunk (default 2048)

> [!NOTE]
>
> Offline encoding is one forward pass per chunk at batch size 1, so its cost scales with the number of chunks, not context length. On one A6000 it runs at roughly 45 ms/chunk for the 3B backbone and 80 ms/chunk for 7B — a few minutes for PersonaMem-32K, LoCoMo, or the ALFWorld skill bank. LongMemEval is the outlier: 124k chunks across 500 per-question haystacks means ~1.5 h (3B) or ~4 h (7B) and 14–25 GB of cache. Encode once and reuse: every later run hits the cache.

**Training (GRPO)**
- `--num_generations`: group size per query (4 or 8 in our runs)
- `--num_epochs`, `--lr`: schedule; 2e-5 is the default across datasets
- `--temperature`: sampling temperature for the retrieval control token
- `--eval_every`: validation cadence in steps
- `--grad_ckpt`: gradient checkpointing — cuts training VRAM, no effect on inference (the eval path re-enables the KV cache)

**Eval / transfer**
- `--checkpoint_dir` (PersonaMem) / `--load_checkpoint` (MC10 datasets) / `--checkpoint` (ALFWorld)
- `--split_mode eval_only`: LongMemEval-MC10 evaluation over all 500 questions without training — the setting we use for cross-dataset transfer
- `--n_tokens_fixed`: override the learned policy with a constant budget; use it to ablate elastic allocation

**ALFWorld-specific**
- `--memory-ratio`: fraction of expert games forming the memory pool (0.8)
- `--max_steps` (30) / `--per_game_timeout` (240 s) / `--history_steps` (5): episode budget and how much interaction history enters the prompt
- `--max_new_tokens`: 16 for action generation (5 for the multiple-choice datasets)

**Logging**
- `--wandb-project` / `WANDB_API_KEY`: W&B is optional; set `WANDB_MODE=offline` to disable network logging




## 🙏 Acknowledgments

We thank the authors and maintainers of **[PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem)**, **[LoCoMo](https://github.com/snap-research/locomo)**, **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)**, and **[ALFWorld](https://github.com/alfworld/alfworld)** for releasing their datasets, evaluation protocols, and supporting code. We also thank **[MemSkill](https://github.com/ViktorAxelsen/MemSkill)**, whose ALFWorld expert-replay collector we reuse for offline trajectory collection, and the **[Qwen](https://github.com/QwenLM/Qwen2.5)**, **[PEFT](https://github.com/huggingface/peft)**, and **[Transformers](https://github.com/huggingface/transformers)** teams for the tooling this work builds on. Open benchmarks and open models are what make agent-memory research reproducible.




## 📚 Citation

```bibtex
@article{feng2026elasticmem,
  title={ElasticMem: Latent Memory as a Learnable Resource for LLM Agents},
  author={Feng, Tao and Ye, Chongrui and Yu, Fangxu and Luo, Tianyang and Xu, Jingjun and Xu, Xueqiang and Zhang, Haozhen and Zhang, Weizhi and Lei, Zijie and You, Jiaxuan},
  journal={arXiv preprint arXiv:2605.30690},
  year={2026}
}
```
