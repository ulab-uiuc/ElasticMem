"""
Core pipeline: sample-based retrieval + latent token injection + GRPO training.

Flow:
  1. question → Reasoner → logits → sample token_t (stochastic retrieval decision)
  2. token_t → one more forward step → h_{t+1} → cosine sim → top-Z chunks
  3. chunks → random n_tokens → projector → soft tokens (external, masked in loss)
  4. [question | token_t | soft_tokens | instruction+options] → generate answer
  5. GRPO loss: mask only soft tokens, compute loss on token_t + answer

Trainable: Projector (MLP) + Reasoner (LoRA)
"""
import random
import re
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

from projector import LatentProjector


class LatentChunkPipeline(nn.Module):

    def __init__(
        self,
        reasoner: nn.Module,
        tokenizer,
        projector: LatentProjector,
        top_z: int = 9,
        n_tokens_min: int = 1,
        n_tokens_max: int = 20,
        max_hidden_cache: int = 20,
        temperature: float = 1.0,
        lora_config: dict = None,
        policy: nn.Module = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.projector = projector
        self.policy = policy  # optional BudgetPolicy for joint training
        self.top_z = top_z
        self.n_tokens_min = n_tokens_min
        self.n_tokens_max = n_tokens_max
        self.max_hidden_cache = max_hidden_cache
        self.temperature = temperature

        if lora_config is not None:
            target_modules = lora_config.get("target_modules", "q_proj,v_proj")
            if isinstance(target_modules, str):
                target_modules = [m.strip() for m in target_modules.split(",")]
            peft_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.reasoner = get_peft_model(reasoner, peft_config)
            self.reasoner.print_trainable_parameters()
        else:
            self.reasoner = reasoner
            # If reasoner is already a PeftModel (pre-loaded LoRA), keep LoRA trainable;
            # otherwise freeze everything.
            from peft import PeftModel
            is_peft = isinstance(reasoner, PeftModel)
            for name, param in self.reasoner.named_parameters():
                if is_peft and "lora" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    @property
    def device(self):
        return next(self.projector.parameters()).device

    @property
    def embed_dtype(self):
        return self.reasoner.get_input_embeddings().weight.dtype

    def get_trainable_parameters(self):
        params = list(self.projector.parameters())
        for name, param in self.reasoner.named_parameters():
            if param.requires_grad:
                params.append(param)
        if self.policy is not None:
            params += list(self.policy.parameters())
        return params

    def _sample_n_tokens_via_policy(
        self,
        query_emb: torch.Tensor,        # [hidden]
        chunk_embs_top: torch.Tensor,   # [Z, hidden]
        chunk_scores: torch.Tensor,     # [Z]
    ) -> tuple:
        """
        Use policy to sequentially sample n_tokens for each chunk.

        Returns:
            n_tokens_list: list of int, length Z
            actions:       [Z] tensor of action indices (0..n_choices-1)
            log_probs:     [Z] tensor of log probs (NOT detached)
        """
        device = next(self.policy.parameters()).device
        Z = chunk_embs_top.size(0)
        n_choices = self.policy.n_choices

        q = query_emb.to(device).to(torch.float32).unsqueeze(0)
        c = chunk_embs_top.to(device).to(torch.float32).unsqueeze(0)
        s = chunk_scores.to(device).to(torch.float32).unsqueeze(0)

        n_tokens_list = []
        actions = []
        log_probs = []
        actions_so_far = []

        for i in range(Z):
            prev_choices = torch.full((1, Z), n_choices, device=device, dtype=torch.long)
            for k, a in enumerate(actions_so_far):
                prev_choices[0, k] = a

            logits = self.policy(q, c, s, prev_choices)
            logits_i = logits[0, i]
            from torch.distributions import Categorical
            dist = Categorical(logits=logits_i)
            action = dist.sample()
            log_prob = F.log_softmax(logits_i, dim=-1)[action]

            n_i = action.item() + 1
            n_tokens_list.append(n_i)
            actions.append(action.item())
            log_probs.append(log_prob)
            actions_so_far.append(action.item())

        return n_tokens_list, torch.tensor(actions, device=device, dtype=torch.long), torch.stack(log_probs)

    def _compute_policy_logprobs(
        self,
        query_emb: torch.Tensor,
        chunk_embs_top: torch.Tensor,
        chunk_scores: torch.Tensor,
        actions: torch.Tensor,           # [Z] action indices (saved from rollout)
        return_entropy: bool = False,
    ) -> torch.Tensor:
        """Recompute policy log_probs of the given actions with current weights.

        If return_entropy, also returns per-step entropy [Z].
        """
        device = next(self.policy.parameters()).device
        Z = chunk_embs_top.size(0)
        n_choices = self.policy.n_choices

        q = query_emb.to(device).to(torch.float32).unsqueeze(0)
        c = chunk_embs_top.to(device).to(torch.float32).unsqueeze(0)
        s = chunk_scores.to(device).to(torch.float32).unsqueeze(0)

        log_probs = []
        entropies = []
        for i in range(Z):
            prev_choices = torch.full((1, Z), n_choices, device=device, dtype=torch.long)
            for k in range(i):
                prev_choices[0, k] = actions[k].item()
            logits = self.policy(q, c, s, prev_choices)
            logits_i = logits[0, i]
            logp_all = F.log_softmax(logits_i, dim=-1)
            log_prob = logp_all[actions[i]]
            log_probs.append(log_prob)
            if return_entropy:
                probs = torch.exp(logp_all)
                ent = -(probs * logp_all).sum()
                entropies.append(ent)
        if return_entropy:
            return torch.stack(log_probs), torch.stack(entropies)
        return torch.stack(log_probs)

    # ──────────────────────────────────────────────────────
    #  Step 1: Process question → sample retrieval token
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_retrieval_token(
        self, question: str, temperature: float = None,
    ) -> Dict:
        """
        question → Reasoner → logits → sample token_t → one more step → h_{t+1}

        Returns dict with:
            sampled_token_id: the sampled token id
            query_emb: h_{t+1} [1, hidden_size], for retrieval
            kv_cache: KV cache after [question, token_t]
            question_len: number of question tokens
        """
        temp = temperature or self.temperature
        embed_layer = self.reasoner.get_input_embeddings()

        # Forward question
        q_ids = self.tokenizer.encode(
            question, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        q_embeds = embed_layer(q_ids).to(self.embed_dtype)

        q_out = self.reasoner(
            inputs_embeds=q_embeds,
            use_cache=True,
            output_hidden_states=True,
        )
        kv_cache = q_out.past_key_values

        # Sample retrieval token from logits
        logits = q_out.logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(logits / temp, dim=-1)
        sampled_token_id = torch.multinomial(probs, 1)  # [1]

        # One more forward step with sampled token → h_{t+1}
        token_embeds = embed_layer(sampled_token_id.unsqueeze(0)).to(self.embed_dtype)  # [1,1,H]
        token_out = self.reasoner(
            inputs_embeds=token_embeds,
            past_key_values=kv_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        h_t1 = token_out.hidden_states[-1][0, -1, :]  # [hidden_size]
        kv_cache = token_out.past_key_values

        return {
            "sampled_token_id": sampled_token_id,       # [1]
            "query_emb": h_t1.unsqueeze(0),             # [1, hidden_size]
            "kv_cache": kv_cache,
            "question_len": q_ids.size(1),
        }

    # ──────────────────────────────────────────────────────
    #  Step 2: Retrieve
    # ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query_emb: torch.Tensor,     # [1, hidden_size]
        chunk_embs: torch.Tensor,     # [num_chunks, 1, hidden_size]
    ) -> torch.Tensor:
        q = F.normalize(query_emb.squeeze(0), dim=-1)
        c = F.normalize(chunk_embs.squeeze(1), dim=-1)
        scores = c @ q
        top_z = min(self.top_z, scores.size(0))
        _, indices = scores.topk(top_z)
        return indices

    # ──────────────────────────────────────────────────────
    #  Step 3: Build full sequence for forward/loss
    # ──────────────────────────────────────────────────────

    def _sample_n_tokens_list(self, num_chunks: int) -> List[int]:
        return [random.randint(self.n_tokens_min, self.n_tokens_max) for _ in range(num_chunks)]

    def build_full_sequence(
        self,
        question: str,
        sampled_token_id: torch.Tensor,   # [1]
        chunk_hiddens: torch.Tensor,       # [Z, max_hidden_cache, hidden_size]
        instruction: str,
        all_options: str,
        answer_text: str = None,
        n_tokens_list: List[int] = None,
    ) -> Dict:
        """
        Build the full sequence with proper label masking:

        [question tokens]   labels=-100  (input)
        [sampled_token_t]   labels=token_id  (model decision, in loss)
        [soft_tokens...]    labels=-100  (retrieved, masked)
        [\\n separators]     labels=-100
        [instruction+opts]  labels=-100  (input)
        [answer tokens]     labels=answer_ids  (model decision, in loss)

        Returns dict with inputs_embeds, attention_mask, labels, metadata.
        """
        embed_layer = self.reasoner.get_input_embeddings()
        device = self.device
        dtype = self.embed_dtype

        num_chunks = chunk_hiddens.size(0)
        if n_tokens_list is None:
            n_tokens_list = self._sample_n_tokens_list(num_chunks)

        # ── Part 1: question tokens (masked) ──
        q_ids = self.tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(device)
        q_embeds = embed_layer(q_ids).squeeze(0).to(dtype)  # [q_len, H]
        q_labels = torch.full((q_ids.size(1),), -100, device=device, dtype=torch.long)

        # ── Part 2: sampled retrieval token (in loss) ──
        token_embeds = embed_layer(sampled_token_id.unsqueeze(0)).squeeze(0).to(dtype)  # [1, H]
        token_labels = sampled_token_id.clone()  # [1], actual token id

        # ── Part 3: soft tokens from retrieved chunks (masked) ──
        sep_ids = self.tokenizer.encode("\n", add_special_tokens=False, return_tensors="pt").to(device)
        sep_embeds = embed_layer(sep_ids).squeeze(0).to(dtype)

        soft_embeds_list = []
        soft_labels_list = []
        for i in range(num_chunks):
            n = n_tokens_list[i]
            hidden = chunk_hiddens[i].to(device).to(dtype)  # [max_cache, H]
            soft = self.projector(hidden, n=n)  # [n, H]
            soft_embeds_list.append(soft)
            soft_labels_list.append(torch.full((n,), -100, device=device, dtype=torch.long))
            # separator
            soft_embeds_list.append(sep_embeds)
            soft_labels_list.append(torch.full((sep_embeds.size(0),), -100, device=device, dtype=torch.long))

        soft_embeds = torch.cat(soft_embeds_list, dim=0)
        soft_labels = torch.cat(soft_labels_list, dim=0)

        # ── Part 4: instruction + options (masked) ──
        suffix = instruction + "\n" + all_options
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt").to(device)
        suffix_embeds = embed_layer(suffix_ids).squeeze(0).to(dtype)
        suffix_labels = torch.full((suffix_ids.size(1),), -100, device=device, dtype=torch.long)

        # ── Part 5: answer tokens (in loss), optional ──
        if answer_text is not None and isinstance(answer_text, str) and answer_text.strip():
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False, return_tensors="pt").to(device)
            if ans_ids.numel() == 0:
                ans_embeds = torch.empty(0, q_embeds.size(-1), device=device, dtype=dtype)
                ans_labels = torch.empty(0, device=device, dtype=torch.long)
            else:
                ans_embeds = embed_layer(ans_ids).squeeze(0).to(dtype)
                ans_labels = ans_ids.squeeze(0)  # actual ids
        else:
            ans_embeds = torch.empty(0, q_embeds.size(-1), device=device, dtype=dtype)
            ans_labels = torch.empty(0, device=device, dtype=torch.long)

        # ── Concatenate all ──
        all_embeds = torch.cat([
            q_embeds, token_embeds, soft_embeds, suffix_embeds, ans_embeds
        ], dim=0).unsqueeze(0)  # [1, total_len, H]

        all_labels = torch.cat([
            q_labels, token_labels, soft_labels, suffix_labels, ans_labels
        ], dim=0).unsqueeze(0)  # [1, total_len]

        attention_mask = torch.ones(1, all_embeds.size(1), device=device, dtype=torch.long)

        # Track where the answer starts (for generation)
        answer_start = q_embeds.size(0) + token_embeds.size(0) + soft_embeds.size(0) + suffix_embeds.size(0)
        # Position of sampled retrieval token (for extracting h_{t+1})
        sampled_token_pos = q_embeds.size(0)  # the position AT which sampled_token is placed

        return {
            "inputs_embeds": all_embeds,
            "attention_mask": attention_mask,
            "labels": all_labels,
            "n_tokens_list": n_tokens_list,
            "answer_start": answer_start,
            "sampled_token_pos": sampled_token_pos,
        }

    # ──────────────────────────────────────────────────────
    #  GRPO rollout: generate multiple trajectories
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def grpo_rollout(
        self,
        question: str,
        correct_answer: str,
        instruction: str,
        all_options: str,
        chunk_embs: torch.Tensor,       # [num_chunks, 1, hidden_size]
        chunk_hiddens: torch.Tensor,     # [num_chunks, max_hidden_cache, hidden_size]
        num_generations: int = 8,
        max_new_tokens: int = 5,
    ) -> List[Dict]:
        """
        Sample G trajectories, each with:
          - different sampled retrieval token → different retrieval
          - different random n_tokens → different compression
        """
        trajectories = []

        # Rollout needs KV cache, which gradient_checkpointing disables.
        # Toggle off for the duration of rollout, then restore.
        ckpt_was_on = getattr(self.reasoner, "is_gradient_checkpointing", False)
        if ckpt_was_on:
            self.reasoner.gradient_checkpointing_disable()
            self.reasoner.config.use_cache = True

        for _ in range(num_generations):
            # Step 1: sample retrieval token
            sample_info = self.sample_retrieval_token(question)
            sampled_token_id = sample_info["sampled_token_id"]
            query_emb = sample_info["query_emb"]
            kv_cache = sample_info["kv_cache"]

            # Step 2: retrieve (also keep scores for policy)
            q_norm = F.normalize(query_emb.squeeze(0), dim=-1)
            c_norm = F.normalize(chunk_embs.squeeze(1), dim=-1)
            scores_all = c_norm @ q_norm
            top_z = min(self.top_z, scores_all.size(0))
            chunk_scores_top, top_indices = scores_all.topk(top_z)
            retrieved_hiddens = chunk_hiddens[top_indices]
            chunk_embs_top = chunk_embs[top_indices].squeeze(1)  # [Z, hidden]

            # Step 3: sample n_tokens via policy or randomly
            if self.policy is not None:
                n_tokens_list, policy_actions, policy_log_probs = self._sample_n_tokens_via_policy(
                    query_emb.squeeze(0), chunk_embs_top, chunk_scores_top
                )
                policy_actions = policy_actions.detach()
                policy_log_probs_old = policy_log_probs.detach()
            else:
                n_tokens_list = self._sample_n_tokens_list(retrieved_hiddens.size(0))
                policy_actions = None
                policy_log_probs_old = None

            # Step 4: build input (without answer) and generate
            embed_layer = self.reasoner.get_input_embeddings()
            dtype = self.embed_dtype
            device = self.device

            # Build soft tokens + suffix, append to KV cache
            sep_ids = self.tokenizer.encode("\n", add_special_tokens=False, return_tensors="pt").to(device)
            sep_embeds = embed_layer(sep_ids).squeeze(0).to(dtype)

            remaining_embeds = []
            for i in range(retrieved_hiddens.size(0)):
                n = n_tokens_list[i]
                hidden = retrieved_hiddens[i].to(device).to(dtype)
                soft = self.projector(hidden, n=n)
                remaining_embeds.append(soft)
                remaining_embeds.append(sep_embeds)

            suffix = instruction + "\n" + all_options
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt").to(device)
            suffix_embeds = embed_layer(suffix_ids).squeeze(0).to(dtype)
            remaining_embeds.append(suffix_embeds)

            remaining = torch.cat(remaining_embeds, dim=0).unsqueeze(0)  # [1, L, H]
            remaining_mask = torch.ones(1, remaining.size(1), device=device, dtype=torch.long)

            # Extend KV cache with remaining context
            remaining_out = self.reasoner(
                inputs_embeds=remaining,
                past_key_values=kv_cache,
                use_cache=True,
            )
            full_kv = remaining_out.past_key_values

            # Generate answer
            # Get the last logits and generate autoregressively
            full_attention_len = full_kv[0][0].size(2)  # total KV length
            full_mask = torch.ones(1, full_attention_len, device=device, dtype=torch.long)

            # Sample answer with temperature=1.0 during rollout for trajectory diversity
            answer_temp = 1.0
            generated_ids = []
            next_logits = remaining_out.logits[0, -1, :]
            cur_kv = full_kv

            for _ in range(max_new_tokens):
                probs = F.softmax(next_logits / answer_temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [1]
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token.item())

                next_embeds = embed_layer(next_token.unsqueeze(0)).to(dtype)  # [1,1,H]
                step_out = self.reasoner(
                    inputs_embeds=next_embeds,
                    past_key_values=cur_kv,
                    use_cache=True,
                )
                next_logits = step_out.logits[0, -1, :]
                cur_kv = step_out.past_key_values

            answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            reward = self._compute_reward(answer_text, correct_answer)

            trajectories.append({
                "sampled_token_id": sampled_token_id,
                "top_indices": top_indices,
                "n_tokens_list": n_tokens_list,
                "answer_text": answer_text,
                "reward": reward,
                # policy-related (None if no policy)
                "policy_actions": policy_actions,
                "policy_log_probs_old": policy_log_probs_old,
                "query_emb_for_policy": query_emb.squeeze(0).detach() if self.policy is not None else None,
                "chunk_embs_top": chunk_embs_top.detach() if self.policy is not None else None,
                "chunk_scores_top": chunk_scores_top.detach() if self.policy is not None else None,
            })

        # Restore checkpointing if we toggled it off above.
        if ckpt_was_on:
            self.reasoner.gradient_checkpointing_enable()
            self.reasoner.config.use_cache = False

        return trajectories

    # ──────────────────────────────────────────────────────
    #  GRPO loss computation
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_old_logprobs(
        self,
        question: str,
        instruction: str,
        all_options: str,
        chunk_hiddens_all: torch.Tensor,
        trajectories: List[Dict],
    ) -> List[torch.Tensor]:
        """Compute old log probs for all trajectories (detached, no grad)."""
        old_logprobs = []
        for traj in trajectories:
            retrieved_hiddens = chunk_hiddens_all[traj["top_indices"]]
            seq = self.build_full_sequence(
                question=question,
                sampled_token_id=traj["sampled_token_id"],
                chunk_hiddens=retrieved_hiddens,
                instruction=instruction,
                all_options=all_options,
                answer_text=traj["answer_text"],
                n_tokens_list=traj["n_tokens_list"],
            )
            outputs = self.reasoner(
                inputs_embeds=seq["inputs_embeds"],
                attention_mask=seq["attention_mask"],
            )
            logits = outputs.logits
            labels = seq["labels"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            valid_mask = (shift_labels != -100)
            
            # FIX: gather only valid positions to avoid numerical issues with invalid indices
            gather_labels = shift_labels.clone()
            gather_labels[~valid_mask] = 0  # placeholder for gather
            per_token_logps = log_probs.gather(2, gather_labels.unsqueeze(2)).squeeze(2)
            # Explicitly mask invalid positions to 0
            per_token_logps = per_token_logps * valid_mask.float()
            old_logprobs.append(per_token_logps.detach())  # detached
        return old_logprobs

    def grpo_loss(
        self,
        question: str,
        instruction: str,
        all_options: str,
        chunk_hiddens_all: torch.Tensor,
        trajectories: List[Dict],
        old_logprobs: List[torch.Tensor],
        epsilon: float = 0.2,
        policy_entropy_coef: float = 0.0,
        chunk_embs_all: torch.Tensor = None,     # [num_chunks, 1, H] for aux loss
        dense_scores: torch.Tensor = None,        # [num_chunks] target from dense retriever
        aux_lambda: float = 0.0,                  # weight of aux retrieval loss
        aux_temp: float = 1.0,
    ) -> Dict:
        """
        Compute GRPO loss over trajectories.
        old_logprobs: pre-computed log probs from before this gradient step.
        """
        rewards = torch.tensor([t["reward"] for t in trajectories], device=self.device)
        mean_r = rewards.mean()
        # Stable group-relative normalization under sparse rewards.
        std_r = max(rewards.std().item(), 1e-4)
        advantages = (rewards - mean_r) / (std_r + 1e-8)
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)

        total_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0
        # Diagnostic entropies accumulated across trajectories (for monitoring only)
        llm_ent_sum = 0.0
        llm_ent_count = 0
        policy_ent_sum = 0.0
        policy_ent_count = 0

        for i, traj in enumerate(trajectories):
            retrieved_hiddens = chunk_hiddens_all[traj["top_indices"]]

            seq = self.build_full_sequence(
                question=question,
                sampled_token_id=traj["sampled_token_id"],
                chunk_hiddens=retrieved_hiddens,
                instruction=instruction,
                all_options=all_options,
                answer_text=traj["answer_text"],
                n_tokens_list=traj["n_tokens_list"],
            )

            # Forward with current (updated) weights
            need_hidden = (aux_lambda > 0 and chunk_embs_all is not None and dense_scores is not None)
            outputs = self.reasoner(
                inputs_embeds=seq["inputs_embeds"],
                attention_mask=seq["attention_mask"],
                output_hidden_states=need_hidden,
            )
            logits = outputs.logits
            labels = seq["labels"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            valid_mask = (shift_labels != -100)

            if valid_mask.sum() == 0:
                continue

            # LLM per-token entropy over vocab at valid (answer + sampled_token) positions
            with torch.no_grad():
                probs = log_probs.exp()
                per_pos_ent = -(probs * log_probs).sum(dim=-1)     # [1, seq-1]
                valid_ent = per_pos_ent[valid_mask]                # [num_valid_positions]
                if valid_ent.numel() > 0:
                    llm_ent_sum += valid_ent.mean().item()
                    llm_ent_count += 1

            gather_labels = shift_labels.clone()
            gather_labels[~valid_mask] = 0
            cur_per_token_logps = log_probs.gather(2, gather_labels.unsqueeze(2)).squeeze(2)
            cur_per_token_logps = cur_per_token_logps * valid_mask.float()

            # Per-token ratio for answer/sampled-token positions
            old_lp = old_logprobs[i]
            per_token_ratio = torch.exp(cur_per_token_logps - old_lp)

            adv = advantages[i]
            surr1 = per_token_ratio * adv
            surr2 = torch.clamp(per_token_ratio, 1 - epsilon, 1 + epsilon) * adv
            per_token_loss = -torch.min(surr1, surr2)

            answer_loss = (per_token_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
            traj_loss = answer_loss

            # ── Policy loss (if policy is in use) ──
            if self.policy is not None and traj.get("policy_actions") is not None:
                cur_policy_logps, cur_policy_ent = self._compute_policy_logprobs(
                    query_emb=traj["query_emb_for_policy"],
                    chunk_embs_top=traj["chunk_embs_top"],
                    chunk_scores=traj["chunk_scores_top"],
                    actions=traj["policy_actions"],
                    return_entropy=True,
                )
                old_policy_lp = traj["policy_log_probs_old"]
                policy_ratio = torch.exp(cur_policy_logps - old_policy_lp)
                p_surr1 = policy_ratio * adv
                p_surr2 = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon) * adv
                policy_loss = (-torch.min(p_surr1, p_surr2)).mean()
                # Entropy bonus (subtract mean entropy, scaled by coef)
                entropy_bonus = policy_entropy_coef * cur_policy_ent.mean()
                traj_loss = traj_loss + policy_loss - entropy_bonus
                with torch.no_grad():
                    policy_ent_sum += cur_policy_ent.mean().item()
                    policy_ent_count += 1

            # ── Aux retrieval loss: align our_scores with dense_scores ──
            if need_hidden:
                # Extract h_{t+1}: hidden state AT sampled_token position (i.e. output after processing it)
                h_t1 = outputs.hidden_states[-1][0, seq["sampled_token_pos"], :]  # [H]
                # Compute our scores across ALL chunks
                chunk_lasts = chunk_embs_all.squeeze(1).to(h_t1.dtype).to(h_t1.device)  # [num_chunks, H]
                q_norm = F.normalize(h_t1, dim=-1)
                c_norm = F.normalize(chunk_lasts, dim=-1)
                our_scores = c_norm @ q_norm            # [num_chunks]
                our_logp = F.log_softmax(our_scores / aux_temp, dim=0)
                target = F.softmax(dense_scores.to(our_scores.device) / aux_temp, dim=0)
                aux_loss = F.kl_div(our_logp, target, reduction="sum")
                traj_loss = traj_loss + aux_lambda * aux_loss

            total_loss = total_loss + traj_loss
            num_valid += 1

        loss = total_loss / max(num_valid, 1)

        return {
            "loss": loss,
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "llm_entropy": (llm_ent_sum / llm_ent_count) if llm_ent_count else 0.0,
            "policy_entropy": (policy_ent_sum / policy_ent_count) if policy_ent_count else 0.0,
        }

    # ──────────────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        question: str,
        instruction: str,
        all_options: str,
        chunk_embs: torch.Tensor,
        chunk_hiddens: torch.Tensor,
        n_tokens_fixed: int = None,
        max_new_tokens: int = 5,
    ) -> str:
        """
        Inference: SAME pipeline structure as training.
          1. Encode question → argmax next token (greedy, matches training structure)
          2. Append argmax token → h_{t+1} for retrieval
          3. Retrieve → projector → soft tokens
          4. Generate answer
        """
        # KV cache is required; toggle gradient checkpointing off for inference.
        ckpt_was_on = getattr(self.reasoner, "is_gradient_checkpointing", False)
        if ckpt_was_on:
            self.reasoner.gradient_checkpointing_disable()
            self.reasoner.config.use_cache = True

        embed_layer = self.reasoner.get_input_embeddings()
        dtype = self.embed_dtype
        device = self.device

        # ── Step 1: encode question ──
        q_ids = self.tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(device)
        q_embeds = embed_layer(q_ids).to(dtype)
        q_out = self.reasoner(inputs_embeds=q_embeds, use_cache=True)
        kv_cache = q_out.past_key_values

        # ── Step 2: argmax next token (deterministic version of training's sampling) ──
        next_logits = q_out.logits[0, -1, :]
        retrieval_token_id = next_logits.argmax(dim=-1, keepdim=True)  # [1]

        # Forward one more step with this token to get h_{t+1}
        token_embeds = embed_layer(retrieval_token_id.unsqueeze(0)).to(dtype)  # [1,1,H]
        token_out = self.reasoner(
            inputs_embeds=token_embeds,
            past_key_values=kv_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        h_t1 = token_out.hidden_states[-1][0, -1, :]
        kv_cache = token_out.past_key_values

        # ── Step 3: retrieve using h_{t+1} (matches training) ──
        q_norm = F.normalize(h_t1, dim=-1)
        c_norm = F.normalize(chunk_embs.squeeze(1), dim=-1)
        scores_all = c_norm @ q_norm
        top_z = min(self.top_z, scores_all.size(0))
        chunk_scores_top, top_indices = scores_all.topk(top_z)
        retrieved_hiddens = chunk_hiddens[top_indices]
        chunk_embs_top = chunk_embs[top_indices].squeeze(1)

        # Build soft tokens + suffix (structure identical to training build_full_sequence)
        sep_ids = self.tokenizer.encode("\n", add_special_tokens=False, return_tensors="pt").to(device)
        sep_embeds = embed_layer(sep_ids).squeeze(0).to(dtype)

        # n_tokens: use policy (greedy/argmax) if available, else random/fixed
        if n_tokens_fixed is not None:
            n_tokens_list = [n_tokens_fixed] * retrieved_hiddens.size(0)
        elif self.policy is not None:
            # Deterministic argmax over policy
            Z = retrieved_hiddens.size(0)
            n_choices = self.policy.n_choices
            q = h_t1.to(device).to(torch.float32).unsqueeze(0)
            c = chunk_embs_top.to(device).to(torch.float32).unsqueeze(0)
            s = chunk_scores_top.to(device).to(torch.float32).unsqueeze(0)
            n_tokens_list = []
            actions_so_far = []
            for i in range(Z):
                prev_choices = torch.full((1, Z), n_choices, device=device, dtype=torch.long)
                for k, a in enumerate(actions_so_far):
                    prev_choices[0, k] = a
                logits = self.policy(q, c, s, prev_choices)
                action = logits[0, i].argmax().item()
                actions_so_far.append(action)
                n_tokens_list.append(action + 1)
        else:
            n_tokens_list = self._sample_n_tokens_list(retrieved_hiddens.size(0))

        remaining_embeds = []
        for i in range(retrieved_hiddens.size(0)):
            n_i = n_tokens_list[i]
            hidden = retrieved_hiddens[i].to(device).to(dtype)
            soft = self.projector(hidden, n=n_i)
            remaining_embeds.append(soft)
            remaining_embeds.append(sep_embeds)

        suffix = instruction + "\n" + all_options
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt").to(device)
        suffix_embeds = embed_layer(suffix_ids).squeeze(0).to(dtype)
        remaining_embeds.append(suffix_embeds)

        remaining = torch.cat(remaining_embeds, dim=0).unsqueeze(0)
        remaining_out = self.reasoner(inputs_embeds=remaining, past_key_values=kv_cache, use_cache=True)
        cur_kv = remaining_out.past_key_values
        next_logits = remaining_out.logits[0, -1, :]

        # ── Step 4: greedy generation ──
        generated_ids = []
        for _ in range(max_new_tokens):
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            generated_ids.append(next_token.item())
            next_embeds = embed_layer(next_token.unsqueeze(0)).to(dtype)
            step_out = self.reasoner(inputs_embeds=next_embeds, past_key_values=cur_kv, use_cache=True)
            next_logits = step_out.logits[0, -1, :]
            cur_kv = step_out.past_key_values

        # Restore checkpointing if we toggled it off above.
        if ckpt_was_on:
            self.reasoner.gradient_checkpointing_enable()
            self.reasoner.config.use_cache = False

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ──────────────────────────────────────────────────────
    #  Reward
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_reward(prediction: str, correct_answer: str) -> float:
        def extract(text):
            text = text.strip().lower()
            m = re.search(r"\(([a-d])\)", text)
            if m:
                return m.group(1)
            m = re.search(r"\b([a-d])\b", text)
            return m.group(1) if m else ""
        return 1.0 if extract(prediction) == extract(correct_answer) and extract(correct_answer) != "" else 0.0
