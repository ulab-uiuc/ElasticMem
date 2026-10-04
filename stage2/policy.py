"""
Stage 2 Policy Network: a Transformer-based policy that decides
how many latent tokens to allocate to each chunk.

Sequential decision making with causal self-attention:
    State at step i: [query_emb, (chunk_0, n_0), (chunk_1, n_1), ..., chunk_i]
    Action: n_i ∈ [1, n_choices] (classification)

Uses causal mask so each chunk decision only depends on earlier decisions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BudgetPolicy(nn.Module):
    """
    Small Transformer policy.

    Input features:
        query_emb:      [B, hidden_in]  query latent embedding (from stage 1 reasoner)
        chunk_embs:     [B, Z, hidden_in]  retrieved chunk embeddings
        chunk_scores:   [B, Z]  retrieval cosine similarities
    Output (sequentially for each chunk):
        logits over n_choices
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_choices: int = 10,
        max_chunks: int = 30,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_choices = n_choices
        self.max_chunks = max_chunks

        # Project query / chunk embeddings into shared hidden space
        self.query_proj = nn.Linear(input_dim, hidden_size)
        self.chunk_proj = nn.Linear(input_dim, hidden_size)

        # Scalar feature projection (retrieval score)
        self.score_proj = nn.Linear(1, hidden_size)

        # Embedding for previous n choices
        self.n_emb = nn.Embedding(n_choices + 1, hidden_size)  # +1 for "not yet decided"

        # Positional encoding (over sequence positions 0..Z)
        self.pos_emb = nn.Embedding(max_chunks + 2, hidden_size)

        # Segment embedding: 0 = query, 1 = chunk
        self.seg_emb = nn.Embedding(2, hidden_size)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action head: output logits over n_choices
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, n_choices),
        )

    def _build_sequence(
        self,
        query_emb: torch.Tensor,      # [B, hidden_in]
        chunk_embs: torch.Tensor,     # [B, Z, hidden_in]
        chunk_scores: torch.Tensor,   # [B, Z]
        prev_choices: torch.Tensor,   # [B, Z]   filled with n_choices (i.e. "not yet") for future positions
    ) -> torch.Tensor:
        """
        Build the input sequence:
            pos 0:    query token
            pos 1:    chunk_0 token (with score + prev choice info if decided)
            pos 2:    chunk_1 token
            ...
            pos Z:    chunk_{Z-1} token
        Shape: [B, Z+1, hidden_size]
        """
        B, Z, _ = chunk_embs.shape
        device = chunk_embs.device

        # Query token
        q_tok = self.query_proj(query_emb)                             # [B, H]
        q_tok = q_tok + self.seg_emb.weight[0]                         # segment = 0
        q_tok = q_tok + self.pos_emb.weight[0]                         # pos = 0
        q_tok = q_tok.unsqueeze(1)                                     # [B, 1, H]

        # Chunk tokens: chunk_emb + score + prev_choice + pos + segment
        c_tok = self.chunk_proj(chunk_embs)                            # [B, Z, H]
        c_tok = c_tok + self.score_proj(chunk_scores.unsqueeze(-1))    # [B, Z, H]
        c_tok = c_tok + self.n_emb(prev_choices)                       # [B, Z, H]
        # segment = 1 for all chunks
        c_tok = c_tok + self.seg_emb.weight[1].unsqueeze(0).unsqueeze(0)
        # position = 1..Z
        positions = torch.arange(1, Z + 1, device=device).unsqueeze(0).expand(B, -1)
        c_tok = c_tok + self.pos_emb(positions)

        # Concatenate: [q | c_0, c_1, ..., c_{Z-1}]
        seq = torch.cat([q_tok, c_tok], dim=1)                         # [B, Z+1, H]
        return seq

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        """Causal mask: position i can only attend to 0..i."""
        mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        return mask  # True = masked out

    def forward(
        self,
        query_emb: torch.Tensor,      # [B, hidden_in]
        chunk_embs: torch.Tensor,     # [B, Z, hidden_in]
        chunk_scores: torch.Tensor,   # [B, Z]
        prev_choices: torch.Tensor,   # [B, Z]
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, Z, n_choices]   logits for each chunk's n decision
        """
        seq = self._build_sequence(query_emb, chunk_embs, chunk_scores, prev_choices)
        L = seq.size(1)
        mask = self._causal_mask(L, seq.device)

        out = self.transformer(seq, mask=mask)   # [B, Z+1, H]

        # chunk_i decision uses the output at position i+1 (where chunk_i lives).
        chunk_outputs = out[:, 1:, :]             # [B, Z, H]
        logits = self.head(chunk_outputs)         # [B, Z, n_choices]
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
