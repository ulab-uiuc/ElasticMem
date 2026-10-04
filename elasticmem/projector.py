"""
Simple MLP projector: maps base model hidden states to reasoner input embedding space.

No additional compression module - just space translation.
The "compression" is already done by the LLM's forward pass (each hidden state
encodes prior context via causal attention).

Pipeline:
    chunk → LLM forward → last N hidden states → MLP → N soft tokens
"""
import torch
import torch.nn as nn


class LatentProjector(nn.Module):
    """
    2-layer MLP with GELU + LayerNorm.

    Input:  [N, hidden_size] or [B, N, hidden_size]  (hidden states from base model)
    Output: same shape, projected into input embedding space
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        # extra args kept for backward-compat; unused
        max_queries: int = None,
        num_heads: int = None,
        num_layers: int = None,
        ffn_mult: int = None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, n: int = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [max_cache, hidden_size] or [B, max_cache, hidden_size]
            n: number of output tokens (select last n positions)
        Returns:
            [n, hidden_size] or [B, n, hidden_size]
        """
        if n is not None:
            # Take last n positions
            if hidden_states.dim() == 2:
                hidden_states = hidden_states[-n:]
            else:
                hidden_states = hidden_states[:, -n:]
        return self.ln(self.mlp(hidden_states))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
