"""Cross-attention-based condition embedder.

Replaces naive scalar-replication-then-concatenation with:

    1. Per-metric linear projections -> token sequence [B, n_tokens, d_model]
    2. One-hot label projection -> appended tokens
    3. CrossAttentionBlock: spatial features (Q) attend to condition tokens (K, V)

This allows the network to dynamically focus on relevant metrics at
different spatial locations without inflating the input tensor size.

Reference: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    Q = flattened spatial feature map from denoiser intermediate layer
    K, V = condition token sequence from this embedder
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from cgdap.models.base import BaseConditionEmbedder, register_embedder


# ---------------------------------------------------------------------------
# Cross-attention building block
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """Single cross-attention layer.

    Spatial features attend to condition tokens.

    Args:
        d_model:  feature dimension for Q (spatial) and K/V (condition)
        n_heads:  number of attention heads
        dropout:  attention dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # [B, H*W, d_model]  spatial queries
        context: torch.Tensor,    # [B, n_tokens, d_model]  condition K/V
    ) -> torch.Tensor:
        """Returns updated spatial features, same shape as x."""
        B, L, D = x.shape
        N = context.shape[1]

        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, L, d_h]
        K = self.k_proj(context).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(context).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, L, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                 # [B, h, L, d_h]
        out = out.transpose(1, 2).contiguous().view(B, L, D)        # [B, L, D]
        out = self.out_proj(out)

        # Residual + LayerNorm
        return self.norm(x + out)


# ---------------------------------------------------------------------------
# Condition Embedder
# ---------------------------------------------------------------------------


@register_embedder("cross_attention")
class CrossAttentionConditionEmbedder(BaseConditionEmbedder):
    """Condition embedder that produces attention K/V token sequence.

    Takes:
        metrics [B, n_metrics]       -- per-modality differentiable metrics
        labels_onehot [B, n_classes] -- activity one-hot encoding

    Produces:
        tokens [B, n_cond_tokens, d_model] -- K and V source for CrossAttentionBlock

    Architecture:
        Each metric scalar -> linear(1, d_model) -> token
        Label vector -> linear(n_classes, d_model) -> token(s)
        All tokens concatenated -> [B, n_metrics + 1, d_model]
        Optional learned positional encoding applied
    """

    def __init__(
        self,
        n_metrics: int,
        n_classes: int,
        d_model: int,
        n_heads: int,
        n_cond_tokens: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_metrics = n_metrics
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_cond_tokens = n_cond_tokens

        # Each metric independently projected to d_model
        self.metric_projs = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_metrics)
        ])

        # Label embedding
        self.label_proj = nn.Linear(n_classes, d_model)

        # Optional learned positional encoding for token sequence
        n_raw_tokens = n_metrics + 1  # metrics + label
        self.pos_emb = nn.Parameter(torch.randn(1, n_raw_tokens, d_model) * 0.02)

        # If we want to project from n_raw_tokens to n_cond_tokens, use a linear mixer
        # Otherwise set n_cond_tokens == n_raw_tokens
        self.n_raw_tokens = n_raw_tokens
        if n_cond_tokens != n_raw_tokens:
            self.token_mixer = nn.Linear(n_raw_tokens, n_cond_tokens)
        else:
            self.token_mixer = None

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "CrossAttentionConditionEmbedder":
        c = cfg.model.condition
        return cls(
            n_metrics=int(c.n_metrics),
            n_classes=int(cfg.model.num_classes),
            d_model=int(c.d_model),
            n_heads=int(c.n_heads),
            n_cond_tokens=int(c.n_cond_tokens),
            dropout=float(c.dropout),
        )

    def forward(
        self,
        metrics: torch.Tensor,       # [B, n_metrics]
        labels_onehot: torch.Tensor, # [B, n_classes]
    ) -> torch.Tensor:
        """Return condition tokens [B, n_cond_tokens, d_model]."""
        B = metrics.shape[0]

        # Project each metric independently: [B, 1] -> [B, d_model]
        metric_tokens = []
        for i, proj in enumerate(self.metric_projs):
            tok = proj(metrics[:, i : i + 1])   # [B, d_model]
            metric_tokens.append(tok)

        # Project labels: [B, n_classes] -> [B, d_model]
        label_tok = self.label_proj(labels_onehot.float())   # [B, d_model]
        metric_tokens.append(label_tok)

        # Stack into sequence: [B, n_raw_tokens, d_model]
        tokens = torch.stack(metric_tokens, dim=1)
        tokens = tokens + self.pos_emb
        tokens = self.dropout(self.norm(tokens))

        # Optional token count mixing
        if self.token_mixer is not None:
            # [B, d_model, n_raw_tokens] -> [B, d_model, n_cond_tokens] -> [B, n_cond_tokens, d_model]
            tokens = self.token_mixer(tokens.transpose(1, 2)).transpose(1, 2)

        return tokens   # [B, n_cond_tokens, d_model]

    def extra_repr(self) -> str:
        return (
            f"n_metrics={self.n_metrics}, n_classes={self.n_classes}, "
            f"d_model={self.d_model}, n_cond_tokens={self.n_cond_tokens}"
        )
