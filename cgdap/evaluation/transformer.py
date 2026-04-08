"""Encoder-only Transformer HAR classifier.

Architecture (per modality):
    Spectrogram [B, 3, F, T]
    -> Linear patch embedding over time (F channels flattened as feature dim)
    -> Positional encoding
    -> Transformer encoder (n_layers, n_heads)
    -> CLS token output: [B, d_model]

Late fusion: concatenate modality CLS tokens -> FC -> logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SpectrogramPatchEmbed(nn.Module):
    """Project each time frame of a spectrogram into a d_model embedding."""

    def __init__(self, in_channels: int, freq_bins: int, d_model: int) -> None:
        super().__init__()
        in_dim = in_channels * freq_bins
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)   # [B, T, C*F]
        return self.proj(x)                                 # [B, T, d_model]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.dropout(x + self.pe[:, : x.shape[1]])


class ModalityTransformerEncoder(nn.Module):
    """Per-modality Transformer encoder producing a single CLS vector."""

    def __init__(
        self,
        in_channels: int,
        freq_bins: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_embed = SpectrogramPatchEmbed(in_channels, freq_bins, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        B = x.shape[0]
        tokens = self.patch_embed(x)                                  # [B, T, d_model]
        tokens = self.pos_enc(tokens)
        cls = self.cls_token.expand(B, -1, -1)                        # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)                      # [B, 1+T, d_model]
        out = self.transformer(tokens)                                 # [B, 1+T, d_model]
        return self.norm(out[:, 0])                                    # [B, d_model]  CLS


class HATransformerClassifier(nn.Module):
    """Encoder-only Transformer HAR classifier with late fusion.

    Args:
        n_classes:    number of activity classes
        modalities:   e.g. ["acc", "gyr"]
        in_channels:  spectral channels (3 for xyz)
        freq_bins:    F dimension of the spectrogram (derived from config)
        d_model:      Transformer hidden dim
        n_heads:      multi-head attention heads
        n_layers:     Transformer encoder depth
        ffn_dim:      feedforward layer dim (typically 4 * d_model)
        dropout:      dropout probability
    """

    def __init__(
        self,
        n_classes: int,
        modalities: list[str],
        in_channels: int = 3,
        freq_bins: int = 129,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = modalities

        self.encoders = nn.ModuleDict({
            mod: ModalityTransformerEncoder(
                in_channels=in_channels,
                freq_bins=freq_bins,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for mod in modalities
        })

        fusion_dim = d_model * len(modalities)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    @classmethod
    def from_config(cls, cfg: DictConfig, freq_bins: int = 129) -> "HATransformerClassifier":
        e = cfg.get("evaluation", {})
        tr = e.get("transformer", {})
        return cls(
            n_classes=len(cfg.dataset.activities),
            modalities=list(cfg.model.modalities),
            in_channels=int(cfg.model.in_channels),
            freq_bins=freq_bins,
            d_model=int(tr.get("d_model", 256)),
            n_heads=int(tr.get("n_heads", 4)),
            n_layers=int(tr.get("n_layers", 4)),
            ffn_dim=int(tr.get("ffn_dim", 1024)),
            dropout=float(tr.get("dropout", 0.1)),
        )

    def forward(self, batch: dict[str, dict]) -> torch.Tensor:
        """
        Args:
            batch: PairedDataset item {mod: {"spectrogram": [B,3,F,T]}}

        Returns:
            logits: [B, n_classes]
        """
        feats = [self.encoders[mod](batch[mod]["spectrogram"]) for mod in self.modalities]
        fused = torch.cat(feats, dim=-1)    # [B, d_model * n_mod]
        return self.classifier(fused)
