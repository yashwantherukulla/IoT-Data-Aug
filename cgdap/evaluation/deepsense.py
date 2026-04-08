"""DeepSense-style multimodal HAR classifier.

Architecture (per modality):
    Spectrogram [B, 3, F, T]
    -> Conv1d over time (per freq channel)
    -> GRU across time windows
    -> [B, hidden_dim]

Late fusion: concatenate all modality representations -> FC -> logits.

Reference: Yao et al., "DeepSense: A Unified Deep Learning Framework for
Time-Series Mobile Sensing Data Processing", WWW 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ModalityEncoderCNN(nn.Module):
    """CNN feature extractor for one modality spectrogram."""

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: int = 64,
        kernel_size: int = 5,
        n_conv_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(n_conv_layers):
            layers += [
                nn.Conv2d(ch, conv_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
                nn.BatchNorm2d(conv_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ]
            ch = conv_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, None))   # collapse F dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, F, T]
        h = self.conv(x)       # [B, conv_ch, F, T]
        h = self.pool(h)       # [B, conv_ch, 1, T]
        return h.squeeze(2)    # [B, conv_ch, T]


class DeepSenseClassifier(nn.Module):
    """Multimodal HAR classifier following the DeepSense architecture.

    Each modality goes through a CNN encoder then a GRU.
    Late fusion: concatenate GRU final states -> FC head.
    """

    def __init__(
        self,
        n_classes: int,
        modalities: list[str],
        in_channels: int = 3,
        conv_channels: int = 64,
        n_conv_layers: int = 3,
        rnn_hidden: int = 128,
        n_rnn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.modalities = modalities

        self.encoders = nn.ModuleDict({
            mod: ModalityEncoderCNN(in_channels, conv_channels, n_conv_layers=n_conv_layers, dropout=dropout)
            for mod in modalities
        })

        self.rnns = nn.ModuleDict({
            mod: nn.GRU(
                input_size=conv_channels,
                hidden_size=rnn_hidden,
                num_layers=n_rnn_layers,
                batch_first=True,
                dropout=dropout if n_rnn_layers > 1 else 0.0,
                bidirectional=False,
            )
            for mod in modalities
        })

        fusion_dim = rnn_hidden * len(modalities)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DeepSenseClassifier":
        e = cfg.get("evaluation", {})
        ds = e.get("deepsense", {})
        return cls(
            n_classes=len(cfg.dataset.activities),
            modalities=list(cfg.model.modalities),
            in_channels=int(cfg.model.in_channels),
            conv_channels=int(ds.get("conv_channels", 64)),
            n_conv_layers=int(ds.get("n_conv_layers", 3)),
            rnn_hidden=int(ds.get("rnn_hidden", 128)),
            n_rnn_layers=int(ds.get("n_rnn_layers", 2)),
            dropout=float(ds.get("dropout", 0.2)),
        )

    def forward(self, batch: dict[str, dict]) -> torch.Tensor:
        """
        Args:
            batch: PairedDataset item with {mod: {"spectrogram": [B,3,F,T]}}

        Returns:
            logits: [B, n_classes]
        """
        mod_feats: list[torch.Tensor] = []
        for mod in self.modalities:
            x = batch[mod]["spectrogram"]              # [B, 3, F, T]
            h = self.encoders[mod](x)                  # [B, conv_ch, T]
            h = h.transpose(1, 2)                      # [B, T, conv_ch]
            _, h_n = self.rnns[mod](h)                 # h_n: [n_layers, B, hidden]
            feat = h_n[-1]                             # [B, hidden]
            mod_feats.append(feat)

        fused = torch.cat(mod_feats, dim=-1)           # [B, hidden * n_mod]
        return self.classifier(fused)                  # [B, n_classes]
