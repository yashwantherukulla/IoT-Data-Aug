"""Conditional U-Net denoiser for spectrogram diffusion.

Architecture:
    Input: x_t [B, in_channels, F, T] (noisy spectrogram)
    Timestep: t [B] -> sinusoidal embedding -> MLP -> temb [B, temb_dim]
    Condition: cond [B, n_tokens, d_model] from CrossAttentionConditionEmbedder

    Encoder: 3 DownBlocks (stride-2 conv on both F and T)
    Bottleneck: ResBlock + CrossAttention + ResBlock
    Decoder: 3 UpBlocks with skip connections (bilinear upsample + conv)
    Output conv: pred_noise [B, in_channels, F, T]

Timestep injection: AdaGN (Adaptive Group Norm) -- learned scale/shift from temb.
Cross-attention: injected at configurable depths (default: bottleneck + deepest skip).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from cgdap.models.base import BaseDenoiser, register_denoiser
from cgdap.models.condition import CrossAttentionBlock


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        timesteps: [B] integer timestep indices
        dim: embedding dimension

    Returns:
        [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepMLP(nn.Module):
    """Project sinusoidal embedding to temb_dim via 2-layer MLP."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ---------------------------------------------------------------------------
# Adaptive Group Norm (AdaGN)
# ---------------------------------------------------------------------------


class AdaptiveGroupNorm(nn.Module):
    """GroupNorm with learned scale/shift conditioned on timestep embedding."""

    def __init__(self, num_groups: int, num_channels: int, temb_dim: int) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(temb_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  temb: [B, temb_dim]
        scale_shift = self.proj(temb)[:, :, None, None]   # [B, 2C, 1, 1]
        scale, shift = scale_shift.chunk(2, dim=1)
        return self.gn(x) * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with AdaGN timestep injection and optional cross-attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_dim: int,
        num_groups: int = 8,
        dropout: float = 0.1,
        d_model: int | None = None,
        n_heads: int = 4,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.norm1 = AdaptiveGroupNorm(min(num_groups, in_ch), in_ch, temb_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(min(num_groups, out_ch), out_ch, temb_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if use_cross_attn and d_model is not None:
            # Project spatial features to d_model for Q in cross-attention
            self.spatial_proj = nn.Conv2d(out_ch, d_model, 1)
            self.cross_attn = CrossAttentionBlock(d_model, n_heads, dropout=dropout)
            self.out_proj = nn.Conv2d(d_model, out_ch, 1)
        else:
            self.spatial_proj = None
            self.cross_attn = None
            self.out_proj = None

    def forward(
        self,
        x: torch.Tensor,                   # [B, C, H, W]
        temb: torch.Tensor,                # [B, temb_dim]
        condition: torch.Tensor | None,    # [B, n_tokens, d_model]
    ) -> torch.Tensor:
        h = self.act(self.norm1(x, temb))
        h = self.conv1(h)
        h = self.dropout(self.act(self.norm2(h, temb)))
        h = self.conv2(h)
        h = h + self.shortcut(x)

        if self.use_cross_attn and condition is not None and self.cross_attn is not None:
            B, C, H, W = h.shape
            # Project to d_model -> flatten spatial -> cross-attend -> reshape back
            q = self.spatial_proj(h)       # [B, d_model, H, W]
            q = q.flatten(2).transpose(1, 2)   # [B, H*W, d_model]
            q = self.cross_attn(q, condition)  # [B, H*W, d_model]
            q = q.transpose(1, 2).view(B, -1, H, W)
            h = h + self.out_proj(q)

        return h


class DownBlock(nn.Module):
    """Downsampling block: n_res ResBlocks then stride-2 conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_dim: int,
        n_res: int = 2,
        dropout: float = 0.1,
        use_cross_attn: bool = False,
        d_model: int | None = None,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_ch if i == 0 else out_ch,
                out_ch,
                temb_dim=temb_dim,
                dropout=dropout,
                use_cross_attn=use_cross_attn,
                d_model=d_model,
                n_heads=n_heads,
            )
            for i in range(n_res)
        ])
        # Stride-2 on both F and T
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.res_blocks:
            x = blk(x, temb, condition)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block: bilinear upsample then n_res ResBlocks with skip."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        temb_dim: int,
        n_res: int = 2,
        dropout: float = 0.1,
        use_cross_attn: bool = False,
        d_model: int | None = None,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res_blocks = nn.ModuleList([
            ResBlock(
                (in_ch + skip_ch) if i == 0 else out_ch,
                out_ch,
                temb_dim=temb_dim,
                dropout=dropout,
                use_cross_attn=use_cross_attn,
                d_model=d_model,
                n_heads=n_heads,
            )
            for i in range(n_res)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        temb: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        # Handle odd spatial sizes from stride-2 downsampling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for blk in self.res_blocks:
            x = blk(x, temb, condition)
        return x


# ---------------------------------------------------------------------------
# ConditionalUNet
# ---------------------------------------------------------------------------


@register_denoiser("unet")
class ConditionalUNet(BaseDenoiser):
    """3-down, bottleneck, 3-up U-Net with cross-attention conditioning.

    Timestep embedding injected via AdaGN at every ResBlock.
    Cross-attention injected at configurable depths.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: list[int] | None = None,
        n_res_blocks: int = 2,
        temb_dim: int = 512,
        d_model: int = 256,
        n_heads: int = 4,
        cross_attn_depths: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mults = channel_multipliers or [1, 2, 4]
        attn_depths = set(cross_attn_depths or [0, 1])  # 0=bottleneck, 1=deepest skip
        n_levels = len(mults)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.temb_dim = temb_dim

        # Timestep embedding
        self.temb_net = TimestepMLP(base_channels, temb_dim)

        # Input projection
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        channels = [base_channels * m for m in mults]
        in_ch = base_channels
        self.down_blocks = nn.ModuleList()
        for depth, out_ch in enumerate(channels):
            use_ca = depth in attn_depths
            self.down_blocks.append(
                DownBlock(
                    in_ch, out_ch, temb_dim=temb_dim, n_res=n_res_blocks,
                    dropout=dropout, use_cross_attn=use_ca, d_model=d_model, n_heads=n_heads,
                )
            )
            in_ch = out_ch

        # Bottleneck (always uses cross-attention)
        bot_ch = channels[-1]
        self.bottleneck = nn.ModuleList([
            ResBlock(bot_ch, bot_ch, temb_dim=temb_dim, dropout=dropout,
                     use_cross_attn=True, d_model=d_model, n_heads=n_heads),
            ResBlock(bot_ch, bot_ch, temb_dim=temb_dim, dropout=dropout,
                     use_cross_attn=False),
        ])

        # Decoder
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for depth, (skip_ch, out_ch) in enumerate(zip(rev_channels, rev_channels[1:] + [base_channels])):
            use_ca = (n_levels - 1 - depth) in attn_depths
            self.up_blocks.append(
                UpBlock(
                    skip_ch, skip_ch, out_ch, temb_dim=temb_dim, n_res=n_res_blocks,
                    dropout=dropout, use_cross_attn=use_ca, d_model=d_model, n_heads=n_heads,
                )
            )

        # Output head
        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init final conv (common in diffusion models)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ConditionalUNet":
        u = cfg.model.unet
        c = cfg.model.condition
        return cls(
            in_channels=int(cfg.model.in_channels),
            base_channels=int(u.base_channels),
            channel_multipliers=list(u.channel_multipliers),
            n_res_blocks=int(u.n_res_blocks),
            temb_dim=int(u.temb_dim),
            d_model=int(c.d_model),
            n_heads=int(c.n_heads),
            cross_attn_depths=list(u.cross_attn_depths),
            dropout=float(u.dropout),
        )

    def forward(
        self,
        x_t: torch.Tensor,       # [B, C, F, T]
        t: torch.Tensor,         # [B]
        condition: torch.Tensor, # [B, n_tokens, d_model]
    ) -> torch.Tensor:
        # Timestep embedding
        temb = sinusoidal_embedding(t, self.base_channels)  # [B, base_ch]
        temb = self.temb_net(temb)                          # [B, temb_dim]

        # Encoder
        x = self.in_conv(x_t)
        skips: list[torch.Tensor] = []
        for blk in self.down_blocks:
            x, skip = blk(x, temb, condition)
            skips.append(skip)

        # Bottleneck
        for blk in self.bottleneck:
            x = blk(x, temb, condition)

        # Decoder
        for blk, skip in zip(self.up_blocks, reversed(skips)):
            x = blk(x, skip, temb, condition)

        # Output
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, base_channels={self.base_channels}, "
            f"temb_dim={self.temb_dim}"
        )
