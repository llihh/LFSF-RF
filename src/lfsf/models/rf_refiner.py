"""Shortcut-free rectified-flow refiner used by LFSF-RF."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.linspace(0, math.log(10000.0), half, device=t.device, dtype=t.dtype) * -1.0
        )
        arguments = t[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
        return F.pad(embedding, (0, self.dim - embedding.shape[1]))


class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x) + self.time_mlp(time_embedding)[:, :, None, None]
        hidden = F.silu(self.norm1(hidden))
        return F.silu(self.norm2(self.conv2(hidden)))


class StackEncoder(nn.Module):
    """Aggregate a padded variable-length focal stack into a feature map."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = stack.shape
        features = self.net(stack.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, -1, height, width)
        valid = mask[:, :, None, None, None].to(features.dtype)
        mean = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum = features.masked_fill(valid <= 0, -1e4).max(dim=1).values
        return 0.5 * (mean + maximum)


class NoSourceShortcutStackRF(nn.Module):
    """LFSF-RF velocity network without an exact RGB source shortcut.

    ``x_t`` is the only RGB image state. The 1x1 adapter keeps the architecture
    at the 838,275-parameter configuration used by the released checkpoint.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 32,
        time_dim: int = 64,
    ):
        super().__init__()
        if base != 32:
            raise ValueError("The released LFSF-RF architecture requires base=32.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base = int(base)
        self.stack_encoder = StackEncoder(in_channels=in_channels, base=base)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        encoder_channels = in_channels + 6 + base + 1
        self.enc1 = TimeBlock(encoder_channels, base, time_dim)
        self.parameter_match_adapter = nn.Conv2d(base, 27, 1, bias=False)
        self.enc2 = TimeBlock(base, base * 2, time_dim)
        self.enc3 = TimeBlock(base * 2, base * 4, time_dim)
        self.mid = TimeBlock(base * 4, base * 4, time_dim)
        self.dec2 = TimeBlock(base * 6, base * 2, time_dim)
        self.dec1 = TimeBlock(base * 3, base, time_dim)
        self.out = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stack: torch.Tensor,
        stack_mask: torch.Tensor,
        confidence: torch.Tensor,
        uncertainty: torch.Tensor,
        edit_mask: torch.Tensor,
        detail_gap: torch.Tensor,
        edge_base: torch.Tensor,
        edge_stack: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embed(t)
        stack_features = self.stack_encoder(stack, stack_mask)
        time_map = t[:, None, None, None].expand(-1, 1, *x_t.shape[-2:])
        inputs = torch.cat(
            (
                x_t,
                confidence,
                uncertainty,
                edit_mask,
                detail_gap,
                edge_base,
                edge_stack,
                stack_features,
                time_map,
            ),
            dim=1,
        )
        enc1 = self.enc1(inputs, time_embedding)
        adapter = self.parameter_match_adapter(enc1)
        adapter = torch.cat((adapter, torch.zeros_like(enc1[:, :5])), dim=1)
        enc1 = enc1 + adapter
        enc2 = self.enc2(F.avg_pool2d(enc1, 2), time_embedding)
        enc3 = self.enc3(F.avg_pool2d(enc2, 2), time_embedding)
        middle = self.mid(enc3, time_embedding)
        dec2 = F.interpolate(middle, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1), time_embedding)
        dec1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1), time_embedding)
        return self.out(dec1)
