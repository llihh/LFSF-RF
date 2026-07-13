"""Focus-conditioned Rectified Flow velocity network.

Architecture: a compact UNet-style CNN that takes the current flow state x_t,
the LFSF base output f0, per-pixel confidence/uncertainty, a stack-context
feature map, and a time embedding — and predicts the velocity v = dx/dt.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000.0), half, device=t.device, dtype=t.dtype)
            * -1.0
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class TimeBlock(nn.Module):
    """Conv-Norm-Act block modulated by a time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        groups = max(1, min(8, out_ch // 4))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(temb)[:, :, None, None]
        h = F.silu(self.norm1(h))
        h = F.silu(self.norm2(self.conv2(h)))
        return h


# ---------------------------------------------------------------------------
# Stack context encoder
# ---------------------------------------------------------------------------

class StackEncoder(nn.Module):
    """Encode a variable-length focus stack into a fixed-size feature map."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Aggregate a focus stack across the frame dimension.

        Args:
            stack: [B, N, C, H, W] focus stack frames.
            mask:  [B, N] binary validity mask (1 = valid frame, 0 = pad).

        Returns:
            [B, base, H, W] fused context feature.
        """
        bsz, n, ch, h, w = stack.shape
        feat = self.net(stack.reshape(bsz * n, ch, h, w)).reshape(bsz, n, -1, h, w)
        weights = mask[:, :, None, None, None]       # [B, N, 1, 1, 1]
        denom = weights.sum(dim=1).clamp_min(1.0)
        mean_feat = (feat * weights).sum(dim=1) / denom
        max_feat = feat.masked_fill(weights <= 0, -1e4).max(dim=1).values
        return 0.5 * (mean_feat + max_feat)


# ---------------------------------------------------------------------------
# Velocity networks
# ---------------------------------------------------------------------------

class FocusConditionedStackRF(nn.Module):
    """Stack-aware velocity predictor for teacher-guided MFIF flow matching (P0 refiner).

    Args:
        in_channels:  Number of image channels (1 for gray, 3 for RGB).
        out_channels: Number of output channels (= in_channels typically).
        base:         Base channel count for the UNet.
        time_dim:     Dimensionality of the sinusoidal time embedding.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base: int = 32, time_dim: int = 64):
        super().__init__()
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

        # x_t(C) + f0(C) + confidence(1) + uncertainty(1) + stack_feat(base) + t_map(1)
        encode_in = in_channels + in_channels + 1 + 1 + base + 1

        self.enc1 = TimeBlock(encode_in, base, time_dim)
        self.enc2 = TimeBlock(base, base * 2, time_dim)
        self.enc3 = TimeBlock(base * 2, base * 4, time_dim)
        self.mid  = TimeBlock(base * 4, base * 4, time_dim)
        self.dec2 = TimeBlock(base * 6, base * 2, time_dim)
        self.dec1 = TimeBlock(base * 3, base,     time_dim)
        self.out = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x_t:          torch.Tensor,
        t:            torch.Tensor,
        stack:        torch.Tensor,
        stack_mask:   torch.Tensor,
        f0:           torch.Tensor,
        confidence:   torch.Tensor,
        uncertainty:  torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity v = dx/dt."""
        bsz = x_t.shape[0]
        temb = self.time_embed(t)
        stack_feat = self.stack_encoder(stack, stack_mask)
        t_map = t[:, None, None, None].expand(-1, 1, *x_t.shape[-2:])
        x = torch.cat([x_t, f0, confidence, uncertainty, stack_feat, t_map], dim=1)

        e1 = self.enc1(x, temb)
        e2 = self.enc2(F.avg_pool2d(e1, 2), temb)
        e3 = self.enc3(F.avg_pool2d(e2, 2), temb)
        mid = self.mid(e3, temb)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), temb)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), temb)
        return self.out(d1)


class FocusGatedStackRF(nn.Module):
    """Focus-gated velocity predictor for masked rectified-flow refinement (C2).

    Receives explicit edit-region priors (edit_mask, detail_gap, edge_base, edge_stack).
    The returned velocity is multiplied by edit_mask by default.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base: int = 32, time_dim: int = 64):
        super().__init__()
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

        # x_t(C) + f0(C) + confidence/uncertainty/edit/detail/edge0/edgeS(6) + stack_feat(base) + t_map(1)
        encode_in = in_channels + in_channels + 6 + base + 1

        self.enc1 = TimeBlock(encode_in, base, time_dim)
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
        x_t: torch.Tensor, t: torch.Tensor,
        stack: torch.Tensor, stack_mask: torch.Tensor,
        f0: torch.Tensor, confidence: torch.Tensor, uncertainty: torch.Tensor,
        edit_mask: torch.Tensor, detail_gap: torch.Tensor,
        edge_base: torch.Tensor, edge_stack: torch.Tensor,
        apply_mask: bool = True,
    ) -> torch.Tensor:
        temb = self.time_embed(t)
        stack_feat = self.stack_encoder(stack, stack_mask)
        t_map = t[:, None, None, None].expand(-1, 1, *x_t.shape[-2:])
        x = torch.cat(
            [x_t, f0, confidence, uncertainty, edit_mask, detail_gap,
             edge_base, edge_stack, stack_feat, t_map], dim=1)
        e1 = self.enc1(x, temb)
        e2 = self.enc2(F.avg_pool2d(e1, 2), temb)
        e3 = self.enc3(F.avg_pool2d(e2, 2), temb)
        mid = self.mid(e3, temb)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), temb)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), temb)
        velocity = self.out(d1)
        if apply_mask:
            velocity = velocity * edit_mask
        return velocity


class DirectResidualStackCNN(nn.Module):
    """Parameter-matched direct residual baseline.

    Same stack encoder, UNet widths, conditioning maps, and trainable parameter
    count as the RF model. No time input and no flow state — directly predicts
    the endpoint residual from the frozen LFSF output.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base: int = 32, time_dim: int = 64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base = int(base)
        self.time_dim = int(time_dim)

        self.stack_encoder = StackEncoder(in_channels=in_channels, base=base)
        self.condition_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        encode_in = in_channels + in_channels + 6 + base + 1
        self.enc1 = TimeBlock(encode_in, base, time_dim)
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
        f0: torch.Tensor, stack: torch.Tensor, stack_mask: torch.Tensor,
        confidence: torch.Tensor, uncertainty: torch.Tensor,
        edit_mask: torch.Tensor, detail_gap: torch.Tensor,
        edge_base: torch.Tensor, edge_stack: torch.Tensor,
    ) -> torch.Tensor:
        bsz = f0.shape[0]
        null_token = f0.new_zeros((bsz, self.time_dim))
        cond = self.condition_embed(null_token)
        stack_feat = self.stack_encoder(stack, stack_mask)
        null_map = f0.new_zeros((bsz, 1, *f0.shape[-2:]))
        x = torch.cat(
            [f0, f0, confidence, uncertainty, edit_mask, detail_gap,
             edge_base, edge_stack, stack_feat, null_map], dim=1)
        e1 = self.enc1(x, cond)
        e2 = self.enc2(F.avg_pool2d(e1, 2), cond)
        e3 = self.enc3(F.avg_pool2d(e2, 2), cond)
        mid = self.mid(e3, cond)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), cond)
        return self.out(d1)


class NoSourceShortcutStackRF(nn.Module):
    """C3 velocity network without exact source-image conditioning (LFSF-RF).

    x_t is the only RGB image state. Focus confidence and edge/detail maps
    remain as scalar spatial conditions, but the complete RGB f0 image is
    deliberately absent. A parameter-matching adapter preserves capacity
    equality with DirectResidualStackCNN (838,275 parameters).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base: int = 32, time_dim: int = 64):
        super().__init__()
        if base != 32:
            raise ValueError("C3 parameter matching currently requires base=32.")
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

        # x_t(C) + confidence/uncertainty/edit/detail/edge0/edgeS(6) + stack_feat(base) + t_map(1)
        encode_in = in_channels + 6 + base + 1
        self.enc1 = TimeBlock(encode_in, base, time_dim)
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
        x_t: torch.Tensor, t: torch.Tensor,
        stack: torch.Tensor, stack_mask: torch.Tensor,
        confidence: torch.Tensor, uncertainty: torch.Tensor,
        edit_mask: torch.Tensor, detail_gap: torch.Tensor,
        edge_base: torch.Tensor, edge_stack: torch.Tensor,
        zero_conditions: bool = False,
    ) -> torch.Tensor:
        temb = self.time_embed(t)
        if zero_conditions:
            stack_feat = x_t.new_zeros((x_t.shape[0], self.base, *x_t.shape[-2:]))
            confidence = torch.zeros_like(confidence)
            uncertainty = torch.zeros_like(uncertainty)
            edit_mask = torch.zeros_like(edit_mask)
            detail_gap = torch.zeros_like(detail_gap)
            edge_base = torch.zeros_like(edge_base)
            edge_stack = torch.zeros_like(edge_stack)
        else:
            stack_feat = self.stack_encoder(stack, stack_mask)
        t_map = t[:, None, None, None].expand(-1, 1, *x_t.shape[-2:])
        x = torch.cat(
            [x_t, confidence, uncertainty, edit_mask, detail_gap,
             edge_base, edge_stack, stack_feat, t_map], dim=1)
        e1 = self.enc1(x, temb)
        adapter = self.parameter_match_adapter(e1)
        adapter = torch.cat([adapter, torch.zeros_like(e1[:, :5])], dim=1)
        e1 = e1 + adapter
        e2 = self.enc2(F.avg_pool2d(e1, 2), temb)
        e3 = self.enc3(F.avg_pool2d(e2, 2), temb)
        mid = self.mid(e3, temb)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), temb)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), temb)
        return self.out(d1)
