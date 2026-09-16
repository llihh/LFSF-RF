"""Soft pixel-domain fusion used by LFSF."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPixelFusion(nn.Module):
    """Upsample learned focus weights and fuse the observed RGB pixels."""

    def forward(
        self,
        stack: torch.Tensor,
        latent_weights: torch.Tensor,
        latent_stack: torch.Tensor,
        decode_fn=None,
    ) -> Dict[str, torch.Tensor]:
        batch, frames, _, height, width = stack.shape
        pixel_weights = F.interpolate(
            latent_weights.reshape(batch * frames, 1, *latent_weights.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, 1, height, width)
        pixel_weights = pixel_weights / pixel_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (pixel_weights * stack).sum(dim=1)
        latent_fused = (latent_weights * latent_stack).sum(dim=1)
        confidence = F.interpolate(
            latent_weights.max(dim=1).values,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        positions = torch.arange(frames, device=stack.device, dtype=latent_weights.dtype)
        depth = (latent_weights.squeeze(2) * positions[None, :, None, None]).sum(
            dim=1, keepdim=True
        )
        depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        return {
            "fused": fused,
            "pixel_weights": pixel_weights,
            "latent_fused": latent_fused,
            "depth_pred": depth,
            "confidence": confidence,
        }
