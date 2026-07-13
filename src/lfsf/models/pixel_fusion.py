"""Soft pixel-domain focus stack fusion.

Core contribution of LFSF: upsamples focus weights from latent resolution
to pixel resolution and fuses original RGB pixels directly, avoiding
VAE decoder reconstruction loss on non-boundary regions.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPixelFusion(nn.Module):
    """Fuse a focus stack in pixel space using learned continuous focus weights.

    The focus weights are predicted at VAE-latent resolution by a FocusScorer,
    then bilinearly upsampled to the original image resolution for weighted
    averaging of RGB pixels. This avoids routing all output pixels through the
    VAE decoder.

    Supports three fusion modes:
      - "pixel": Soft fusion in RGB pixel space (LFSF, default).
      - "latent": Soft fusion in VAE latent space.
      - "hard_pixel": Straight-through hard selection in pixel space.
    """

    def __init__(self, fusion_mode: str = "pixel"):
        super().__init__()
        if fusion_mode not in {"pixel", "latent", "hard_pixel"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.hard = fusion_mode == "hard_pixel"

    @staticmethod
    def _straight_through(weights: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator for hard argmax selection."""
        index = weights.argmax(dim=1, keepdim=True)
        hard = torch.zeros_like(weights).scatter_(1, index, 1.0)
        return hard.detach() - weights.detach() + weights

    def forward(
        self,
        stack: torch.Tensor,
        latent_weights: torch.Tensor,
        latent_stack: torch.Tensor,
        decode_fn,
    ) -> Dict[str, torch.Tensor]:
        """Fuse the focus stack.

        Args:
            stack: [B, N, C, H, W] original RGB focus stack.
            latent_weights: [B, N, 1, H_lat, W_lat] focus weights at latent resolution.
            latent_stack: [B, N, C_lat, H_lat, W_lat] VAE latent representations.
            decode_fn: callable that decodes a latent tensor to RGB.

        Returns:
            Dict with keys:
              - "fused": [B, C, H, W] fused output.
              - "pixel_weights": [B, N, 1, H, W] upsampled fusion weights.
              - "latent_fused": [B, C_lat, H_lat, W_lat] fused latent.
              - "depth_pred": [B, 1, H, W] expected depth index.
              - "confidence": [B, 1, H, W] max-weight confidence map.
        """
        b, n, c, h, w = stack.shape

        if self.fusion_mode == "latent":
            w_latent = self._straight_through(latent_weights.squeeze(2)) if self.hard else latent_weights
            z_fused = (w_latent * latent_stack).sum(dim=1)
            fused = decode_fn(z_fused)
            pixel_weights = F.interpolate(
                w_latent.reshape(b * n, 1, *latent_weights.shape[-2:]),
                size=(h, w), mode="bilinear", align_corners=False,
            ).reshape(b, n, 1, h, w)
        else:
            # Pixel-domain fusion (LFSF)
            pixel_w = F.interpolate(
                latent_weights.reshape(b * n, 1, *latent_weights.shape[-2:]),
                size=(h, w), mode="bilinear", align_corners=False,
            ).reshape(b, n, 1, h, w)
            pixel_w = pixel_w / pixel_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if self.hard:
                pixel_w = self._straight_through(pixel_w.squeeze(2)).unsqueeze(2)
            fused = (pixel_w * stack).sum(dim=1)
            pixel_weights = pixel_w
            z_fused = (latent_weights * latent_stack).sum(dim=1)

        # Confidence: max weight indicates how "certain" the focus assignment is
        confidence = F.interpolate(
            latent_weights.max(dim=1).values,
            size=(h, w), mode="bilinear", align_corners=False,
        )
        # Expected depth index (continuous, 0 to N-1)
        positions = torch.arange(n, device=stack.device, dtype=latent_weights.dtype)
        depth_pred = (latent_weights.squeeze(2) * positions[None, :, None, None]).sum(dim=1, keepdim=True)
        depth_pred = F.interpolate(depth_pred, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "fused": fused,
            "pixel_weights": pixel_weights,
            "latent_fused": z_fused,
            "depth_pred": depth_pred,
            "confidence": confidence,
        }
