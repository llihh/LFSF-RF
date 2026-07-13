"""Focus scoring head for per-frame focus probability estimation.

Core component of LFSF: takes VAE latent features + local gradient cue,
predicts per-frame focus scores, and produces softmax-normalized weights.
Preserves frame identity — each frame is scored independently through a shared head.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusScorer(nn.Module):
    """Per-frame focus probability head.

    For each frame in a focus stack, computes a scalar focus score from:
      - VAE latent representation (semantic features)
      - Local high-frequency cue (sharpness indicator)

    Scores are softmax-normalized across frames to produce continuous focus weights.

    Args:
        latent_channels: Number of channels in the VAE latent space.
        hidden_channels: Hidden dimension in the scoring MLP.
    """

    def __init__(self, latent_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.latent_channels = int(latent_channels)
        # latent_channels for VAE features + 1 for the gradient cue
        self.head = nn.Sequential(
            nn.Conv2d(latent_channels + 1, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, z_stack: torch.Tensor, gradient_cue: torch.Tensor) -> torch.Tensor:
        """Compute per-frame focus logits.

        Args:
            z_stack: [B, N, C, H, W] VAE latent stack.
            gradient_cue: [B*N, 1, H, W] local gradient magnitude per frame.

        Returns:
            logits: [B, N, H, W] unnormalized focus scores.
        """
        b, n, cz, hz, wz = z_stack.shape
        z_flat = z_stack.reshape(b * n, cz, hz, wz)
        feat = torch.cat([z_flat, gradient_cue], dim=1)
        logits = self.head(feat).reshape(b, n, hz, wz)
        return logits

    @staticmethod
    def gradient_cue(x: torch.Tensor, target_size) -> torch.Tensor:
        """Local high-frequency magnitude as a sharpness cue.

        Computes |I - blur(I)| using a 3x3 average pool as the low-pass filter,
        then resizes to the target spatial size.

        Args:
            x: [B*N, C, H, W] RGB or grayscale image stack.
            target_size: (H, W) tuple for output resolution.
        """
        gray = x.mean(dim=1, keepdim=True)
        cue = (gray - F.avg_pool2d(gray, 3, stride=1, padding=1)).abs()
        return F.interpolate(cue, size=target_size, mode="bilinear", align_corners=False)


def softmax_weights(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert focus logits to softmax-normalized weights.

    Args:
        logits: [B, N, H, W] focus scores.
        mask: [B, N] optional boolean mask for valid frames.

    Returns:
        weights: [B, N, 1, H, W] softmax focus probabilities.
    """
    if mask is not None:
        mask = mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Each stack must contain at least one valid frame")
        logits = logits.masked_fill(~mask[:, :, None, None], torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=1)
    if mask is not None:
        weights = weights * mask[:, :, None, None]
    return weights.unsqueeze(2)  # [B, N, 1, H, W]
