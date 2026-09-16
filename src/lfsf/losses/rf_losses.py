"""Focus conditions used by LFSF-RF training and inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    luminance = rgb_to_luma(x)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    luminance = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
    return torch.abs(F.conv2d(luminance, kernel_x)) + torch.abs(F.conv2d(luminance, kernel_y))


def normalize_per_sample(
    x: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99
) -> torch.Tensor:
    batch = x.shape[0]
    flattened = x.flatten(1)
    low = torch.quantile(flattened, q_low, dim=1).view(batch, 1, 1, 1)
    high = torch.quantile(flattened, q_high, dim=1).view(batch, 1, 1, 1)
    return ((x - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def compute_focus_edit_mask(
    f0: torch.Tensor,
    stack: torch.Tensor,
    stack_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    alpha_uncert: float = 0.70,
    beta_detail: float = 0.85,
    gamma_edge: float = 0.25,
    threshold: float = 0.55,
    soft_width: float = 0.35,
) -> dict[str, torch.Tensor]:
    """Compute the fixed focus/reliability conditioning maps."""
    batch, frames, channels, height, width = stack.shape
    edge_base = normalize_per_sample(sobel_magnitude(f0))
    frame_edges = sobel_magnitude(stack.reshape(batch * frames, channels, height, width))
    frame_edges = frame_edges.reshape(batch, frames, 1, height, width)
    valid = stack_mask[:, :, None, None, None].to(frame_edges.dtype)
    edge_stack = frame_edges.masked_fill(valid <= 0, -1e4).max(dim=1).values
    edge_stack = normalize_per_sample(edge_stack.clamp_min(0.0))
    detail_gap = (edge_stack - edge_base).clamp_min(0.0)
    score = alpha_uncert * uncertainty + beta_detail * detail_gap + gamma_edge * edge_base
    edit_mask = ((score - threshold) / max(soft_width, 1e-6)).clamp(0.0, 1.0)
    return {
        "edit_mask": edit_mask.detach(),
        "detail_gap": detail_gap.detach(),
        "edge_base": edge_base.detach(),
        "edge_stack": edge_stack.detach(),
        "edit_score": score.detach(),
    }
