"""Conditional Flow Matching loss for LFSF-RF rectified flow refinement.

Implements the straight-line conditional flow matching objective:
  x_t = (1-t) * x_0 + t * x_1
  v_target = x_1 - x_0
  L = MSE(v_theta(x_t, t, cond), v_target)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_flow_batch(
    f0: torch.Tensor,
    target: torch.Tensor,
    noise_scale: float = 0.2,
    t_min: float = 1e-4,
    t_max: float = 0.999,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a conditional flow matching batch.

    Args:
        f0:        [B, C, H, W] starting point (LFSF fused output), in [-1, 1].
        target:    [B, C, H, W] endpoint (ground-truth AiF), in [-1, 1].
        noise_scale: Std of Gaussian noise added to f0.
        t_min, t_max: Time range bounds.
        value_range: (vmin, vmax) of the data.

    Returns:
        x_t, t, v_target, x_0
    """
    b = f0.shape[0]
    device = f0.device
    vmin, vmax = value_range
    t = torch.rand(b, device=device).clamp(t_min, t_max)
    noise = noise_scale * torch.randn_like(f0)
    x_0 = torch.clamp(f0 + noise, vmin, vmax)
    t_reshape = t.reshape(b, *([1] * (f0.ndim - 1)))
    x_t = (1.0 - t_reshape) * x_0 + t_reshape * target
    v_target = target - x_0
    return x_t, t, v_target, x_0


class FlowMatchingLoss(nn.Module):
    """Flow matching training loss with auxiliary quality terms.

    Loss = lambda_rf    * MSE(v_pred, v_target)
         + lambda_recon * L1(endpoint, target)
         + lambda_identity * |(1-U) * (endpoint - f0)|
         + lambda_grad  * L1(grad endpoint, grad target)
    """

    def __init__(
        self,
        lambda_rf: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_identity: float = 0.5,
        lambda_grad: float = 0.5,
    ):
        super().__init__()
        self.lambda_rf = float(lambda_rf)
        self.lambda_recon = float(lambda_recon)
        self.lambda_identity = float(lambda_identity)
        self.lambda_grad = float(lambda_grad)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            grads = []
            for i in range(3):
                ch = x[:, i:i+1]
                ch = F.pad(ch, (1, 1, 1, 1), mode="replicate")
                g = torch.abs(F.conv2d(ch, self.kx)) + torch.abs(F.conv2d(ch, self.ky))
                grads.append(g)
            return sum(grads) / 3.0
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return torch.abs(F.conv2d(x, self.kx)) + torch.abs(F.conv2d(x, self.ky))

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        endpoint: torch.Tensor,
        target: torch.Tensor,
        f0: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.kx.device != endpoint.device:
            self.kx = self.kx.to(endpoint.device)
            self.ky = self.ky.to(endpoint.device)
        loss_rf = F.mse_loss(v_pred, v_target)
        loss_recon = F.l1_loss(endpoint, target)
        confidence = (1.0 - uncertainty).detach()
        loss_identity = (confidence * torch.abs(endpoint - f0)).mean()
        loss_grad = F.l1_loss(self._sobel(endpoint), self._sobel(target))
        total = (
            self.lambda_rf * loss_rf
            + self.lambda_recon * loss_recon
            + self.lambda_identity * loss_identity
            + self.lambda_grad * loss_grad
        )
        return {
            "loss_total": total,
            "loss_rf": loss_rf,
            "loss_recon": loss_recon,
            "loss_identity": loss_identity,
            "loss_grad": loss_grad,
        }


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    y = rgb_to_luma(x)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return torch.abs(F.conv2d(y, kx)) + torch.abs(F.conv2d(y, ky))


def normalize_per_sample(x: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99) -> torch.Tensor:
    b = x.shape[0]
    flat = x.flatten(1)
    lo = torch.quantile(flat, q_low, dim=1).view(b, 1, 1, 1)
    hi = torch.quantile(flat, q_high, dim=1).view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


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
    """Build the editable-region mask used by focus-gated flow matching.

    M = clip((alpha*U + beta*D + gamma*E0 - threshold) / soft_width, 0, 1)
    where U=uncertainty, D=detail_gap, E0=edge_base.
    """
    b, n, c, h, w = stack.shape
    edge_base = normalize_per_sample(sobel_magnitude(f0))
    flat_stack = stack.reshape(b * n, c, h, w)
    edge_stack_frames = sobel_magnitude(flat_stack).reshape(b, n, 1, h, w)
    valid = stack_mask[:, :, None, None, None].to(dtype=edge_stack_frames.dtype)
    edge_stack = edge_stack_frames.masked_fill(valid <= 0, -1e4).max(dim=1).values
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


def make_masked_flow_batch(
    f0: torch.Tensor, target: torch.Tensor, edit_mask: torch.Tensor,
    noise_scale: float = 0.05, t_min: float = 1e-4, t_max: float = 0.999,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a focus-gated conditional flow matching batch.

    x_1 = f0 + M * (target - f0)
    x_0 = f0 + M * eps
    """
    b = f0.shape[0]
    device = f0.device
    vmin, vmax = value_range
    t = torch.rand(b, device=device).clamp(t_min, t_max)
    x_1 = f0 + edit_mask * (target - f0)
    noise = noise_scale * torch.randn_like(f0)
    x_0 = torch.clamp(f0 + edit_mask * noise, vmin, vmax)
    t_reshape = t.reshape(b, *([1] * (f0.ndim - 1)))
    x_t = (1.0 - t_reshape) * x_0 + t_reshape * x_1
    v_target = x_1 - x_0
    return x_t, t, v_target, x_0, x_1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(x)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


class FocusGatedFlowLoss(nn.Module):
    """Masked conditional flow matching loss for focus-gated refinement."""

    def __init__(
        self,
        lambda_rf: float = 1.0,
        lambda_recon: float = 2.0,
        lambda_identity: float = 6.0,
        lambda_grad: float = 0.25,
        lambda_residual: float = 0.10,
    ):
        super().__init__()
        self.lambda_rf = float(lambda_rf)
        self.lambda_recon = float(lambda_recon)
        self.lambda_identity = float(lambda_identity)
        self.lambda_grad = float(lambda_grad)
        self.lambda_residual = float(lambda_residual)

    def forward(
        self,
        v_pred: torch.Tensor, v_target: torch.Tensor,
        endpoint: torch.Tensor, target_endpoint: torch.Tensor,
        f0: torch.Tensor, edit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reliable = 1.0 - edit_mask
        loss_rf = _masked_mean((v_pred - v_target).pow(2), edit_mask)
        loss_recon = _masked_mean(torch.abs(endpoint - target_endpoint), edit_mask)
        loss_identity = _masked_mean(torch.abs(endpoint - f0), reliable)
        loss_grad = _masked_mean(
            torch.abs(sobel_magnitude(endpoint) - sobel_magnitude(target_endpoint)), edit_mask)
        loss_residual = _masked_mean(torch.abs(v_pred), reliable)
        total = (
            self.lambda_rf * loss_rf + self.lambda_recon * loss_recon
            + self.lambda_identity * loss_identity + self.lambda_grad * loss_grad
            + self.lambda_residual * loss_residual
        )
        return {
            "loss_total": total, "loss_rf": loss_rf, "loss_recon": loss_recon,
            "loss_identity": loss_identity, "loss_grad": loss_grad, "loss_residual": loss_residual,
        }
