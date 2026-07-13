"""Multi-focus image stack training losses for LFSF.

Provides the full LFSF loss suite: depth regression, gradient max-consistency,
intensity weighted-average consistency, SSIM, and optional AiF supervision.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = to_gray(x)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return torch.abs(F.conv2d(x, self.kx)) + torch.abs(F.conv2d(x, self.ky))


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    x = to_gray(x)
    y = to_gray(y)
    pad = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim_map.clamp(0.0, 1.0).mean()


class StackMFIFLoss(nn.Module):
    """Combined loss for multi-focus image stack fusion (LFSF).

    Loss components:
      - depth: L1 or MSE between predicted and ground-truth depth.
      - grad: L1 between fused gradient and per-pixel max gradient across frames.
      - int (intensity): L1 between fused output and weighted-average target.
      - ssim: Structural similarity against weighted-average target.
      - aif_l1 / aif_ssim / aif_grad: Direct AiF supervision.
      - focus: Cross-entropy on focus index classification.
      - preserve: Preserve confident regions from base fusion.
    """

    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_grad: float = 1.0,
        lambda_int: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_rec: float = 0.1,
        lambda_aif_l1: float = 0.0,
        lambda_aif_ssim: float = 0.0,
        lambda_aif_grad: float = 0.0,
        lambda_focus: float = 0.0,
        lambda_preserve: float = 0.0,
        depth_loss: str = "l1",
    ):
        super().__init__()
        self.lambda_depth = float(lambda_depth)
        self.lambda_grad = float(lambda_grad)
        self.lambda_int = float(lambda_int)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_rec = float(lambda_rec)
        self.lambda_aif_l1 = float(lambda_aif_l1)
        self.lambda_aif_ssim = float(lambda_aif_ssim)
        self.lambda_aif_grad = float(lambda_aif_grad)
        self.lambda_focus = float(lambda_focus)
        self.lambda_preserve = float(lambda_preserve)
        self.depth_loss = depth_loss
        self.sobel = SobelGrad()

    def _depth_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.depth_loss == "mse":
            return F.mse_loss(pred, gt)
        return F.l1_loss(pred, gt)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        if float(x.detach().amin()) < -0.05:
            return (x + 1.0) / 2.0
        return x

    def _target_from_weights(self, stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = stack.shape
        ww = weights.reshape(b * n, 1, weights.shape[-2], weights.shape[-1])
        ww = F.interpolate(ww, size=(h, w), mode="bilinear", align_corners=False).reshape(b, n, 1, h, w)
        ww = ww / ww.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (ww * stack).sum(dim=1)

    def forward(
        self,
        stack: torch.Tensor,
        fused: torch.Tensor,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        weights: torch.Tensor,
        aif_target: Optional[torch.Tensor] = None,
        recon: Optional[torch.Tensor] = None,
        recon_target: Optional[torch.Tensor] = None,
        focus_logits: Optional[torch.Tensor] = None,
        stack_mask: Optional[torch.Tensor] = None,
        base_fused: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        stack_01 = self._normalize(stack)
        fused_01 = self._normalize(fused).clamp(0.0, 1.0)
        target = self._normalize(self._target_from_weights(stack, weights)).clamp(0.0, 1.0)

        b, n, c, h, w = stack_01.shape
        grad_stack = self.sobel(stack_01.reshape(b * n, c, h, w)).reshape(b, n, 1, h, w)
        grad_max = grad_stack.max(dim=1).values
        grad_fused = self.sobel(fused_01)

        loss_depth = self._depth_loss(depth_pred, depth_gt)
        loss_grad = F.l1_loss(grad_fused, grad_max)
        loss_int = F.l1_loss(fused_01, target)
        loss_ssim = ssim_loss(fused_01, target)

        loss_rec = fused_01.new_tensor(0.0)
        if recon is not None and recon_target is not None:
            loss_rec = F.l1_loss(self._normalize(recon), self._normalize(recon_target))

        loss_aif_l1 = fused_01.new_tensor(0.0)
        loss_aif_ssim = fused_01.new_tensor(0.0)
        loss_aif_grad = fused_01.new_tensor(0.0)
        metric_aif_psnr = fused_01.new_tensor(0.0)
        if aif_target is not None:
            aif_01 = self._normalize(aif_target).clamp(0.0, 1.0)
            loss_aif_l1 = F.l1_loss(fused_01, aif_01)
            loss_aif_ssim = ssim_loss(fused_01, aif_01)
            loss_aif_grad = F.l1_loss(self.sobel(fused_01), self.sobel(aif_01))
            mse = F.mse_loss(fused_01, aif_01).clamp_min(1e-12)
            metric_aif_psnr = -10.0 * torch.log10(mse)

        loss_focus = fused_01.new_tensor(0.0)
        metric_focus_acc = fused_01.new_tensor(0.0)
        if focus_logits is not None:
            target_depth = F.interpolate(depth_gt, size=focus_logits.shape[-2:], mode="nearest")[:, 0]
            if stack_mask is None:
                valid_n = torch.full((focus_logits.shape[0],), focus_logits.shape[1],
                                     device=focus_logits.device, dtype=torch.long)
            else:
                valid_n = stack_mask.sum(dim=1).to(device=focus_logits.device, dtype=torch.long)
            target_index = torch.round(target_depth * (valid_n[:, None, None] - 1)).long()
            loss_focus = F.cross_entropy(focus_logits, target_index)
            metric_focus_acc = (focus_logits.argmax(dim=1) == target_index).float().mean()

        loss_preserve = fused_01.new_tensor(0.0)
        if base_fused is not None and uncertainty is not None:
            base_01 = self._normalize(base_fused).clamp(0.0, 1.0)
            loss_preserve = F.l1_loss((1.0 - uncertainty) * fused_01, (1.0 - uncertainty) * base_01)

        total = (
            self.lambda_depth * loss_depth
            + self.lambda_grad * loss_grad
            + self.lambda_int * loss_int
            + self.lambda_ssim * loss_ssim
            + self.lambda_rec * loss_rec
            + self.lambda_aif_l1 * loss_aif_l1
            + self.lambda_aif_ssim * loss_aif_ssim
            + self.lambda_aif_grad * loss_aif_grad
            + self.lambda_focus * loss_focus
            + self.lambda_preserve * loss_preserve
        )
        return {
            "loss_total": total,
            "loss_depth": loss_depth,
            "loss_grad": loss_grad,
            "loss_int": loss_int,
            "loss_ssim": loss_ssim,
            "loss_rec": loss_rec,
            "loss_aif_l1": loss_aif_l1,
            "loss_aif_ssim": loss_aif_ssim,
            "loss_aif_grad": loss_aif_grad,
            "loss_focus": loss_focus,
            "loss_preserve": loss_preserve,
            "metric_aif_psnr": metric_aif_psnr,
            "metric_focus_acc": metric_focus_acc,
        }
