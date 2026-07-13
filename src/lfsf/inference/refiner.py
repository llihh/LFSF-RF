"""Frozen LFSF-RF and Direct endpoint inference."""

from __future__ import annotations

import torch

from lfsf.losses.rf_losses import compute_focus_edit_mask


def _conditions(lfsf_outputs, stack, stack_mask, gate_cfg):
    f0 = lfsf_outputs["fused"]
    confidence = lfsf_outputs["confidence"]
    uncertainty = lfsf_outputs.get("uncertainty", 1.0 - confidence)
    gate = compute_focus_edit_mask(f0, stack, stack_mask, uncertainty, **gate_cfg)
    return f0, confidence, uncertainty, gate


@torch.no_grad()
def run_lfsf_rf(lfsf_outputs, stack, stack_mask, rf_model, gate_cfg):
    f0, confidence, uncertainty, gate = _conditions(lfsf_outputs, stack, stack_mask, gate_cfg)
    t = torch.zeros(f0.shape[0], device=f0.device, dtype=f0.dtype)
    velocity = rf_model(
        x_t=f0, t=t, stack=stack, stack_mask=stack_mask,
        confidence=confidence, uncertainty=uncertainty,
        edit_mask=gate["edit_mask"], detail_gap=gate["detail_gap"],
        edge_base=gate["edge_base"], edge_stack=gate["edge_stack"],
    )
    return torch.clamp(f0 + velocity, -1.0, 1.0)


@torch.no_grad()
def run_direct(lfsf_outputs, stack, stack_mask, direct_model, gate_cfg):
    f0, confidence, uncertainty, gate = _conditions(lfsf_outputs, stack, stack_mask, gate_cfg)
    residual = direct_model(
        f0=f0, stack=stack, stack_mask=stack_mask,
        confidence=confidence, uncertainty=uncertainty,
        edit_mask=gate["edit_mask"], detail_gap=gate["detail_gap"],
        edge_base=gate["edge_base"], edge_stack=gate["edge_stack"],
    )
    return torch.clamp(f0 + residual, -1.0, 1.0)
