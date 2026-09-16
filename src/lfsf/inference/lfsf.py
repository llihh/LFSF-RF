"""Shared full-stack LFSF inference."""

from __future__ import annotations

import torch

from lfsf.models.focus_scorer import FocusScorer, softmax_weights


@torch.no_grad()
def run_lfsf(stack, stack_mask, scorer, vae, fusion, frame_chunk=2):
    if stack.ndim != 5:
        raise ValueError("stack must have shape [B,N,C,H,W]")
    batch, frames, channels, height, width = stack.shape
    if frames < 2:
        raise ValueError("A focus stack must contain at least two frames")
    if stack_mask.shape != (batch, frames):
        raise ValueError("stack_mask must have shape [B,N]")
    flat = stack.reshape(batch * frames, channels, height, width)
    if frame_chunk <= 0:
        frame_chunk = len(flat)
    latent_parts = [
        vae.encode_image(flat[start : start + frame_chunk])
        for start in range(0, len(flat), frame_chunk)
    ]
    latent = torch.cat(latent_parts, dim=0)
    latent_stack = latent.reshape(batch, frames, *latent.shape[1:])
    gradient = FocusScorer.gradient_cue(flat, latent.shape[-2:])
    logits = scorer(latent_stack, gradient)
    weights = softmax_weights(logits, stack_mask)
    result = fusion(stack, weights, latent_stack, vae.decode_latent)
    result.update({"latent_stack": latent_stack, "logits": logits, "weights": weights})
    result["uncertainty"] = 1.0 - result["confidence"]
    return result
