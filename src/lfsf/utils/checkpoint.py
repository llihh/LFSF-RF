"""Strict loaders for the frozen paper checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parameter_count(model: nn.Module, expected: int) -> int:
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != expected:
        raise RuntimeError(f"Parameter-count mismatch: expected {expected:,}, got {actual:,}")
    return actual


def _load(path: str | Path, device: torch.device | str) -> dict:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    return payload


def _strict_load(path, model, state, role, expected_parameters):
    model.load_state_dict(state, strict=True)
    parameters = validate_parameter_count(model, expected_parameters)
    print(f"Checkpoint: {path}")
    print(f"SHA-256: {checkpoint_sha256(path)}")
    print(f"Checkpoint role: {role}")
    print(f"Loaded state tensors: {len(state)}/{len(model.state_dict())}")
    print("Missing keys: 0")
    print("Unexpected keys: 0")
    print(f"Trainable parameters: {parameters:,}")
    return model


def load_lfsf_checkpoint(path, model, device):
    payload = _load(path, device)
    if "focus_scorer" not in payload:
        raise KeyError("LFSF checkpoint has no 'focus_scorer' state")
    latent_channels = int(payload.get("latent_channels", -1))
    if latent_channels != int(model.latent_channels):
        raise RuntimeError(
            f"latent_channels mismatch: checkpoint={latent_channels}, model={model.latent_channels}"
        )
    return _strict_load(path, model, payload["focus_scorer"], "focus_scorer", 39361)


def load_refiner_checkpoint(
    path, model, device, expected_model_type, weight_source: str = "ema"
):
    if weight_source not in {"ema", "online"}:
        raise ValueError("weight_source must be 'ema' or 'online'")
    payload = _load(path, device)
    actual_type = payload.get("model_type")
    if actual_type != expected_model_type:
        raise RuntimeError(f"model_type mismatch: expected {expected_model_type}, got {actual_type}")
    state_key = "velocity_model" if weight_source == "ema" else "online_model"
    if state_key not in payload:
        raise KeyError(f"Checkpoint has no '{state_key}' state")
    expected = int(payload.get("trainable_params", -1))
    if expected != 838275:
        raise RuntimeError(f"Unexpected checkpoint trainable_params: {expected}")
    return _strict_load(path, model, payload[state_key], state_key, expected)


def load_checkpoint_metadata(path, device="cpu"):
    """Return non-tensor metadata used to construct the frozen architecture."""
    payload = _load(path, device)
    return {
        "args": dict(payload.get("args") or {}),
        "gate_config": dict(payload.get("gate_config") or {}),
        "model_type": payload.get("model_type"),
        "latent_channels": payload.get("latent_channels"),
    }


def verify_checkpoint_integrity(path: str | Path, expected_sha256: str) -> bool:
    return checkpoint_sha256(path) == expected_sha256.lower()
