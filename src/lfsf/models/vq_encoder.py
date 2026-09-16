"""Frozen VQ-VAE wrapper for encoding/decoding focus stacks.

Reuses the RFfusion pretrained VQ-VAE (NeurIPS 2025) as a frozen feature extractor.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml

from lfsf.utils.config import instantiate_from_config


def _first_tensor(value):
    """Extract the first tensor from VAE encode output."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if hasattr(value, "mode"):
        return value.mode()
    if hasattr(value, "sample"):
        return value.sample()
    raise TypeError(f"Cannot extract tensor from VAE encode output of type {type(value)}")


class RFVAEWrapper(nn.Module):
    """Normalize RFfusion VAE/VQModel encode/decode behavior.

    Args:
        vae: The underlying VQ-VAE module.
        latent_scale: Scale factor applied to latent representations (default 4.0 for RF-VAE).
        input_range: Expected input normalization, either "-1_1" or "0_1".
    """

    def __init__(self, vae: nn.Module, latent_scale: float = 4.0, input_range: str = "-1_1"):
        super().__init__()
        self.vae = vae
        self.latent_scale = float(latent_scale)
        self.input_range = input_range
        self.latent_channels = int(getattr(vae, "embed_dim", 0) or 0)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        dtype = next(self.vae.parameters()).dtype
        z = _first_tensor(self.vae.encode(x.to(dtype)))
        return (z * self.latent_scale).to(x.dtype)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        dtype = next(self.vae.parameters()).dtype
        out = self.vae.decode((z / self.latent_scale).to(dtype))
        return out.to(z.dtype)

    def forward_recon(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_latent(self.encode_image(x))


def load_rffusion_vae(
    config_path: str,
    ckpt_path: Optional[str],
    device: torch.device,
    force_fp32: bool = False,
) -> RFVAEWrapper:
    """Load a frozen RFfusion VAE from config and checkpoint.

    Args:
        config_path: Path to the VQ encoder YAML config.
        ckpt_path: Path to autoencoder.ckpt.
        device: Target device.
        force_fp32: If True, disable fp16 even when config says use_fp16.

    Returns:
        RFVAEWrapper with VAE loaded and set to eval mode.
    """
    config = _load_yaml(config_path)
    ae_cfg, resolved_ckpt = _resolve_autoencoder_config(config, ckpt_path)
    use_fp16 = bool(ae_cfg.pop("use_fp16", False))
    # Replace LPIPS loss with no-op to avoid loading VGG/discriminator
    ae_cfg["params"]["lossconfig"] = {"target": "torch.nn.Identity", "params": {}}
    vae = instantiate_from_config(ae_cfg).to(device)
    sd = torch.load(resolved_ckpt, map_location=device)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    vae_keys = {k for k in vae.state_dict()}
    sd = {k: v for k, v in sd.items() if k in vae_keys}
    missing = sorted(vae_keys - set(sd))
    if missing:
        raise RuntimeError(f"VQ checkpoint is missing {len(missing)} model tensors: {missing[:8]}")
    vae.load_state_dict(sd, strict=True)
    vae.eval()
    if use_fp16 and device.type == "cuda" and not force_fp32:
        vae = vae.half().to(device)
    else:
        vae = vae.float().to(device)
    for p in vae.parameters():
        assert p.device.type == device.type, f"Parameter on {p.device}, expected {device}"
    wrapper = RFVAEWrapper(vae, latent_scale=4.0, input_range="-1_1").to(device)
    print(f"[load_rffusion_vae] config={config_path} ckpt={resolved_ckpt} fp16={next(vae.parameters()).dtype == torch.float16}")
    return wrapper


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_autoencoder_config(config: Dict[str, Any], vae_ckpt: Optional[str]):
    if "autoencoder" in config:
        ae_cfg = dict(config["autoencoder"])
    elif "model" in config and "target" in config["model"]:
        ae_cfg = dict(config["model"])
    else:
        raise ValueError("Could not find autoencoder config in config file.")
    if vae_ckpt:
        ae_cfg["ckpt_path"] = vae_ckpt
    ckpt_path = ae_cfg.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError("VAE checkpoint path is required.")
    return ae_cfg, ckpt_path
