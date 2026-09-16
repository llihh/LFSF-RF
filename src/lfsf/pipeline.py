"""High-level frozen LFSF/LFSF-RF inference pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from lfsf.inference import run_lfsf, run_lfsf_rf
from lfsf.models import FocusScorer, NoSourceShortcutStackRF, SoftPixelFusion, load_rffusion_vae
from lfsf.utils.checkpoint import (
    checkpoint_sha256,
    load_checkpoint_metadata,
    load_lfsf_checkpoint,
    load_refiner_checkpoint,
)


class FrozenLFSFPipeline(nn.Module):
    """Load the released components and run LFSF or LFSF-RF.

    Inputs are RGB tensors normalized to ``[-1, 1]`` with shape ``[B,N,3,H,W]``.
    The mask has shape ``[B,N]`` and marks valid focal planes.
    """

    def __init__(
        self,
        *,
        lfsf_checkpoint: str | Path,
        vae_config: str | Path,
        vae_checkpoint: str | Path,
        device: str | torch.device,
        rf_checkpoint: str | Path | None = None,
        weight_source: str = "ema",
        force_fp32_vae: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.lfsf_checkpoint = Path(lfsf_checkpoint)
        self.rf_checkpoint = Path(rf_checkpoint) if rf_checkpoint else None

        lfsf_meta = load_checkpoint_metadata(self.lfsf_checkpoint)
        self.vae = load_rffusion_vae(
            str(vae_config), str(vae_checkpoint), self.device, force_fp32=force_fp32_vae
        ).eval()
        self.scorer = FocusScorer(int(lfsf_meta["latent_channels"]), 64).to(self.device).eval()
        load_lfsf_checkpoint(self.lfsf_checkpoint, self.scorer, self.device)
        self.fusion = SoftPixelFusion().to(self.device).eval()

        self.rf = None
        self.gate_config: dict = {}
        if self.rf_checkpoint is not None:
            rf_meta = load_checkpoint_metadata(self.rf_checkpoint)
            if rf_meta["model_type"] != "NoSourceShortcutStackRF":
                raise RuntimeError(
                    "Only shortcut-free NoSourceShortcutStackRF checkpoints are supported"
                )
            model_args = rf_meta["args"]
            self.rf = NoSourceShortcutStackRF(
                in_channels=int(model_args["image_channels"]),
                out_channels=int(model_args["image_channels"]),
                base=int(model_args["base_channels"]),
            ).to(self.device).eval()
            load_refiner_checkpoint(
                self.rf_checkpoint,
                self.rf,
                self.device,
                "NoSourceShortcutStackRF",
                weight_source,
            )
            self.gate_config = rf_meta["gate_config"]
            if not self.gate_config:
                raise RuntimeError("RF checkpoint does not contain gate_config")

        self.artifacts = {
            "lfsf_checkpoint": str(self.lfsf_checkpoint),
            "lfsf_sha256": checkpoint_sha256(self.lfsf_checkpoint),
            "vae_checkpoint": str(Path(vae_checkpoint)),
            "vae_sha256": checkpoint_sha256(vae_checkpoint),
        }
        if self.rf_checkpoint is not None:
            self.artifacts.update(
                {
                    "rf_checkpoint": str(self.rf_checkpoint),
                    "rf_sha256": checkpoint_sha256(self.rf_checkpoint),
                    "rf_weight_source": weight_source,
                }
            )

    @torch.inference_mode()
    def forward(
        self,
        stack: torch.Tensor,
        stack_mask: torch.Tensor | None = None,
        *,
        frame_chunk: int = 2,
    ) -> torch.Tensor:
        stack = stack.to(self.device)
        if stack_mask is None:
            stack_mask = torch.ones(
                stack.shape[:2], dtype=torch.bool, device=self.device
            )
        else:
            stack_mask = stack_mask.to(self.device).bool()
        base = run_lfsf(
            stack, stack_mask, self.scorer, self.vae, self.fusion, frame_chunk
        )
        if self.rf is None:
            return base["fused"]
        return run_lfsf_rf(
            base, stack, stack_mask, self.rf, self.gate_config
        )
