"""LFSF: Latent-Focus Soft Fusion for multi-focus image stacks."""

from lfsf.models.focus_scorer import FocusScorer, softmax_weights
from lfsf.models.pixel_fusion import SoftPixelFusion
from lfsf.models.rf_refiner import (
    DirectResidualStackCNN,
    FocusConditionedStackRF,
    FocusGatedStackRF,
    NoSourceShortcutStackRF,
)
from lfsf.models.vq_encoder import RFVAEWrapper, load_rffusion_vae

__all__ = [
    "FocusScorer",
    "softmax_weights",
    "SoftPixelFusion",
    "RFVAEWrapper",
    "load_rffusion_vae",
    "FocusConditionedStackRF",
    "FocusGatedStackRF",
    "NoSourceShortcutStackRF",
    "DirectResidualStackCNN",
]
