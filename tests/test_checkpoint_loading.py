from pathlib import Path

import pytest
import torch

from lfsf.models import DirectResidualStackCNN, FocusScorer, NoSourceShortcutStackRF
from lfsf.utils.checkpoint import load_lfsf_checkpoint, load_refiner_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_real_paper_checkpoints_strict_load():
    scorer = FocusScorer(3)
    load_lfsf_checkpoint(ROOT / "checkpoints/lfsf_seed3407.pth", scorer, "cpu")
    rf = NoSourceShortcutStackRF()
    load_refiner_checkpoint(ROOT / "checkpoints/lfsf_rf_seed3407.pth", rf, "cpu", "NoSourceShortcutStackRF")
    direct = DirectResidualStackCNN()
    load_refiner_checkpoint(ROOT / "checkpoints/direct_seed3407.pth", direct, "cpu", "DirectResidualStackCNN")
    assert sum(p.numel() for p in scorer.parameters()) == 39361
    assert sum(p.numel() for p in rf.parameters()) == 838275
    assert sum(p.numel() for p in direct.parameters()) == 838275


def test_wrong_model_type_fails():
    with pytest.raises(RuntimeError, match="model_type mismatch"):
        load_refiner_checkpoint(ROOT / "checkpoints/lfsf_rf_seed3407.pth", NoSourceShortcutStackRF(), "cpu", "DirectResidualStackCNN")


def test_missing_state_key_fails(tmp_path):
    payload = torch.load(ROOT / "checkpoints/lfsf_seed3407.pth", map_location="cpu", weights_only=True)
    payload["focus_scorer"].pop(next(iter(payload["focus_scorer"])))
    path = tmp_path / "broken.pth"; torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_lfsf_checkpoint(path, FocusScorer(3), "cpu")


def test_third_party_vq_imports():
    from lfsf.third_party.ldm.models.autoencoder import VQModel
    from lfsf.third_party.ldm.modules.attention import LinearAttention
    assert VQModel and LinearAttention
