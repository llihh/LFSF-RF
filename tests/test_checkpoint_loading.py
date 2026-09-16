import pytest
import torch

from lfsf.models import FocusScorer, NoSourceShortcutStackRF
from lfsf.utils.checkpoint import load_lfsf_checkpoint, load_refiner_checkpoint

def write_test_checkpoints(tmp_path):
    scorer = FocusScorer(3)
    rf = NoSourceShortcutStackRF()
    paths = {
        "lfsf": tmp_path / "lfsf.pth",
        "rf": tmp_path / "rf.pth",
    }
    torch.save(
        {"focus_scorer": scorer.state_dict(), "latent_channels": 3}, paths["lfsf"]
    )
    torch.save(
        {
            "velocity_model": rf.state_dict(),
            "online_model": rf.state_dict(),
            "model_type": "NoSourceShortcutStackRF",
            "trainable_params": 838275,
        },
        paths["rf"],
    )
    return paths


def test_valid_checkpoints_strict_load(tmp_path):
    paths = write_test_checkpoints(tmp_path)
    scorer = FocusScorer(3)
    load_lfsf_checkpoint(paths["lfsf"], scorer, "cpu")
    rf = NoSourceShortcutStackRF()
    load_refiner_checkpoint(paths["rf"], rf, "cpu", "NoSourceShortcutStackRF")
    assert sum(p.numel() for p in scorer.parameters()) == 39361
    assert sum(p.numel() for p in rf.parameters()) == 838275


def test_wrong_model_type_fails(tmp_path):
    paths = write_test_checkpoints(tmp_path)
    with pytest.raises(RuntimeError, match="model_type mismatch"):
        load_refiner_checkpoint(paths["rf"], NoSourceShortcutStackRF(), "cpu", "AnotherModel")


def test_missing_state_key_fails(tmp_path):
    paths = write_test_checkpoints(tmp_path)
    payload = torch.load(paths["lfsf"], map_location="cpu", weights_only=True)
    payload["focus_scorer"].pop(next(iter(payload["focus_scorer"])))
    path = tmp_path / "broken.pth"; torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_lfsf_checkpoint(path, FocusScorer(3), "cpu")


def test_third_party_vq_imports():
    from lfsf.third_party.ldm.models.autoencoder import VQModel
    from lfsf.third_party.ldm.modules.attention import LinearAttention
    assert VQModel and LinearAttention
