import inspect

import torch

from lfsf.inference import run_lfsf_rf
from lfsf.models import NoSourceShortcutStackRF


def test_rf_has_no_f0_shortcut_and_matches_released_size():
    signature = inspect.signature(NoSourceShortcutStackRF.forward)
    assert "f0" not in signature.parameters
    assert sum(p.numel() for p in NoSourceShortcutStackRF().parameters()) == 838275


class ConstantRF(torch.nn.Module):
    def forward(self, **kwargs):
        return torch.ones_like(kwargs["x_t"])


def test_rf_endpoint_is_clamped_not_hard_masked(monkeypatch):
    monkeypatch.setattr("lfsf.inference.refiner.compute_focus_edit_mask", lambda *a, **k: {
        name: torch.zeros(1,1,2,2) for name in ("edit_mask","detail_gap","edge_base","edge_stack")})
    base={"fused":torch.full((1,3,2,2),0.5),"confidence":torch.ones(1,1,2,2)}
    stack=torch.zeros(1,2,3,2,2); mask=torch.ones(1,2)
    result=run_lfsf_rf(base,stack,mask,ConstantRF(),{})
    assert torch.all(result == 1.0)
