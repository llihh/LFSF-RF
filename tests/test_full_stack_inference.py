import torch
import torch.nn.functional as F

from lfsf.inference import run_lfsf
from lfsf.models import FocusScorer, SoftPixelFusion


class FakeVAE:
    def encode_image(self, x):
        return F.avg_pool2d(x, 2)
    def decode_latent(self, z):
        return F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)


def test_chunk_invariance_and_masking():
    torch.manual_seed(3407)
    stack = torch.rand(1, 5, 3, 16, 16) * 2 - 1
    mask = torch.tensor([[True, True, True, True, False]])
    scorer = FocusScorer(3, hidden_channels=4).eval(); fusion = SoftPixelFusion("pixel")
    outputs = [run_lfsf(stack, mask, scorer, FakeVAE(), fusion, chunk) for chunk in (1,2,4,5)]
    for output in outputs:
        torch.testing.assert_close(output["weights"].sum(1), torch.ones_like(output["weights"].sum(1)))
        assert torch.count_nonzero(output["weights"][:, 4]) == 0
        torch.testing.assert_close(output["fused"], outputs[0]["fused"], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(output["weights"], outputs[0]["weights"], rtol=1e-6, atol=1e-7)
