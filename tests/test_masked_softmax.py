"""Tests for masked softmax weight computation."""

import torch
from lfsf.models.focus_scorer import softmax_weights


def test_all_valid_frames():
    """Weights should sum to 1 across frames when all frames are valid."""
    b, n, h, w = 2, 4, 16, 16
    logits = torch.randn(b, n, h, w)
    weights = softmax_weights(logits)
    assert weights.shape == (b, n, 1, h, w)
    assert torch.allclose(weights.sum(dim=1), torch.ones(b, 1, h, w), atol=1e-5)


def test_padded_frames_zero_weight():
    """Padded frames (mask=0) should have zero weight."""
    b, n, h, w = 2, 6, 16, 16
    logits = torch.randn(b, n, h, w)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, 4:] = False  # last 2 frames are padded
    weights = softmax_weights(logits, mask)
    assert torch.all(weights[:, 4:] == 0), "Padded frames should have zero weight"


def test_valid_frames_normalized():
    """Valid frames should still sum to 1."""
    b, n, h, w = 2, 6, 16, 16
    logits = torch.randn(b, n, h, w)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, 3:] = False
    weights = softmax_weights(logits, mask)
    assert torch.allclose(weights[:, :3].sum(dim=1), torch.ones(b, 1, h, w), atol=1e-5)


def test_single_frame():
    """Single-frame stack should have weight 1 everywhere."""
    b, n, h, w = 2, 1, 16, 16
    logits = torch.randn(b, n, h, w)
    weights = softmax_weights(logits)
    assert torch.allclose(weights, torch.ones(b, 1, 1, h, w), atol=1e-7)


def test_all_masked():
    """An invalid all-padded stack must fail explicitly."""
    import pytest
    b, n, h, w = 2, 4, 16, 16
    logits = torch.randn(b, n, h, w)
    mask = torch.zeros(b, n, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one valid frame"):
        softmax_weights(logits, mask)
