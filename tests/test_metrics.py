import numpy as np

from lfsf.metrics import compute_psnr, compute_ssim, evaluate_stack_fusion, rgb_to_bt601_gray


def test_bt601_and_full_reference_identity():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    np.testing.assert_allclose(rgb_to_bt601_gray(rgb), [[76.245, 149.685, 29.07]])
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.isinf(compute_psnr(gray, gray))
    assert compute_ssim(gray, gray) == 1.0


def test_standard_psnr():
    gt = np.zeros((8, 8)); pred = np.ones((8, 8))
    np.testing.assert_allclose(compute_psnr(pred, gt), 20 * np.log10(255.0))


def test_stack_metrics_use_first_and_last_only():
    first = np.zeros((32, 32), dtype=np.uint8); last = np.full((32, 32), 255, dtype=np.uint8)
    fused = np.full((32, 32), 127, dtype=np.uint8)
    baseline = evaluate_stack_fusion(np.stack([first, last]), fused)
    changed_middle = evaluate_stack_fusion(np.stack([first, np.random.default_rng(1).integers(0,256,(32,32),dtype=np.uint8), last]), fused)
    assert baseline == changed_middle
