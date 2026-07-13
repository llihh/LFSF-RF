"""Paper full-reference PSNR/SSIM protocol (prediction versus all-in-focus GT)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def rgb_to_bt601_gray(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        return value.astype(np.float64)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("Expected grayscale or RGB image")
    value = value.astype(np.float64)
    return 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]


def load_uint8(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def compute_psnr(pred, gt, data_range=255) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(float(data_range) ** 2 / mse)


def compute_ssim(pred, gt, data_range=255) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    return float(structural_similarity(gt, pred, data_range=float(data_range)))


def evaluate_full_reference(pred_path, gt_path):
    pred_rgb, gt_rgb = load_uint8(pred_path), load_uint8(gt_path)
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(f"Canonical-resolution mismatch: {pred_rgb.shape} vs {gt_rgb.shape}")
    pred_y, gt_y = rgb_to_bt601_gray(pred_rgb), rgb_to_bt601_gray(gt_rgb)
    return {"PSNR": compute_psnr(pred_y, gt_y), "SSIM": compute_ssim(pred_y, gt_y)}
