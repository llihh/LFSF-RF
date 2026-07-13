"""Image I/O utilities."""

import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path) -> List:
    """Natural sort key: splits on digits for human-friendly ordering."""
    text = Path(path).stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def list_images(folder: Path) -> List[Path]:
    """List image files in a directory, sorted by natural key."""
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=natural_key)
    return files


def read_image_tensor(path, gray: bool = False) -> torch.Tensor:
    """Read an image file as a float32 tensor in [0, 1].

    Args:
        path: Path to the image file.
        gray: If True, convert to single-channel grayscale.

    Returns:
        Tensor of shape [C, H, W] with values in [0, 1].
    """
    img = Image.open(path)
    if gray:
        arr = np.asarray(img.convert("L"), dtype=np.float32)[None, ...] / 255.0
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def save_image_tensor(tensor: torch.Tensor, path, normalize: str = "-1_1"):
    """Save a tensor as an image file.

    Args:
        tensor: [C, H, W] or [1, C, H, W] tensor.
        path: Output file path.
        normalize: Input range, either "-1_1" or "0_1".
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    x = tensor.detach().cpu().float()
    if normalize == "-1_1":
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    if x.shape[0] == 1:
        arr = np.rint(x[0].numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        arr = np.rint(x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    img.save(path)
