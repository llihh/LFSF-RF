"""Multi-focus image stack dataset in StackMFF format.

Expected directory structure:
    stack_root/scene_001/1.png 2.png ... 8.png
    depth_root/scene_001.png
    aif_root/scene_001.png
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path):
    text = Path(path).stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=natural_key)
    return files


def _pil_to_tensor(img: Image.Image, gray: bool) -> torch.Tensor:
    if gray:
        arr = np.asarray(img.convert("L"), dtype=np.float32)[None, ...] / 255.0
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def _resize_tensor(x: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    if image_size <= 0:
        return x
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    kwargs = {"size": (image_size, image_size), "mode": mode}
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    return F.interpolate(x.unsqueeze(0), **kwargs).squeeze(0)


class StackMFFDataset(Dataset):
    """StackMFF-style multi-focus image stack dataset."""

    def __init__(
        self,
        stack_root: str,
        depth_root: Optional[str] = None,
        aif_root: Optional[str] = None,
        image_size: int = 384,
        max_stack_size: int = 24,
        min_stack_size: int = 2,
        random_stack_size: bool = True,
        fixed_stack_size: int = 8,
        transform=None,
        gray: bool = False,
        normalize: str = "-1_1",
        depth_resize_mode: str = "bilinear",
        image_resize_mode: str = "bicubic",
        sample_mode: str = "uniform",
    ):
        self.stack_root = Path(stack_root)
        self.depth_root = Path(depth_root) if depth_root else None
        self.aif_root = Path(aif_root) if aif_root else None
        self.image_size = int(image_size)
        self.max_stack_size = int(max_stack_size)
        self.min_stack_size = int(min_stack_size)
        self.random_stack_size = bool(random_stack_size)
        self.fixed_stack_size = int(fixed_stack_size)
        self.transform = transform
        self.gray = bool(gray)
        self.normalize = normalize
        self.depth_resize_mode = depth_resize_mode
        self.image_resize_mode = image_resize_mode
        self.sample_mode = sample_mode
        self.items = self._collect_items()
        if not self.items:
            raise RuntimeError(f"No stack scenes found under {self.stack_root}")

    def _collect_items(self) -> List[Dict]:
        items = []
        for scene_dir in sorted(
            [p for p in self.stack_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
        ):
            frames = list_images(scene_dir)
            if len(frames) < self.min_stack_size:
                continue
            depth_path = None
            if self.depth_root is not None:
                for ext in IMAGE_EXTS:
                    candidate = self.depth_root / f"{scene_dir.name}{ext}"
                    if candidate.exists():
                        depth_path = candidate
                        break
                if depth_path is None:
                    continue
            aif_path = None
            if self.aif_root is not None:
                for ext in IMAGE_EXTS:
                    candidate = self.aif_root / f"{scene_dir.name}{ext}"
                    if candidate.exists():
                        aif_path = candidate
                        break
                if aif_path is None:
                    continue
            items.append({
                "name": scene_dir.name,
                "frames": frames,
                "depth": depth_path,
                "aif": aif_path,
            })
        return items

    def _sample_indices(self, total: int) -> List[int]:
        max_n = min(total, self.max_stack_size if self.max_stack_size > 0 else total)
        if self.fixed_stack_size > 0:
            target = min(max(self.min_stack_size, self.fixed_stack_size), max_n)
        elif self.random_stack_size:
            target = random.randint(self.min_stack_size, max_n)
        else:
            target = max_n
        if target >= total:
            return list(range(total))
        if self.random_stack_size and self.sample_mode == "contiguous":
            start = random.randint(0, total - target)
            return list(range(start, start + target))
        return np.linspace(0, total - 1, target).round().astype(np.int64).tolist()

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path)
        x = _pil_to_tensor(img, self.gray)
        x = _resize_tensor(x, self.image_size, self.image_resize_mode)
        if self.transform is not None:
            x = self.transform(x)
        if self.normalize == "-1_1":
            x = x * 2.0 - 1.0
        elif self.normalize != "0_1":
            raise ValueError(f"Unsupported normalize mode: {self.normalize}")
        return x.float()

    def _load_depth(self, path: Optional[Path]) -> torch.Tensor:
        if path is None:
            size = self.image_size if self.image_size > 0 else 1
            return torch.zeros(1, size, size, dtype=torch.float32)
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        depth = torch.from_numpy(arr[None, ...])
        return _resize_tensor(depth, self.image_size, self.depth_resize_mode).clamp(0.0, 1.0).float()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        indices = self._sample_indices(len(item["frames"]))
        frames = [self._load_image(item["frames"][i]) for i in indices]
        stack = torch.stack(frames, dim=0)
        depth = self._load_depth(item["depth"])
        aif = self._load_image(item["aif"]) if item["aif"] is not None else torch.zeros_like(stack[0])
        return {
            "stack": stack,
            "depth": depth,
            "aif": aif,
            "has_aif": item["aif"] is not None,
            "name": item["name"],
            "num_frames": stack.shape[0],
        }


def stack_collate_fn(batch):
    max_n = max(item["stack"].shape[0] for item in batch)
    bsz = len(batch)
    c, h, w = batch[0]["stack"].shape[1:]
    stack = torch.zeros(bsz, max_n, c, h, w, dtype=torch.float32)
    mask = torch.zeros(bsz, max_n, dtype=torch.float32)
    names = []
    num_frames = []
    for i, item in enumerate(batch):
        n = item["stack"].shape[0]
        stack[i, :n] = item["stack"]
        mask[i, :n] = 1.0
        names.append(item["name"])
        num_frames.append(int(item["num_frames"]))
    return {
        "stack": stack,
        "stack_mask": mask,
        "depth": torch.stack([item["depth"] for item in batch], dim=0),
        "aif": torch.stack([item["aif"] for item in batch], dim=0),
        "has_aif": torch.tensor([item["has_aif"] for item in batch], dtype=torch.bool),
        "name": names,
        "num_frames": torch.tensor(num_frames, dtype=torch.long),
    }


class MFIFInferenceDataset(Dataset):
    """Dataset for MFIF inference without teacher targets.

    Supports two layouts:
      - Stack layout: root/<case_name>/*.png
      - Pair layout: root/source_1/*.png, root/source_2/*.png
    """

    def __init__(
        self,
        stack_root: str,
        image_size: int = 384,
        gray: bool = False,
        normalize: str = "-1_1",
        image_resize_mode: str = "bicubic",
        max_stack_size: int = 99,
        scenes: Optional[List[str]] = None,
    ):
        self.stack_root = Path(stack_root)
        self.image_size = int(image_size)
        self.gray = bool(gray)
        self.normalize = normalize
        self.image_resize_mode = image_resize_mode
        self.max_stack_size = max_stack_size
        self.scenes = scenes
        self.items = self._collect_items()
        if not self.items:
            raise RuntimeError(f"No scenes found under {self.stack_root}")

    def _collect_items(self) -> List[Dict]:
        items = []
        scene_dirs = ([self.stack_root / name for name in self.scenes] if self.scenes is not None else
                      sorted([p for p in self.stack_root.iterdir() if p.is_dir()], key=lambda p: natural_key(p)))
        for scene_dir in scene_dirs:
            frames = list_images(scene_dir)
            if len(frames) < 2:
                raise RuntimeError(f"Scene {scene_dir.name} has fewer than two frames")
            if len(frames) > self.max_stack_size > 0:
                indices = np.linspace(0, len(frames) - 1, self.max_stack_size).round().astype(np.int64)
                frames = [frames[int(index)] for index in indices]
            items.append({"name": scene_dir.name, "frames": frames})
        return items

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path)
        x = _pil_to_tensor(img, self.gray)
        x = _resize_tensor(x, self.image_size, self.image_resize_mode)
        if self.normalize == "-1_1":
            x = x * 2.0 - 1.0
        elif self.normalize != "0_1":
            raise ValueError(f"Unsupported normalize mode: {self.normalize}")
        return x.float()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = [self._load_image(p) for p in item["frames"]]
        stack = torch.stack(frames, dim=0)
        return {
            "stack": stack,
            "name": item["name"],
            "num_frames": stack.shape[0],
        }


class FlatFocalStackDataset(MFIFInferenceDataset):
    """Inference dataset for flat ``<stack_id>_<plane>.<ext>`` layouts."""

    def __init__(self, stack_root: str, expected_frames: int = 0, **kwargs):
        self.expected_frames = int(expected_frames)
        super().__init__(stack_root, **kwargs)

    def _collect_items(self) -> List[Dict]:
        grouped: Dict[str, List[tuple[int, Path]]] = {}
        pattern = re.compile(r"^(.+)_([0-9]+)$")
        for path in self.stack_root.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            match = pattern.match(path.stem)
            if match:
                grouped.setdefault(match.group(1), []).append((int(match.group(2)), path))
        items = []
        for name in sorted(grouped, key=lambda value: natural_key(Path(value))):
            indexed = sorted(grouped[name])
            indices = [index for index, _ in indexed]
            if indices != list(range(len(indices))):
                raise RuntimeError(f"Stack {name} has non-contiguous plane indices: {indices}")
            if self.expected_frames and len(indexed) != self.expected_frames:
                raise RuntimeError(
                    f"Stack {name} has {len(indexed)} frames; expected {self.expected_frames}"
                )
            if len(indexed) < 2:
                raise RuntimeError(f"Stack {name} has fewer than two frames")
            frames = [path for _, path in indexed]
            if len(frames) > self.max_stack_size > 0:
                selected = np.linspace(0, len(frames) - 1, self.max_stack_size).round().astype(np.int64)
                frames = [frames[int(index)] for index in selected]
            items.append({"name": name, "frames": frames})
        return items
