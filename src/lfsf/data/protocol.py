"""Strict five-benchmark scene registry and pairing checks."""

from __future__ import annotations

from pathlib import Path

import yaml

from .datasets import list_images, IMAGE_EXTS


def read_scene_list(path):
    scenes = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"Duplicate scene names in {path}")
    return scenes


def load_dataset_registry(path):
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry = payload.get("datasets", payload)
    repo_root = config_path.parent.parent
    for spec in registry.values():
        scene_list = Path(spec["scene_list"])
        spec["scene_list"] = str(scene_list if scene_list.is_absolute() else repo_root / scene_list)
    return registry


def find_unique_image(folder: Path, scene: str) -> Path:
    matches = [p for p in folder.iterdir() if p.is_file() and p.stem == scene and p.suffix.lower() in IMAGE_EXTS]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one image for '{scene}' in {folder}; found {len(matches)}")
    return matches[0]


def validate_dataset(data_root, dataset, spec, require_gt=True):
    dataset_root = Path(data_root) / dataset
    stack_root = dataset_root / spec["stack_subdir"]
    gt_root = dataset_root / spec["gt_subdir"]
    scenes = read_scene_list(spec["scene_list"])
    expected = int(spec["expected_scenes"])
    errors = []
    if len(scenes) != expected:
        errors.append(f"scene list has {len(scenes)} entries; expected {expected}")
    for scene in scenes:
        frames = list_images(stack_root / scene)
        if len(frames) < 2:
            errors.append(f"{scene}: expected >=2 readable frame paths, found {len(frames)}")
        if require_gt:
            try:
                find_unique_image(gt_root, scene)
            except (FileNotFoundError, NotADirectoryError) as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError(f"Dataset protocol validation failed for {dataset}:\n" + "\n".join(errors))
    return {"scenes": scenes, "stack_root": stack_root, "gt_root": gt_root}
