"""Shared data and logging helpers for the public training entry points."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, Dataset, Subset

from lfsf.data import StackMFFDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data_config(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    train = config.get("train")
    validation = config.get("validation")
    if not isinstance(train, list) or not train or not isinstance(validation, dict):
        raise ValueError("Data config requires a non-empty 'train' list and 'validation' mapping")

    def resolve(entry: dict) -> dict:
        resolved = dict(entry)
        for key in ("stack_root", "depth_root", "aif_root"):
            value = resolved.get(key)
            if value and not Path(value).is_absolute():
                resolved[key] = str((path.parent / value).resolve())
        return resolved

    return [resolve(entry) for entry in train], resolve(validation)


def build_dataset(
    entries: list[dict] | dict,
    *,
    image_size: int,
    max_stack: int,
    require_depth: bool,
    random_stack_size: bool,
    max_samples: int = 0,
) -> Dataset:
    entries = entries if isinstance(entries, list) else [entries]
    datasets: list[Dataset] = []
    for entry in entries:
        missing = [key for key in ("stack_root", "aif_root") if not entry.get(key)]
        if require_depth and not entry.get("depth_root"):
            missing.append("depth_root")
        if missing:
            raise ValueError(f"Dataset {entry.get('name', '<unnamed>')} is missing {missing}")
        dataset = StackMFFDataset(
            stack_root=entry["stack_root"],
            depth_root=entry.get("depth_root") if require_depth else None,
            aif_root=entry["aif_root"],
            image_size=image_size,
            fixed_stack_size=0,
            random_stack_size=random_stack_size,
            min_stack_size=2,
            max_stack_size=max_stack,
            gray=False,
            normalize="-1_1",
        )
        print(f"{entry.get('name', Path(entry['stack_root']).name)}: {len(dataset)} scenes")
        datasets.append(dataset)
    combined: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if max_samples > 0:
        combined = Subset(combined, range(min(max_samples, len(combined))))
    return combined


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_args(namespace) -> dict:
    """Convert argparse values to weights-only-safe checkpoint metadata."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(namespace).items()
    }


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, online_model: torch.nn.Module, decay: float) -> None:
    for ema, online in zip(ema_model.parameters(), online_model.parameters()):
        ema.mul_(decay).add_(online, alpha=1.0 - decay)
    for ema, online in zip(ema_model.buffers(), online_model.buffers()):
        ema.copy_(online)
