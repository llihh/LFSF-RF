#!/usr/bin/env python3
"""Evaluate the frozen Table 3 first/last-source protocol."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from lfsf.data.datasets import list_images
from lfsf.data.protocol import find_unique_image, read_scene_list
from lfsf.metrics import evaluate_stack_fusion


def gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True); parser.add_argument("--stack-root", required=True)
    parser.add_argument("--scene-list", required=True); parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--dataset", required=True); parser.add_argument("--method", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != args.expected_scenes:
        raise RuntimeError(f"Expected {args.expected_scenes} scenes, got {len(scenes)}")
    rows = []
    for scene in scenes:
        frames = list_images(Path(args.stack_root) / scene)
        if len(frames) < 2: raise RuntimeError(f"{scene}: fewer than two source frames")
        fused_path = find_unique_image(Path(args.pred_dir), scene)
        fused = gray(fused_path); sources = [gray(frames[0]), gray(frames[-1])]
        if any(source.shape != fused.shape for source in sources):
            raise ValueError(f"{scene}: source/prediction resolution mismatch")
        rows.append({"dataset": args.dataset, "scene": scene, "method": args.method,
                     **evaluate_stack_fusion(np.stack(sources), fused)})
    output = Path(args.output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
