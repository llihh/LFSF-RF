#!/usr/bin/env python3
"""Evaluate per-scene predictions with PSNR and SSIM."""

import argparse
import csv
from pathlib import Path

from lfsf.data.protocol import find_unique_image, read_scene_list
from lfsf.metrics import evaluate_full_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--expected-scenes", required=True, type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != args.expected_scenes:
        raise RuntimeError(f"Expected {args.expected_scenes} scenes, got {len(scenes)}")
    rows = []
    for scene in scenes:
        pred = find_unique_image(Path(args.pred_dir), scene)
        gt = find_unique_image(Path(args.gt_dir), scene)
        values = evaluate_full_reference(pred, gt)
        rows.append({"dataset": args.dataset, "scene": scene, "method": args.method,
                     "pred_path": str(pred), "gt_path": str(gt), **values})
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
