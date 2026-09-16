#!/usr/bin/env python3
"""Run LFSF or LFSF-RF on a folder of focal stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lfsf.data import FlatFocalStackDataset, MFIFInferenceDataset
from lfsf.pipeline import FrozenLFSFPipeline
from lfsf.utils.io import save_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", required=True, type=Path)
    parser.add_argument("--layout", choices=("nested", "flat"), default="nested")
    parser.add_argument(
        "--expected-frames", type=int, default=0,
        help="Require this many planes per stack in flat layout",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vae-config", default="configs/vq_encoder.yaml", type=Path)
    parser.add_argument("--vae-ckpt", required=True, type=Path)
    parser.add_argument(
        "--lfsf-checkpoint", default="checkpoints/lfsf_seed3407.pth", type=Path
    )
    parser.add_argument("--rf-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-chunk", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_chunk < 1:
        raise ValueError("--frame-chunk must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipeline = FrozenLFSFPipeline(
        lfsf_checkpoint=args.lfsf_checkpoint,
        rf_checkpoint=args.rf_checkpoint,
        vae_config=args.vae_config,
        vae_checkpoint=args.vae_ckpt,
        device=device,
        weight_source="ema",
    ).eval()
    dataset_class = FlatFocalStackDataset if args.layout == "flat" else MFIFInferenceDataset
    dataset_kwargs = {
        "image_size": 0,
        "max_stack_size": args.max_frames,
    }
    if args.layout == "flat":
        dataset_kwargs["expected_frames"] = args.expected_frames
    dataset = dataset_class(str(args.stack_root), **dataset_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in dataset:
        destination = args.output_dir / f"{item['name']}.png"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        prediction = pipeline(
            item["stack"].unsqueeze(0), frame_chunk=args.frame_chunk
        )
        save_image_tensor(prediction, destination)
        print(f"[{item['name']}] {item['num_frames']} frames -> {destination}")
    (args.output_dir / "inference.json").write_text(
        json.dumps(
            {
                "stack_root": str(args.stack_root.resolve()),
                "layout": args.layout,
                "method": "LFSF-RF" if args.rf_checkpoint else "LFSF",
                "frame_chunk": args.frame_chunk,
                "max_frames": args.max_frames,
                "artifacts": pipeline.artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
