#!/usr/bin/env python3
"""Strict frozen-checkpoint LFSF inference for one official benchmark."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lfsf.data import MFIFInferenceDataset, load_dataset_registry, validate_dataset
from lfsf.inference import run_lfsf
from lfsf.models import FocusScorer, SoftPixelFusion, load_rffusion_vae
from lfsf.utils.checkpoint import load_checkpoint_metadata, load_lfsf_checkpoint
from lfsf.utils.io import save_image_tensor


def save_diagnostics(result, root, scene):
    root = Path(root)
    for name in ("expected_focus_index", "confidence", "uncertainty"):
        folder = root / name; folder.mkdir(parents=True, exist_ok=True)
        value = result["depth_pred" if name == "expected_focus_index" else name]
        lo, hi = value.amin(), value.amax()
        normalized = (value - lo) / (hi - lo).clamp_min(1e-8)
        save_image_tensor(normalized, folder / f"{scene}.png", normalize="0_1")
    folder = root / "weights"; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{scene}.npz", weights=result["weights"].cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/lfsf_seed3407.pth")
    parser.add_argument("--vae-config", default="configs/vq_encoder.yaml"); parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--dataset-config", default="configs/datasets.yaml"); parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--frame-chunk", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=0, help="Inference resize; 0 preserves native resolution (paper protocol)")
    parser.add_argument("--save-diagnostics", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metadata = load_checkpoint_metadata(args.checkpoint); checkpoint_size = int(metadata["args"]["image_size"])
    image_size = args.image_size
    if image_size > 0 and image_size != checkpoint_size:
        print(f"WARNING: requested image_size={image_size}; checkpoint training size was {checkpoint_size}")
    registry = load_dataset_registry(args.dataset_config); spec = registry[args.dataset]
    protocol = validate_dataset(args.data_root, args.dataset, spec)
    vae = load_rffusion_vae(args.vae_config, args.vae_ckpt, device).eval()
    scorer = FocusScorer(int(metadata["latent_channels"]), 64).to(device).eval()
    load_lfsf_checkpoint(args.checkpoint, scorer, device)
    fusion = SoftPixelFusion("pixel").to(device).eval()
    dataset = MFIFInferenceDataset(protocol["stack_root"], image_size=image_size, max_stack_size=24, scenes=protocol["scenes"])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for item in dataset:
        target_path = next(path for path in protocol["gt_root"].iterdir() if path.stem == item["name"])
        from PIL import Image
        target_size = Image.open(target_path).size[::-1]
        destination = output / f"{item['name']}.png"
        if destination.exists() and not args.overwrite: raise FileExistsError(destination)
        stack = item["stack"].unsqueeze(0).to(device); mask = torch.ones((1, stack.shape[1]), dtype=torch.bool, device=device)
        result = run_lfsf(stack, mask, scorer, vae, fusion, args.frame_chunk)
        fused = result["fused"]
        if fused.shape[-2:] != target_size:
            fused = F.interpolate(fused, size=target_size, mode="bicubic", align_corners=False)
        fused = fused.clamp(-1, 1)
        save_image_tensor(fused, destination)
        if args.save_diagnostics: save_diagnostics(result, output / "diagnostics", item["name"])


if __name__ == "__main__": main()
