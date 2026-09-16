#!/usr/bin/env python3
"""Train LFSF with a frozen VQ encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lfsf.data import stack_collate_fn
from lfsf.inference import run_lfsf
from lfsf.losses import StackMFIFLoss
from lfsf.models import FocusScorer, SoftPixelFusion, load_rffusion_vae, softmax_weights
from lfsf.training import build_dataset, checkpoint_args, load_data_config, set_seed, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--vae-config", default="configs/vq_encoder.yaml", type=Path)
    parser.add_argument("--vae-ckpt", required=True, type=Path)
    parser.add_argument("--output-dir", default="runs/lfsf", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-stack", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frame-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def validation_psnr(loader, scorer, vae, fusion, device, frame_chunk: int) -> float:
    scorer.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", leave=False):
            stack = batch["stack"].to(device)
            mask = batch["stack_mask"].to(device)
            target = batch["aif"].to(device)
            output = run_lfsf(stack, mask, scorer, vae, fusion, frame_chunk)["fused"]
            error = (((output - target) / 2.0) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
            values.extend((-10.0 * torch.log10(error)).cpu().tolist())
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_config, validation_config = load_data_config(args.data_config)
    train_data = build_dataset(
        train_config,
        image_size=args.image_size,
        max_stack=args.max_stack,
        require_depth=True,
        random_stack_size=True,
        max_samples=args.max_train_samples,
    )
    validation_data = build_dataset(
        validation_config,
        image_size=args.image_size,
        max_stack=args.max_stack,
        require_depth=True,
        random_stack_size=False,
        max_samples=args.max_val_samples,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=stack_collate_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=stack_collate_fn,
    )

    vae = load_rffusion_vae(str(args.vae_config), str(args.vae_ckpt), device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    scorer = FocusScorer(vae.latent_channels, hidden_channels=64).to(device)
    fusion = SoftPixelFusion().to(device)
    criterion = StackMFIFLoss(
        lambda_depth=1.0,
        lambda_grad=1.0,
        lambda_int=1.0,
        lambda_ssim=0.5,
        lambda_rec=0.1,
        lambda_aif_l1=1.0,
        lambda_aif_ssim=0.5,
        lambda_aif_grad=0.5,
        lambda_focus=0.5,
        lambda_preserve=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rows: list[dict] = []
    best_psnr = -float("inf")
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        scorer.load_state_dict(payload["focus_scorer"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        rows = list(payload.get("rows", []))
        best_psnr = float(payload.get("best_val_psnr", -float("inf")))
        start_epoch = int(payload["epoch"]) + 1

    for epoch in range(start_epoch, args.epochs):
        scorer.train()
        totals: dict[str, float] = {}
        for batch in tqdm(train_loader, desc=f"LFSF epoch {epoch}"):
            stack = batch["stack"].to(device)
            depth = batch["depth"].to(device)
            target = batch["aif"].to(device)
            mask = batch["stack_mask"].to(device)
            batch_size, frames, channels, height, width = stack.shape
            with torch.no_grad():
                flat = stack.reshape(batch_size * frames, channels, height, width)
                latent_parts = [
                    vae.encode_image(flat[start : start + args.frame_chunk])
                    for start in range(0, len(flat), args.frame_chunk)
                ]
                latents = torch.cat(latent_parts)
                latent_stack = latents.reshape(batch_size, frames, *latents.shape[1:])
            gradient = FocusScorer.gradient_cue(flat, latents.shape[-2:])
            logits = scorer(latent_stack, gradient)
            weights = softmax_weights(logits, mask.bool())
            output = fusion(stack, weights, latent_stack, vae.decode_latent)
            losses = criterion(
                stack=stack,
                fused=output["fused"],
                depth_pred=output["depth_pred"],
                depth_gt=depth,
                weights=output["pixel_weights"],
                aif_target=target,
                focus_logits=logits,
                stack_mask=mask,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
        scheduler.step()
        stats = {key: value / max(len(train_loader), 1) for key, value in totals.items()}
        stats.update(
            epoch=epoch,
            lr=float(scheduler.get_last_lr()[0]),
            val_psnr=validation_psnr(
                validation_loader, scorer, vae, fusion, device, args.frame_chunk
            ),
        )
        rows.append(stats)
        write_csv(args.output_dir / "train_log.csv", rows)
        payload = {
            "focus_scorer": scorer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_psnr": max(best_psnr, stats["val_psnr"]),
            "rows": rows,
            "args": checkpoint_args(args),
            "latent_channels": vae.latent_channels,
        }
        torch.save(payload, args.output_dir / "checkpoint_latest.pth")
        if stats["val_psnr"] > best_psnr:
            best_psnr = stats["val_psnr"]
            torch.save(payload, args.output_dir / "checkpoint_best.pth")
        print(f"epoch={epoch} loss={stats['loss_total']:.6f} val_psnr={stats['val_psnr']:.4f}")


if __name__ == "__main__":
    main()
