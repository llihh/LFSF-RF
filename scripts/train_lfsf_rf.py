#!/usr/bin/env python3
"""Train the final shortcut-free LFSF-RF refiner from a frozen LFSF model."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lfsf.data import stack_collate_fn
from lfsf.inference import run_lfsf
from lfsf.losses import compute_focus_edit_mask, sobel_magnitude
from lfsf.models import FocusScorer, NoSourceShortcutStackRF, SoftPixelFusion, load_rffusion_vae
from lfsf.training import (
    build_dataset,
    checkpoint_args,
    load_data_config,
    set_seed,
    update_ema,
    write_csv,
)
from lfsf.utils.checkpoint import load_checkpoint_metadata, load_lfsf_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--lfsf-checkpoint", required=True, type=Path)
    parser.add_argument("--vae-config", default="configs/vq_encoder.yaml", type=Path)
    parser.add_argument("--vae-ckpt", required=True, type=Path)
    parser.add_argument("--output-dir", default="runs/lfsf_rf", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-stack", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--frame-chunk",
        type=int,
        default=0,
        help="VQ encoder chunk size in frames; 0 matches the submitted full-batch training",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--t0-probability", type=float, default=0.25)
    parser.add_argument("--t-min", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-endpoint", type=float, default=1.5)
    parser.add_argument("--lambda-grad", type=float, default=0.25)
    parser.add_argument("--lambda-anchor", type=float, default=0.4)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.8)
    parser.add_argument("--detail-weight", type=float, default=1.2)
    parser.add_argument("--alpha-uncert", type=float, default=0.70)
    parser.add_argument("--beta-detail", type=float, default=0.85)
    parser.add_argument("--gamma-edge", type=float, default=0.25)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--gate-soft-width", type=float, default=0.35)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def gate_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "alpha_uncert": args.alpha_uncert,
        "beta_detail": args.beta_detail,
        "gamma_edge": args.gamma_edge,
        "threshold": args.gate_threshold,
        "soft_width": args.gate_soft_width,
    }


def masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def loss_terms(velocity, target_velocity, endpoint, target, f0, gate, confidence, uncertainty, args):
    weight = (
        1.0
        + args.boundary_weight * gate["edge_stack"]
        + args.uncertainty_weight * uncertainty.detach()
        + args.detail_weight * gate["detail_gap"]
    ).detach().clamp(1.0, 1.0 + args.boundary_weight + args.uncertainty_weight + args.detail_weight)
    loss_fm = masked_mean((velocity - target_velocity).square(), weight)
    loss_endpoint = masked_mean((endpoint - target).abs(), weight)
    gradient_weight = (1.0 + args.boundary_weight * gate["edge_stack"]).detach()
    loss_gradient = masked_mean(
        (sobel_magnitude(endpoint) - sobel_magnitude(target)).abs(), gradient_weight
    )
    reliable = ((1.0 - gate["edit_mask"]) * confidence.detach()).clamp(0.0, 1.0)
    loss_anchor = masked_mean((endpoint - f0).abs(), reliable)
    total = (
        args.lambda_fm * loss_fm
        + args.lambda_endpoint * loss_endpoint
        + args.lambda_grad * loss_gradient
        + args.lambda_anchor * loss_anchor
    )
    return {
        "loss_total": total,
        "loss_fm": loss_fm,
        "loss_endpoint": loss_endpoint,
        "loss_grad": loss_gradient,
        "loss_anchor": loss_anchor,
    }


@torch.no_grad()
def refine(model, f0, stack, mask, confidence, uncertainty, gate):
    time = torch.zeros(f0.shape[0], device=f0.device, dtype=f0.dtype)
    velocity = model(
        f0,
        time,
        stack,
        mask,
        confidence,
        uncertainty,
        gate["edit_mask"],
        gate["detail_gap"],
        gate["edge_base"],
        gate["edge_stack"],
    )
    return (f0 + velocity).clamp(-1.0, 1.0)


@torch.no_grad()
def validate(model, loader, vae, scorer, fusion, device, args) -> dict[str, float]:
    model.eval()
    psnr: list[float] = []
    l1: list[float] = []
    for batch in tqdm(loader, desc="validate", leave=False):
        stack = batch["stack"].to(device)
        mask = batch["stack_mask"].to(device)
        target = batch["aif"].to(device)
        base = run_lfsf(stack, mask, scorer, vae, fusion, args.frame_chunk)
        f0 = base["fused"]
        confidence = base["confidence"]
        uncertainty = 1.0 - confidence
        gate = compute_focus_edit_mask(f0, stack, mask, uncertainty, **gate_config(args))
        endpoint = refine(model, f0, stack, mask, confidence, uncertainty, gate)
        error = (((endpoint - target) / 2.0) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
        psnr.extend((-10.0 * torch.log10(error)).cpu().tolist())
        l1.extend((endpoint - target).abs().mean(dim=(1, 2, 3)).cpu().tolist())
    return {"val_psnr": float(np.mean(psnr)), "val_loss": float(np.mean(l1))}


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
        require_depth=False,
        random_stack_size=False,
        max_samples=args.max_train_samples,
    )
    validation_data = build_dataset(
        validation_config,
        image_size=args.image_size,
        max_stack=args.max_stack,
        require_depth=False,
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
    metadata = load_checkpoint_metadata(args.lfsf_checkpoint)
    scorer = FocusScorer(int(metadata["latent_channels"]), hidden_channels=64).to(device).eval()
    load_lfsf_checkpoint(args.lfsf_checkpoint, scorer, device)
    for parameter in scorer.parameters():
        parameter.requires_grad = False
    fusion = SoftPixelFusion().to(device).eval()
    model = NoSourceShortcutStackRF().to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rows: list[dict] = []
    best_psnr = -math.inf
    best_loss = math.inf
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("model_type") != "NoSourceShortcutStackRF":
            raise RuntimeError("Resume checkpoint is not an LFSF-RF model")
        model.load_state_dict(payload["online_model"], strict=True)
        ema_model.load_state_dict(payload["velocity_model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        rows = list(payload.get("rows", []))
        best_psnr = float(payload.get("best_psnr", -math.inf))
        best_loss = float(payload.get("best_loss", math.inf))
        start_epoch = int(payload["epoch"]) + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals: dict[str, float] = {}
        zero_count = 0
        sample_count = 0
        for batch in tqdm(train_loader, desc=f"LFSF-RF epoch {epoch}"):
            stack = batch["stack"].to(device)
            mask = batch["stack_mask"].to(device)
            target = batch["aif"].to(device)
            with torch.no_grad():
                base = run_lfsf(stack, mask, scorer, vae, fusion, args.frame_chunk)
                f0 = base["fused"]
                confidence = base["confidence"]
                uncertainty = 1.0 - confidence
                gate = compute_focus_edit_mask(f0, stack, mask, uncertainty, **gate_config(args))
            time = torch.rand(f0.shape[0], device=device).clamp_min(args.t_min)
            use_t0 = torch.rand(f0.shape[0], device=device) < args.t0_probability
            time = torch.where(use_t0, torch.zeros_like(time), time)
            time_map = time[:, None, None, None]
            target_velocity = target - f0
            state = (1.0 - time_map) * f0 + time_map * target
            velocity = model(
                state,
                time,
                stack,
                mask,
                confidence,
                uncertainty,
                gate["edit_mask"],
                gate["detail_gap"],
                gate["edge_base"],
                gate["edge_stack"],
            )
            endpoint = state + (1.0 - time_map) * velocity
            losses = loss_terms(
                velocity, target_velocity, endpoint, target, f0, gate, confidence, uncertainty, args
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)
            zero_count += int(use_t0.sum())
            sample_count += int(use_t0.numel())
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
        scheduler.step()
        stats = {key: value / max(len(train_loader), 1) for key, value in totals.items()}
        stats.update(validate(ema_model, validation_loader, vae, scorer, fusion, device, args))
        stats.update(
            epoch=epoch,
            lr=float(scheduler.get_last_lr()[0]),
            t0_fraction=zero_count / max(sample_count, 1),
        )
        rows.append(stats)
        write_csv(args.output_dir / "train_log.csv", rows)
        improved = stats["val_psnr"] > best_psnr + 1e-12
        tied = abs(stats["val_psnr"] - best_psnr) <= 1e-12 and stats["val_loss"] < best_loss
        if improved or tied:
            best_psnr, best_loss = stats["val_psnr"], stats["val_loss"]
        payload = {
            "model_type": "NoSourceShortcutStackRF",
            "stage": "lfsf_rf",
            "velocity_model": ema_model.state_dict(),
            "online_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": {**checkpoint_args(args), "image_channels": 3, "base_channels": 32},
            "gate_config": gate_config(args),
            "stats": stats,
            "trainable_params": sum(parameter.numel() for parameter in model.parameters()),
            "best_psnr": best_psnr,
            "best_loss": best_loss,
            "rows": rows,
        }
        torch.save(payload, args.output_dir / "checkpoint_latest.pth")
        if improved or tied:
            torch.save(payload, args.output_dir / "checkpoint_best.pth")
        if epoch + 1 == args.epochs:
            torch.save(payload, args.output_dir / "checkpoint_final.pth")
        print(f"epoch={epoch} loss={stats['loss_total']:.6f} val_psnr={stats['val_psnr']:.4f}")


if __name__ == "__main__":
    main()
