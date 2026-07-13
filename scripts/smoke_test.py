#!/usr/bin/env python3
"""Real one-scene VQ + LFSF + LFSF-RF smoke test."""

import argparse
from pathlib import Path

import torch

from lfsf.data import MFIFInferenceDataset, load_dataset_registry, validate_dataset
from lfsf.inference import run_lfsf, run_lfsf_rf
from lfsf.models import FocusScorer, NoSourceShortcutStackRF, SoftPixelFusion, load_rffusion_vae
from lfsf.utils.checkpoint import load_checkpoint_metadata, load_lfsf_checkpoint, load_refiner_checkpoint
from lfsf.utils.io import save_image_tensor


def main():
    p=argparse.ArgumentParser(); p.add_argument("--vae-ckpt",required=True); p.add_argument("--data-root",required=True)
    p.add_argument("--dataset",default="Middlebury"); p.add_argument("--scene",default="Motorcycle"); p.add_argument("--device",default="cuda:0")
    p.add_argument("--output-dir",default="smoke_output"); p.add_argument("--dataset-config",default="configs/datasets.yaml")
    p.add_argument("--vae-config",default="configs/vq_encoder.yaml"); p.add_argument("--lfsf-checkpoint",default="checkpoints/lfsf_seed3407.pth"); p.add_argument("--rf-checkpoint",default="checkpoints/lfsf_rf_seed3407.pth")
    args=p.parse_args(); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    lm=load_checkpoint_metadata(args.lfsf_checkpoint); rm=load_checkpoint_metadata(args.rf_checkpoint); registry=load_dataset_registry(args.dataset_config)
    protocol=validate_dataset(args.data_root,args.dataset,registry[args.dataset]);
    if args.scene not in protocol["scenes"]: raise ValueError(f"Scene {args.scene} is not in the official list")
    vae=load_rffusion_vae(args.vae_config,args.vae_ckpt,device).eval(); scorer=FocusScorer(int(lm["latent_channels"]),64).to(device).eval(); load_lfsf_checkpoint(args.lfsf_checkpoint,scorer,device)
    fusion=SoftPixelFusion("pixel").to(device).eval(); ma=rm["args"]; rf=NoSourceShortcutStackRF(in_channels=int(ma["image_channels"]),out_channels=int(ma["image_channels"]),base=int(ma["base_channels"])).to(device).eval(); load_refiner_checkpoint(args.rf_checkpoint,rf,device,"NoSourceShortcutStackRF")
    item=MFIFInferenceDataset(protocol["stack_root"],image_size=0,max_stack_size=int(ma["max_stack"]),scenes=[args.scene])[0]
    stack=item["stack"].unsqueeze(0).to(device); mask=torch.ones((1,stack.shape[1]),dtype=torch.bool,device=device); base=run_lfsf(stack,mask,scorer,vae,fusion,1); endpoint=run_lfsf_rf(base,stack,mask,rf,rm["gate_config"])
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); save_image_tensor(base["fused"],output/f"{args.scene}_lfsf.png"); save_image_tensor(endpoint,output/f"{args.scene}_lfsf_rf.png")
    print(f"PASS: {args.dataset}/{args.scene} -> {output}")


if __name__=="__main__": main()
