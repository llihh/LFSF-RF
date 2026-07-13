#!/usr/bin/env python3
"""Frozen Direct residual control inference (not a proposed method)."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from lfsf.data import MFIFInferenceDataset, load_dataset_registry, validate_dataset
from lfsf.inference import run_direct, run_lfsf
from lfsf.models import DirectResidualStackCNN, FocusScorer, SoftPixelFusion, load_rffusion_vae
from lfsf.utils.checkpoint import load_checkpoint_metadata, load_lfsf_checkpoint, load_refiner_checkpoint
from lfsf.utils.io import save_image_tensor


def main():
    p=argparse.ArgumentParser(); p.add_argument("--lfsf-checkpoint",default="checkpoints/lfsf_seed3407.pth"); p.add_argument("--direct-checkpoint",default="checkpoints/direct_seed3407.pth")
    p.add_argument("--vae-config",default="configs/vq_encoder.yaml"); p.add_argument("--vae-ckpt",required=True); p.add_argument("--dataset-config",default="configs/datasets.yaml")
    p.add_argument("--data-root",required=True); p.add_argument("--dataset",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--frame-chunk",type=int,default=2); p.add_argument("--weight-source",choices=("ema","online"),default="ema"); p.add_argument("--overwrite",action="store_true")
    args=p.parse_args(); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    lm=load_checkpoint_metadata(args.lfsf_checkpoint); dm=load_checkpoint_metadata(args.direct_checkpoint); registry=load_dataset_registry(args.dataset_config)
    protocol=validate_dataset(args.data_root,args.dataset,registry[args.dataset]); vae=load_rffusion_vae(args.vae_config,args.vae_ckpt,device).eval()
    scorer=FocusScorer(int(lm["latent_channels"]),64).to(device).eval(); load_lfsf_checkpoint(args.lfsf_checkpoint,scorer,device); fusion=SoftPixelFusion("pixel").to(device).eval()
    ma=dm["args"]; model=DirectResidualStackCNN(in_channels=int(ma["image_channels"]),out_channels=int(ma["image_channels"]),base=int(ma["base_channels"])).to(device).eval()
    load_refiner_checkpoint(args.direct_checkpoint,model,device,"DirectResidualStackCNN",args.weight_source)
    data=MFIFInferenceDataset(protocol["stack_root"],image_size=0,max_stack_size=int(ma["max_stack"]),scenes=protocol["scenes"]); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    for item in data:
        destination=output/f"{item['name']}.png"
        if destination.exists() and not args.overwrite: raise FileExistsError(destination)
        stack=item["stack"].unsqueeze(0).to(device); mask=torch.ones((1,stack.shape[1]),dtype=torch.bool,device=device)
        base=run_lfsf(stack,mask,scorer,vae,fusion,args.frame_chunk); endpoint=run_direct(base,stack,mask,model,dm["gate_config"])
        target=next(path for path in protocol["gt_root"].iterdir() if path.stem==item["name"]); assert endpoint.shape[-2:]==Image.open(target).size[::-1]
        save_image_tensor(endpoint,destination)


if __name__=="__main__": main()
