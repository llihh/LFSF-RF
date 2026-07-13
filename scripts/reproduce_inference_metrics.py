#!/usr/bin/env python3
"""Run frozen inference, full-reference evaluation, and aggregation end to end."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from lfsf.data import load_dataset_registry
from lfsf.utils.checkpoint import checkpoint_sha256


def run(command, log):
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--artifact-config",required=True)
    p.add_argument("--dataset-config",default="configs/datasets.yaml"); p.add_argument("--methods",nargs="+",choices=("lfsf","lfsf_rf"),default=("lfsf","lfsf_rf"))
    p.add_argument("--device",default="cuda:0"); p.add_argument("--output-root",required=True); p.add_argument("--overwrite",action="store_true")
    args=p.parse_args(); root=Path(args.output_root)
    if root.exists() and any(root.iterdir()) and not args.overwrite: raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True); logs=root/"logs"; logs.mkdir(exist_ok=True); log=logs/"reproduction.log"
    artifacts=yaml.safe_load(Path(args.artifact_config).read_text(encoding="utf-8"))
    for name,spec in artifacts.items():
        if not isinstance(spec,dict) or "path" not in spec: continue
        actual=checkpoint_sha256(spec["path"])
        if actual!=spec["sha256"]: raise RuntimeError(f"{name} SHA-256 mismatch")
    registry=load_dataset_registry(args.dataset_config); per_scene=root/"metrics"/"per_scene"; per_scene.mkdir(parents=True,exist_ok=True)
    for method in args.methods:
        for dataset,spec in registry.items():
            output=root/"outputs"/method/dataset
            script="scripts/infer_lfsf.py" if method=="lfsf" else "scripts/infer_lfsf_rf.py"
            command=[sys.executable,script,"--vae-ckpt",artifacts["vq_encoder"]["path"],"--data-root",args.data_root,
                     "--dataset",dataset,"--output-dir",str(output),"--device",args.device]
            if args.overwrite: command.append("--overwrite")
            run(command,log)
            csv_path=per_scene/f"{method}_{dataset.replace(' ','_')}.csv"
            run([sys.executable,"scripts/evaluate_full_reference.py","--pred-dir",str(output),"--gt-dir",str(Path(args.data_root)/dataset/spec["gt_subdir"]),
                 "--scene-list",spec["scene_list"],"--expected-scenes",str(spec["expected_scenes"]),"--dataset",dataset,
                 "--method",method.upper().replace("_","-"),"--output-csv",str(csv_path)],log)
    summary=root/"metrics"/"summary"; run([sys.executable,"scripts/aggregate_metrics.py","--input-dir",str(per_scene),"--output-dir",str(summary)],log)
    manifests=root/"manifests"; manifests.mkdir(exist_ok=True)
    environment={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available()}
    (manifests/"environment.json").write_text(json.dumps(environment,indent=2)+"\n",encoding="utf-8")
    report=["# Reproduction Report","","Status: PASS","",f"Methods: {', '.join(args.methods)}",f"Datasets: {len(registry)}",f"Log: {log}"]
    (root/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")


if __name__=="__main__": main()
