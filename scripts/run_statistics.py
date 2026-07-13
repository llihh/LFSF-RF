#!/usr/bin/env python3
"""Reproduce the pre-registered 24-test paired statistical family."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lfsf.metrics.statistics import bootstrap_ci, compute_win_tie_loss, holm_correction, wilcoxon_test


def load_method(path):
    frame = pd.read_csv(path)
    required = {"dataset", "scene", "PSNR", "SSIM"}
    if not required.issubset(frame.columns): raise ValueError(f"{path} lacks {sorted(required-frame.columns)}")
    if frame.duplicated(["dataset","scene"]).any(): raise ValueError(f"Duplicate dataset+scene pairs in {path}")
    return frame


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-dir",required=True)
    args=p.parse_args(); config=yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    datasets=config["datasets"]; metrics=config.get("metrics",["PSNR","SSIM"]); rows=[]
    for comparison in config["comparisons"]:
        proposed,baseline=comparison["proposed"],comparison["baseline"]
        a,b=load_method(config["files"][proposed]),load_method(config["files"][baseline])
        merged=a.merge(b,on=["dataset","scene"],suffixes=("_a","_b"),validate="one_to_one")
        if len(merged)!=len(a) or len(merged)!=len(b): raise RuntimeError(f"Pairing mismatch: {proposed} vs {baseline}")
        for metric in metrics:
            dataset_means=[]
            for dataset in datasets:
                selected=merged[merged.dataset==dataset]; va=selected[f"{metric}_a"].to_numpy(); vb=selected[f"{metric}_b"].to_numpy()
                if not len(va): raise RuntimeError(f"No pairs for {dataset}")
                boot=bootstrap_ci(va,vb,n_bootstrap=10000,seed=3407); wilc=wilcoxon_test(va,vb); wtl=compute_win_tie_loss(va,vb,1e-12)
                rows.append({"proposed":proposed,"baseline":baseline,"dataset":dataset,"metric":metric,**boot,
                             "wilcoxon_raw_p":wilc["p_value"],**wtl})
                dataset_means.append((va.mean(),vb.mean()))
            va=np.array([x[0] for x in dataset_means]); vb=np.array([x[1] for x in dataset_means])
            boot=bootstrap_ci(va,vb,n_bootstrap=10000,seed=3407); wilc=wilcoxon_test(va,vb); wtl=compute_win_tie_loss(va,vb,1e-12)
            rows.append({"proposed":proposed,"baseline":baseline,"dataset":"Equal-dataset macro","metric":metric,**boot,
                         "wilcoxon_raw_p":wilc["p_value"],**wtl})
    if len(rows)!=24: raise RuntimeError(f"Expected 24 pre-registered Wilcoxon tests, got {len(rows)}")
    corrected=holm_correction([row["wilcoxon_raw_p"] for row in rows])
    for row,value in zip(rows,corrected): row["wilcoxon_holm_p"]=value
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/"table4_paired_results.csv",index=False)
    (out/"holm_family.json").write_text(json.dumps({"family_size":24,"tests":rows},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
