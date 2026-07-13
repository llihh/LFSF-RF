#!/usr/bin/env python3
"""Aggregate scene CSVs into dataset means and equal-dataset macros."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); files = sorted(Path(args.input_dir).glob("*.csv"))
    if not files: raise RuntimeError("No per-scene CSV files found")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["method", "dataset"]; numeric = [c for c in ("PSNR","SSIM","Qabf","VIF","MI","SF","AG") if c in frame]
    means = frame.groupby(keys, as_index=False)[numeric].mean()
    macro = means.groupby("method", as_index=False)[numeric].mean(); macro.insert(1, "dataset", "Equal-dataset macro")
    reduced = means[means.dataset != "Road-MF"].groupby("method", as_index=False)[numeric].mean(); reduced.insert(1,"dataset","Reduced macro")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    means.to_csv(out / "dataset_means.csv", index=False); macro.to_csv(out / "equal_dataset_macro.csv", index=False)
    reduced.to_csv(out / "reduced_macro.csv", index=False)


if __name__ == "__main__": main()
