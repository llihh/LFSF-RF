#!/usr/bin/env python3
"""Download the frozen VQ encoder checkpoint.

The VQ-VAE autoencoder checkpoint (~722 MB) from RFfusion (NeurIPS 2025)
must be obtained separately due to its size. This script provides clear
instructions for manual download.

Usage:
    python scripts/download_vq_encoder.py [--output checkpoints/autoencoder.ckpt]
"""

import argparse
import hashlib
import sys
from pathlib import Path


EXPECTED_SHA256 = "aacf13951f4b18f5af9b47febdc696cf9559305d6de0821084abeaf342439251"

DOWNLOAD_INSTRUCTIONS = """
================================================================================
VQ Encoder Checkpoint Required
================================================================================

LFSF requires a frozen VQ-VAE autoencoder from the RFfusion project
(NeurIPS 2025). The checkpoint is ~722 MB and must be obtained separately.

To obtain the checkpoint:

1. Download from the official RFfusion release page:
   https://github.com/zirui0625/RFfusion

2. Copy the file to the checkpoints directory:
   cp /path/to/downloaded/autoencoder.ckpt checkpoints/autoencoder.ckpt

3. Verify the checksum:
   sha256sum checkpoints/autoencoder.ckpt

Expected location: checkpoints/autoencoder.ckpt

If you already have the file, place it at the expected location and re-run.
================================================================================
"""


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download VQ encoder checkpoint")
    parser.add_argument("--output", default="checkpoints/autoencoder.ckpt", help="Output path")
    parser.add_argument("--from-local", help="Copy the artifact from a local reviewer archive")
    args = parser.parse_args()

    output = Path(args.output)
    if args.from_local:
        source = Path(args.from_local)
        if not source.is_file():
            raise FileNotFoundError(source)
        if compute_sha256(source) != EXPECTED_SHA256:
            raise RuntimeError("Local VQ artifact SHA-256 does not match the paper artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(source, output)
    if output.exists():
        sha = compute_sha256(output)
        print(f"Checkpoint already exists: {output}")
        print(f"SHA-256: {sha}")
        if sha != EXPECTED_SHA256:
            raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha}")
        return

    print(DOWNLOAD_INSTRUCTIONS)
    print(f"\nExpected location: {output.absolute()}")
    print("Please download the checkpoint manually and run this script again to verify.")
    sys.exit(1)


if __name__ == "__main__":
    main()
