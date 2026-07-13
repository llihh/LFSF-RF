# Checkpoints

This directory contains the pretrained model weights for LFSF, LFSF-RF, and Direct (seed 3407).

## Included Checkpoints

| Model | Seed | File | SHA-256 | Size | Role |
|---|---|---|---|---|---|
| LFSF | 3407 | `lfsf_seed3407.pth` | `34310155...` | 157 KB | Primary LFSF results |
| LFSF-RF | 3407 | `lfsf_rf_seed3407.pth` | `44cf3bd8...` | 13 MB | Primary LFSF-RF results |
| Direct | 3407 | `direct_seed3407.pth` | `907523ff...` | 13 MB | Parameter-matched control |

Full SHA-256 checksums are in `checksums.txt`.

## External Checkpoint

The frozen VQ-VAE encoder checkpoint (`autoencoder.ckpt`, ~722 MB) is **not included** in this repository due to size constraints. Download it using:

```bash
python scripts/download_vq_encoder.py
```

Or obtain it from the [RFfusion repository](https://github.com/zirui0625/RFfusion).

## Optional Checkpoints (Seeds 3408, 3409)

For three-seed stability analysis, additional checkpoints for seeds 3408 and 3409 are available from the authors upon request. They are not required for reproducing the main results.

## Usage

```bash
# LFSF inference
python scripts/infer_lfsf.py \
  --checkpoint checkpoints/lfsf_seed3407.pth \
  --vae-ckpt checkpoints/autoencoder.ckpt ...

# LFSF-RF inference
python scripts/infer_lfsf_rf.py \
  --lfsf-checkpoint checkpoints/lfsf_seed3407.pth \
  --rf-checkpoint checkpoints/lfsf_rf_seed3407.pth \
  --vae-ckpt checkpoints/autoencoder.ckpt ...

# Direct robustness control (not a proposed method)
python scripts/infer_direct.py \
  --vae-ckpt checkpoints/autoencoder.ckpt \
  --data-root /path/to/test_datasets --dataset Middlebury \
  --output-dir outputs/direct/Middlebury
```
