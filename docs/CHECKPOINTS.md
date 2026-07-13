# Checkpoint Documentation

## Paper Checkpoints (Seed 3407)

These are the frozen seed-3407 artifacts used by the inference release.

| Model | File | Params | SHA-256 (first 8) | Source |
|---|---|---|---|---|
| LFSF | `checkpoints/lfsf_seed3407.pth` | 39,361 | `34310155` | Training on NYU+DIODE+DUTS |
| LFSF-RF | `checkpoints/lfsf_rf_seed3407.pth` | 838,275 | `44cf3bd8` | C3 training anchored on LFSF |
| Direct | `checkpoints/direct_seed3407.pth` | 838,275 | `907523ff` | Direct residual training |

Full SHA-256 hashes are recorded in `checkpoints/checksums.txt`.

## VQ Encoder Checkpoint

| Model | File | Size | Source |
|---|---|---|---|
| VQ-VAE | `checkpoints/autoencoder.ckpt` | ~722 MB | RFfusion (NeurIPS 2025) |

The VQ encoder is **frozen** and never trained in LFSF. It must be obtained from the RFfusion project. Use:

```bash
python scripts/download_vq_encoder.py
```

## Optional: Multi-Seed Checkpoints (3408, 3409)

Three random seeds (3407, 3408, 3409) were used for stability analysis.
Seeds 3408 and 3409 checkpoints are available upon request and are not
required for reproducing the main results.

| Seed | LFSF PSNR | LFSF-RF PSNR | Delta |
|---|---|---|---|
| 3407 | 34.20 | 36.37 | +2.17 |
| 3408 | 34.21 | 36.28 | +2.07 |
| 3409 | 34.19 | 36.21 | +2.02 |
| **Mean±SD** | **34.20±0.01** | **36.29±0.08** | **+2.09±0.08** |

## Checkpoint Loading

All checkpoints are standard PyTorch `.pth` files saved with `torch.save()`.
LFSF is loaded only from `focus_scorer`. RF/Direct default to final EMA
`velocity_model`, with `online_model` available only through the explicit
`--weight-source online` diagnostic option. `load_lfsf_checkpoint()` and
`load_refiner_checkpoint()` require exact keys, shapes, model type, and parameter
count; no automatic key filtering is performed.

## Verification

```bash
sha256sum -c checkpoints/checksums.txt
```
