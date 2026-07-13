# Inference Reproduction

This repository reproduces inference and evaluation from frozen artifacts; it
does not include training or synthetic-stack generation.

## Environment and Artifacts

Install with the commands in the root README. Place the VQ artifact at
`checkpoints/autoencoder.ckpt` and run `python scripts/download_vq_encoder.py`
to verify its SHA-256. All model hashes are listed in
`configs/artifacts.example.yaml` and `checkpoints/checksums.txt`.

## Data

Place each dataset under one root using `<dataset>/image stack/<scene>/*` and
`<dataset>/AiF/<scene>.<ext>`. `configs/datasets.yaml` and the committed scene
lists define the exact 11/15/100/80/24-scene protocol. Missing, duplicate, or
single-frame scenes are fatal errors.

## Single Dataset

Run the two commands in the root README. `--frame-chunk` changes only VQ encoder
batching; it never changes the full-stack softmax. `--weight-source online` is
available for diagnostics, while paper reproduction defaults to `ema`.

Full-reference evaluation:

```bash
python scripts/evaluate_full_reference.py \
  --pred-dir outputs/lfsf/FlyingThings3D \
  --gt-dir /data/FlyingThings3D/AiF \
  --scene-list data/scene_lists/flyingthings3d.txt \
  --expected-scenes 100 --dataset FlyingThings3D --method LFSF \
  --output-csv results/per_scene/lfsf_FlyingThings3D.csv
```

Use `scripts/evaluate_stack_metrics.py` for Qabf/VIF/MI/SF/AG and
`scripts/aggregate_metrics.py` for dataset means and equal-dataset macro.

## Expected Behavior

Checkpoint logs must report LFSF 6/6, RF 73/73, and Direct 72/72 tensors with
zero missing/unexpected keys. Outputs are rounded to uint8 after clipping and
resized to each scene's canonical AiF resolution. A run is incomplete if any
official scene is absent.

## Common Errors

- Hash mismatch: obtain the exact reviewer artifact; do not bypass the check.
- Resolution: `image_size=256` in checkpoint metadata is the training crop.
  Paper benchmark inference preserves native resolution; `--image-size` is a
  non-paper diagnostic override.
- CUDA OOM: lower `--frame-chunk`; results should remain invariant.
- Metric shape mismatch: predictions must already match canonical GT/source
  resolution; evaluators do not silently resize or crop.
