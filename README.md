# LFSF: Latent-Focus Soft Fusion

Inference and evaluation code accompanying the submitted manuscript
"Latent-Focus Soft Fusion: Decoupling Focus Reasoning from Source-Pixel
Synthesis for Multi-Focus Image Stacks".

This release supports **inference-only reproduction** with frozen checkpoints.
Training and synthetic-stack generation are outside the scope of this release.

LFSF uses a frozen VQ encoder and a 39,361-parameter shared focus scorer. One
masked softmax is applied across the complete, naturally sorted focus stack,
and observed RGB pixels are fused directly. LFSF-RF adds a frozen 838,275-
parameter, one-step shortcut-free rectified-flow refiner. Direct is a matched
control, not a proposed method.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Python 3.10+, PyTorch 2.0+, and an NVIDIA CUDA environment are recommended.

## Artifacts

The three paper model checkpoints are under `checkpoints/`. The frozen VQ
artifact is distributed separately by the official
[RFfusion repository](https://github.com/zirui0625/RFfusion). Download
[`autoencoder.ckpt` from the RFfusion Google Drive link](https://drive.google.com/file/d/10Rmz6YtGnM2qHk1QfjCY9eEFkh0gsvVZ/view?usp=drive_link),
place it at `checkpoints/autoencoder.ckpt`, and verify it with:

```bash
python scripts/download_vq_encoder.py
```

The expected SHA-256 is
`aacf13951f4b18f5af9b47febdc696cf9559305d6de0821084abeaf342439251`.
The script also supports `--from-local` for a reviewer-archive copy. Expected
hashes and local-path examples are in `configs/artifacts.example.yaml`.

## Inference

```bash
python scripts/infer_lfsf.py \
  --vae-ckpt checkpoints/autoencoder.ckpt \
  --data-root /path/to/test_datasets \
  --dataset FlyingThings3D \
  --output-dir outputs/lfsf/FlyingThings3D

python scripts/infer_lfsf_rf.py \
  --vae-ckpt checkpoints/autoencoder.ckpt \
  --data-root /path/to/test_datasets \
  --dataset FlyingThings3D \
  --output-dir outputs/lfsf_rf/FlyingThings3D
```

Both commands enforce the official scene list, strict checkpoint loading,
native-resolution paper inference, uniform sampling to at most 24 frames, and
canonical GT output resolution. The checkpoint's `image_size=256` records the
training crop size; it is not the paper benchmark inference resize.
LFSF-RF defaults to the final EMA (`velocity_model`) weights.

## Evaluation Scope

- LFSF/LFSF-RF inference on 5 independent benchmarks (230 scenes).
- Proposed rows in Tables 1/2 (GT-referenced PSNR/SSIM).
- Proposed rows in Table 3 (frozen first/last-source fusion protocol).
- Table 4 statistics when matching baseline per-scene artifacts are supplied.

Training, Table 5 ablations, Table 6 profiling, and three-seed training are not
claimed as executable parts of this inference-only release. Precomputed CSVs
are evidence artifacts, never inputs to a fresh metric run.

See [docs/INFERENCE_REPRODUCTION.md](docs/INFERENCE_REPRODUCTION.md),
[docs/METRICS.md](docs/METRICS.md), [docs/DATASETS.md](docs/DATASETS.md), and
[docs/BLOCKED_ITEMS.md](docs/BLOCKED_ITEMS.md).

## Citation

The manuscript is submitted and is not represented here as an accepted or
published Neurocomputing article. Author metadata is in `CITATION.cff`.

## Upstream Resources and Acknowledgements

- [RFfusion](https://github.com/zirui0625/RFfusion): source of the frozen
  VQ-VAE architecture and pretrained `autoencoder.ckpt` used as LFSF's fixed
  feature encoder. Please also cite *Efficient Rectified Flow for Image Fusion*
  when using this artifact.
- [StackMFF-V2](https://github.com/Xinzhe99/StackMFF-V2#-data-preparation):
  source repository for the benchmark preparation conventions and released
  multi-focus stack test resources used by this evaluation protocol. Please
  follow its dataset terms and cite the corresponding StackMFF work.
- [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion) and
  [taming-transformers](https://github.com/CompVis/taming-transformers): source
  of the minimal VQ implementation vendored under `src/lfsf/third_party/`.

These projects are upstream resources and are not claimed as part of the LFSF
authors' original implementation. See `third_party/README.md` for details.

## License

The authors' code license is pending final author approval. Bundled third-party
components retain their own terms; see `third_party/README.md`. The VQ weights
are external and governed by their source release.
