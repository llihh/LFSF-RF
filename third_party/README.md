# Third-Party Code Attribution

This project includes minimal subsets of the following third-party codebases.
All original copyright notices are preserved in the source files.

| Component | Source | License | Modified | Location |
|---|---|---|---|---|
| VQModel autoencoder | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | MIT | Extracted (LPIPS loss bypassed) | `src/lfsf/third_party/ldm/` |
| VectorQuantizer2 | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | MIT | None | `src/lfsf/third_party/taming/` |
| VQ-VAE checkpoint | [RFfusion](https://github.com/zirui0625/RFfusion) (NeurIPS 2025) | See RFfusion | Not included; external download | `checkpoints/autoencoder.ckpt` |

## LDM (Latent Diffusion Models)

- **Repository**: https://github.com/CompVis/latent-diffusion
- **License**: MIT
- **Files used**: `autoencoder.py`, `diffusionmodules/model.py`, `attention.py`, `distributions/distributions.py`, `util.py`
- **Modifications**: The LPIPS perceptual loss and discriminator are bypassed (`lossconfig` replaced with `torch.nn.Identity`) to avoid requiring VGG weights during VAE loading. Only the encoder/decoder weights are loaded from the checkpoint.

## Taming Transformers

- **Repository**: https://github.com/CompVis/taming-transformers
- **License**: MIT
- **Files used**: `modules/vqvae/quantize.py` (VectorQuantizer2 only)
- **Modifications**: None

## RFfusion

- **Paper**: "Efficient Rectified Flow for Image Fusion" (NeurIPS 2025)
- **Repository**: https://github.com/zirui0625/RFfusion
- **Usage**: The frozen VQ-VAE autoencoder from RFfusion is used as the feature extractor for LFSF. The checkpoint must be obtained from the RFfusion repository.

## Other Dependencies

Additional Python packages are listed in `requirements.txt`. These are standard open-source libraries (PyTorch, NumPy, SciPy, scikit-image, etc.) and are not bundled with this repository.
