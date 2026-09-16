# Third-Party Code Attribution

This project includes minimal subsets of the following third-party codebases.
All original copyright notices are preserved in the source files.

| Component | Source | License | Modified | Location |
|---|---|---|---|---|
| VQModel autoencoder | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | [MIT](LICENSE_latent-diffusion.txt) | Extracted (LPIPS loss bypassed) | `src/lfsf/third_party/ldm/` |
| VectorQuantizer2 | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | [MIT](LICENSE_taming-transformers.txt) | None | `src/lfsf/third_party/taming/` |
| VQ-VAE checkpoint | [RFfusion](https://github.com/zirui0625/RFfusion) ([download](https://drive.google.com/file/d/10Rmz6YtGnM2qHk1QfjCY9eEFkh0gsvVZ/view?usp=drive_link)) | See RFfusion | Not included; external download | `checkpoints/autoencoder.ckpt` |
| Benchmark preparation | [StackMFF-V2](https://github.com/Xinzhe99/StackMFF-V2#-data-preparation) | See upstream datasets | Referenced, not bundled | `data/scene_lists/` |

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
- **Checkpoint**: https://drive.google.com/file/d/10Rmz6YtGnM2qHk1QfjCY9eEFkh0gsvVZ/view?usp=drive_link
- **Expected SHA-256**: `aacf13951f4b18f5af9b47febdc696cf9559305d6de0821084abeaf342439251`

## StackMFF-V2

- **Repository**: https://github.com/Xinzhe99/StackMFF-V2
- **Data preparation**: https://github.com/Xinzhe99/StackMFF-V2#-data-preparation
- **Usage**: Referenced for compatible benchmark preparation conventions and
  released multi-focus stack test resources. No StackMFF model implementation
  is vendored into this repository.

## Other Dependencies

Additional Python packages are listed in `requirements.txt`. These are standard open-source libraries (PyTorch, NumPy, SciPy, scikit-image, etc.) and are not bundled with this repository.
