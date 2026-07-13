# Evaluation Metrics

## PSNR and SSIM

Input is the saved uint8 RGB prediction and all-in-focus GT. Both are converted
to floating-point luminance with BT.601 (`0.299R + 0.587G + 0.114B`). Evaluation
uses the whole canonical-resolution image, `data_range=255`, no border crop,
and no implicit resize. PSNR is standard MSE PSNR. SSIM uses scikit-image's
standard full-reference implementation, matching the paper-producing script.

## Qabf, VIF, and MI

The paper's archived Table 3 protocol naturally sorts all frames but uses the
first and last focal planes as the two source references. Qabf follows
Xydeas-Petrovic edge preservation; VIF is the mean of source-to-fused VIF; MI
is the sum of source-to-fused mutual information. This is a fixed two-reference
projection of an N-frame stack, not an average over every frame.

## SF and AG

Spatial Frequency and Average Gradient are no-reference sharpness measures and
use only the fused uint8 grayscale image.

All evaluators fail on missing scenes or resolution mismatch. Dataset values
are scene means. The equal-dataset macro is the unweighted mean of the five
dataset means; the reduced macro excludes Road-MF.
