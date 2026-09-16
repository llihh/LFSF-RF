# Evaluation Metrics

## Scope of the public evaluator

`scripts/evaluate.py` evaluates saved LFSF/LFSF-RF outputs and all-in-focus GT.
Inputs are read at 8-bit precision and converted to floating-point BT.601
luminance (`0.299R + 0.587G + 0.114B`). The proposed-model ablation and
robustness evaluator used the same 8-bit/BT.601 convention, but that evaluator
and its evidence are distributed separately rather than in this main-method
source repository.

The main historical comparison with external baselines used a separate
independent benchmark evaluator. External methods can differ in saved-output
format and canonical-resolution handling, so `scripts/evaluate.py` must not be
presented as sufficient to reconstruct all historical baseline tables. Those
tables also depend on external implementations, released weights, and the
separate reproducibility/evidence package.

## PSNR and SSIM

Evaluation uses the whole same-resolution image, `data_range=255`, no border
crop, and no implicit resize. PSNR is standard MSE PSNR. SSIM uses
scikit-image's standard full-reference implementation.

## Qabf, VIF, and MI

The paper's archived Table 3 protocol naturally sorts all frames but uses the
first and last focal planes as the two source references. Qabf follows
Xydeas-Petrovic edge preservation; VIF is the mean of source-to-fused VIF; MI
is the sum of source-to-fused mutual information. This is a fixed two-reference
projection of an N-frame stack, not an average over every frame.

## SF and AG

Spatial Frequency and Average Gradient are no-reference sharpness measures and
use only the fused uint8 grayscale image.

The public evaluators fail on missing scenes or resolution mismatch and emit
per-scene values. Aggregate statistics and paper-table assembly are outside
this repository and belong to the separate reproducibility/evidence package.
