# Inference Repository Repair Report

Date: 2026-07-13

## Completed

- Strict paper checkpoint loading: LFSF 6/6, LFSF-RF 73/73, Direct 72/72.
- VQ module import conflict removed; VQ artifact hash and strict model-state
  loading verified.
- Shared full-stack LFSF forward implemented. Frame chunks affect only VQ
  encoding; masked softmax runs once over the complete stack.
- Shortcut-free one-step RF and Direct endpoint functions use checkpoint gate
  metadata, EMA by default, and clamp `f0 + residual/velocity` to `[-1, 1]`.
- Exact 230-scene registry enforced at 11/15/100/80/24 scenes.
- GT-referenced BT.601 PSNR/SSIM and archived first/last-source extended metric
  protocol separated into explicit evaluators.
- Dataset means, equal-dataset macro, reduced macro, paired bootstrap,
  Wilcoxon, joint 24-test Holm correction, and W/T/L entry points implemented.
- README and docs now describe an inference-only submitted-manuscript release.

## Verification

```text
pytest -q: 15 passed
py_compile: passed
dataset registry: 11 + 15 + 100 + 80 + 24 = 230 scenes
real smoke: Mobile Depth / balls, VQ + LFSF + LFSF-RF = PASS
```

CPU smoke output versus GT:

| Method | PSNR | SSIM | Archived PSNR | Archived SSIM |
|---|---:|---:|---:|---:|
| LFSF | 36.921098 | 0.9749897 | 36.918087 | 0.9751948 |
| LFSF-RF | 33.750152 | 0.9707232 | 33.734718 | 0.9709247 |

The small CPU/environment differences are reported rather than hidden by a
relaxed loader or metric tolerance.

## Release Decision

The executable inference core is repaired and locally smoke-tested. Do not yet
advertise a public/reviewer link until the top-level code license, final GitHub
URL, VQ distribution/access terms, and private golden archive are supplied.
See `docs/BLOCKED_ITEMS.md`.
