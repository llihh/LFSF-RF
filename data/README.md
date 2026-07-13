# Data Directory

This directory documents the expected dataset structure for training and evaluation.

## Expected Structure

Place benchmark datasets under a data root directory (e.g., `/path/to/test_datasets/`) with the following layout:

```
{data_root}/
├── FlyingThings3D/
│   ├── AiF/           # All-in-Focus ground truth images
│   │   ├── 0000000.png
│   │   ├── 0000010.png
│   │   └── ...
│   ├── image stack/    # Multi-focus stacks
│   │   ├── 0000000/
│   │   │   ├── 0.png
│   │   │   ├── 10.0.png
│   │   │   └── ...
│   │   └── ...
│   └── depth/          # (optional) Depth maps
├── Middlebury/
│   ├── AiF/
│   ├── image stack/
│   └── depth/
├── Road-MF/
│   ├── AiF/
│   └── image stack/
├── 4D-Light-Field/
│   ├── AiF/
│   └── image stack/
├── Mobile Depth/
│   ├── AiF/
│   └── image stack/
└── NYU Depth V2/       # Diagnostic only (not an independent benchmark)
    ├── AiF/
    └── image stack/
```

## Scene Lists

Pre-generated scene lists for each benchmark are provided in `scene_lists/`.

## Training Data

Training requires synthetic focal stacks generated from depth-annotated datasets (NYU Depth V2, DIODE, DUTS-TR). See `docs/DATASETS.md` for acquisition and preprocessing instructions.

## Important Notes

- Datasets are **not included** in this repository due to licensing and size constraints
- NYU Depth V2 is a training/validation source and is **excluded from independent benchmark averages**
- The five independent benchmarks (FlyingThings3D, Middlebury, Road-MF, 4D-Light-Field, Mobile Depth) comprise 230 unique scenes
