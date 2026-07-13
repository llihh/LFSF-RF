# Data Directory

This directory documents the expected dataset structure for inference and evaluation.

The benchmark preparation conventions and released stack resources are linked
from the [StackMFF-V2 Data Preparation section](https://github.com/Xinzhe99/StackMFF-V2#-data-preparation).
Please follow the upstream dataset terms and cite the corresponding StackMFF
work when using those resources.

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

## Important Notes

- Datasets are **not included** in this repository due to licensing and size constraints
- NYU Depth V2 is retained only as a source-overlap diagnostic and is
  **excluded from independent benchmark averages**
- The five independent benchmarks (FlyingThings3D, Middlebury, Road-MF, 4D-Light-Field, Mobile Depth) comprise 230 unique scenes
