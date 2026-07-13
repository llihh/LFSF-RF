# Datasets

The independent protocol contains exactly 230 scenes: Mobile Depth 11,
Middlebury 15, FlyingThings3D 100, Road-MF 80, and 4D-Light-Field 24. Exact
scene names are committed under `data/scene_lists/` and are authoritative.

Expected local layout:

```text
test_datasets/
  <Dataset>/
    image stack/<scene>/<naturally-sorted focal frames>
    AiF/<scene>.<png|jpg|...>
```

`configs/datasets.yaml` records each subdirectory and expected count. Before
inference, every listed scene, all-in-focus GT, and at least two readable frame
paths are required. Frames above the frozen limit of 24 are sampled uniformly
while preserving their natural order. Missing scenes are never skipped.

The datasets are not redistributed by this repository. Obtain FlyingThings3D,
Middlebury, Road-MF, 4D-Light-Field, and Mobile Depth from their official or
author-authorized sources and follow their licenses. Reviewer-only packaging
may provide auxiliary scene mappings or predictions under submission-system
access terms; that does not grant public redistribution rights.

NYU Depth V2 is not part of the five-dataset independent macro. Training data
and synthetic-stack generation are outside this inference-only release.
