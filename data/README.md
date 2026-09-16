# Data layout

Datasets are not redistributed by this repository.

For training, use:

```text
<split>/
  dof_stack/<scene>/<naturally ordered focal frames>
  depth/<scene>.<ext>
  AiF/<scene>.<ext>
```

Configure one or more training sets and one validation set in a copy of
`configs/training.example.yaml`. Depth is required by LFSF training; LFSF-RF
uses only `dof_stack` and `AiF`.

For inference and evaluation, the registered benchmark layout is:

```text
<data-root>/<dataset>/
  image stack/<scene>/<naturally ordered focal frames>
  AiF/<scene>.<ext>
```

`configs/datasets.yaml` and `data/scene_lists/` define the five public
benchmark splits. Stack preparation follows the
[StackMFF-V2 documentation](https://github.com/Xinzhe99/StackMFF-V2#-data-preparation).
