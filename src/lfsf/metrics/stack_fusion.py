"""Paper extended metrics for focus stacks.

The archived Table 3 evaluator used the naturally sorted first and last focal
planes as the two source references for Qabf, VIF, and MI. SF and AG use only
the fused image. This is a frozen protocol, not an average over all N frames.
"""

from __future__ import annotations

import numpy as np

from .fusion_metrics import AG_metric, MI_metric, Qabf_metric, SF_metric, VIF_metric


def evaluate_stack_fusion(source_stack, fused):
    sources = np.asarray(source_stack)
    if sources.ndim != 3 or sources.shape[0] < 2:
        raise ValueError("source_stack must be [N,H,W] with N>=2")
    fused = np.asarray(fused)
    if sources.shape[1:] != fused.shape:
        raise ValueError("Source and fused image shapes differ")
    first, last = sources[0], sources[-1]
    return {
        "Qabf": Qabf_metric(first, last, fused),
        "VIF": VIF_metric(first, last, fused),
        "MI": MI_metric(first, last, fused),
        "SF": SF_metric(fused),
        "AG": AG_metric(fused),
    }
