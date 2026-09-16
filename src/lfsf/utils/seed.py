"""Reproducible random seed management."""

import os
import random

import numpy as np
import torch


def set_reproducible_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable cuDNN deterministic mode (may slow training).

    Sets: Python random, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN flags, PYTHONHASHSEED.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
