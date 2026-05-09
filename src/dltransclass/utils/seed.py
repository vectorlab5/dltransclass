"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 0, deterministic: bool | None = None) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and all GPUs).

    Set ``DLT_DETERMINISTIC=1`` in the environment, or pass
    ``deterministic=True``, to enable cuDNN deterministic mode at the cost of
    a small throughput penalty.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic is None:
        deterministic = os.environ.get("DLT_DETERMINISTIC", "0") == "1"
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
