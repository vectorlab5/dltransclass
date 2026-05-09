"""Standard classification metrics (Macro-F1 is the headline measure)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }
