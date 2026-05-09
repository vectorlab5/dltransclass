"""Training utilities for DLTransClass."""

from dltransclass.training.surrogate import wasserstein_surrogate_loss
from dltransclass.training.trainer import Trainer

__all__ = ["wasserstein_surrogate_loss", "Trainer"]
