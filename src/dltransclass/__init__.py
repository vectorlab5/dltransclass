"""DLTransClass: Metadata-Aware Transformer Classification for Digital Libraries.

Public surface re-exports the model class, the training entry point, and the
realism-calibrated missingness simulator.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dltransclass")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

from dltransclass.models.dltransclass import DLTransClass
from dltransclass.training.surrogate import wasserstein_surrogate_loss
from dltransclass.data.missingness import RealismCalibratedMask

__all__ = [
    "DLTransClass",
    "wasserstein_surrogate_loss",
    "RealismCalibratedMask",
    "__version__",
]
