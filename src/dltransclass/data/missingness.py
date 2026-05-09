"""Realism-calibrated missingness simulator.

Implements the masking scheme described in Section 4.7 of the paper. Per-field
drop probabilities are anchored to the empirical missingness pattern of each
catalog (Title 0%, Abstract 5--19%, Keywords/Subject 8--62%) and then scaled
proportionally so that the global removal ratio reaches a target value
(typically 20%, 40%, or 60%).

A pure uniform-random fallback is also exposed for sanity checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

# Empirical observation rates from Table 2; these are *observation* rates
# (1 - missing_rate). Override by passing ``observation_rates`` explicitly.
DEFAULT_OBSERVATION_RATES: dict[str, dict[str, float]] = {
    "udc_oldbooks":      {"title": 1.00, "abstract": 0.72, "keywords": 0.41, "subject": 0.41},
    "lcsh_theses":       {"title": 1.00, "abstract": 0.83, "keywords": 0.38, "subject": 0.38},
    "jucs_meta":         {"title": 1.00, "abstract": 0.91, "keywords": 0.77, "subject": 0.77},
    "compscholar_meta":  {"title": 1.00, "abstract": 0.95, "keywords": 0.81, "subject": 0.81},
    "arxiv_cs_metadata": {"title": 1.00, "abstract": 0.98, "keywords": 0.92, "subject": 0.92},
    "library_sci_meta":  {"title": 1.00, "abstract": 0.81, "keywords": 0.45, "subject": 0.45},
}


@dataclass
class RealismCalibratedMask:
    """Realism-calibrated missingness simulator.

    Parameters
    ----------
    target_removal : float in [0, 1]
        Desired global field-removal ratio (averaged across fields).
    observation_rates : mapping of field_name -> baseline observation rate.
        If omitted, the dataset id is used to look up
        :data:`DEFAULT_OBSERVATION_RATES`.
    dataset : optional dataset id, used when ``observation_rates`` is None.
    seed : RNG seed for reproducibility.
    mode : ``"realistic"`` (default) or ``"uniform"`` (sanity-check baseline).
    """

    target_removal: float
    observation_rates: Mapping[str, float] | None = None
    dataset: str | None = None
    seed: int = 0
    mode: str = "realistic"

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_removal <= 1.0:
            raise ValueError("target_removal must lie in [0, 1].")
        if self.observation_rates is None:
            if self.dataset is None or self.dataset not in DEFAULT_OBSERVATION_RATES:
                raise ValueError(
                    "Provide either `observation_rates` or a known `dataset` id."
                )
            self.observation_rates = DEFAULT_OBSERVATION_RATES[self.dataset]
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def per_field_drop_probabilities(self) -> dict[str, float]:
        """Compute per-field drop probabilities matching the global target."""
        if self.mode == "uniform":
            return {name: self.target_removal for name in self.observation_rates}

        # Baseline missingness = 1 - observation rate.
        base = {name: 1.0 - rate for name, rate in self.observation_rates.items()}
        mean_base = float(np.mean(list(base.values())))
        if mean_base == 0.0:
            return {name: self.target_removal for name in base}

        # Scale proportionally so the average drop probability hits the target.
        scale = self.target_removal / mean_base
        probs = {name: float(np.clip(p * scale, 0.0, 1.0)) for name, p in base.items()}
        return probs

    # ------------------------------------------------------------------
    def apply(self, observed: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Apply the mask in-place to a batch of observation flags."""
        probs = self.per_field_drop_probabilities()
        out: dict[str, Tensor] = {}
        for name, flag in observed.items():
            drop = self._rng.random(size=tuple(flag.shape)) < probs.get(name, 0.0)
            new = flag.clone()
            new[torch.from_numpy(drop)] = 0
            out[name] = new
        return out
