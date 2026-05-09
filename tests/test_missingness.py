"""Tests for the realism-calibrated missingness simulator."""

import torch

from dltransclass.data.missingness import RealismCalibratedMask


def test_realistic_average_drop_matches_target() -> None:
    mask = RealismCalibratedMask(target_removal=0.4, dataset="library_sci_meta", seed=0)
    probs = mask.per_field_drop_probabilities()
    avg = sum(probs.values()) / len(probs)
    assert abs(avg - 0.4) < 1e-6


def test_uniform_mode_uses_target_for_every_field() -> None:
    mask = RealismCalibratedMask(
        target_removal=0.6, dataset="udc_oldbooks", mode="uniform"
    )
    probs = mask.per_field_drop_probabilities()
    assert all(abs(p - 0.6) < 1e-9 for p in probs.values())


def test_apply_only_clears_observed_flags() -> None:
    mask = RealismCalibratedMask(
        target_removal=0.5, dataset="udc_oldbooks", seed=42
    )
    observed = {
        "title": torch.ones(64, dtype=torch.long),
        "abstract": torch.ones(64, dtype=torch.long),
        "keywords": torch.ones(64, dtype=torch.long),
        "subject": torch.ones(64, dtype=torch.long),
    }
    masked = mask.apply(observed)
    for name, before in observed.items():
        # Mask can only clear, never set, observation flags.
        assert torch.all(masked[name] <= before)
