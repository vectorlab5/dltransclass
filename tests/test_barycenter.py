"""Tests for the closed-form Wasserstein barycenter."""

import torch

from dltransclass.models.barycenter import MetadataBarycenter


def test_barycenter_reduces_to_weighted_mean() -> None:
    torch.manual_seed(0)
    bary = MetadataBarycenter(hidden_dim=16, attn_dim=8)

    H = {f"f{m}": torch.randn(4, 16) for m in range(4)}
    mask = {name: torch.ones(4, dtype=torch.long) for name in H}

    out = bary(H, mask)
    H_stack = torch.stack([H[k] for k in sorted(H)], dim=1)
    expected = torch.einsum("bm,bmh->bh", out["alpha"], H_stack)

    torch.testing.assert_close(out["b"], expected, atol=1e-6, rtol=1e-6)


def test_barycenter_ignores_missing_fields() -> None:
    torch.manual_seed(1)
    bary = MetadataBarycenter(hidden_dim=8, attn_dim=4)

    H = {f"f{m}": torch.randn(2, 8) for m in range(3)}
    # Drop the second field for the first record.
    mask = {
        "f0": torch.tensor([1, 1]),
        "f1": torch.tensor([0, 1]),
        "f2": torch.tensor([1, 1]),
    }

    out = bary(H, mask)
    # The dropped field should receive zero weight for the first record.
    assert out["alpha"][0, 1].item() == 0.0
    # Weights still sum to 1 over observed fields.
    assert torch.isclose(out["alpha"][0].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(out["alpha"][1].sum(), torch.tensor(1.0), atol=1e-5)
