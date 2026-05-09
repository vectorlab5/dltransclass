"""Tests for the missingness-aware Wasserstein surrogate loss."""

import torch

from dltransclass.training.surrogate import wasserstein_surrogate_loss


def test_surrogate_reduces_to_ce_when_lambda_zero() -> None:
    torch.manual_seed(0)
    b = torch.randn(8, 16, requires_grad=True)
    logits = torch.nn.functional.linear(b, torch.randn(5, 16))
    targets = torch.randint(0, 5, (8,))
    rho = torch.full((8,), 0.1)

    res = wasserstein_surrogate_loss(logits, targets, b, rho, lambda_var=0.0)
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    torch.testing.assert_close(res["loss"], ce, atol=1e-6, rtol=1e-6)


def test_surrogate_increases_with_radius() -> None:
    torch.manual_seed(1)
    b = torch.randn(8, 16, requires_grad=True)
    logits = torch.nn.functional.linear(b, torch.randn(5, 16))
    targets = torch.randint(0, 5, (8,))

    small_rho = torch.full((8,), 0.05)
    large_rho = torch.full((8,), 0.50)

    small = wasserstein_surrogate_loss(logits, targets, b, small_rho, lambda_var=1.0)
    large = wasserstein_surrogate_loss(logits, targets, b, large_rho, lambda_var=1.0)
    assert large["loss"].item() >= small["loss"].item()


def test_surrogate_requires_grad_on_barycenter() -> None:
    b = torch.randn(2, 4, requires_grad=False)
    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 1])
    rho = torch.full((2,), 0.1)
    try:
        wasserstein_surrogate_loss(logits, targets, b, rho)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when barycenter does not require gradients.")
