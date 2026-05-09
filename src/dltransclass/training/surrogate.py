"""Missingness-aware Wasserstein surrogate loss (Eq. 12 of the paper).

The exact inner supremum of the Wasserstein DRO risk is intractable to solve
during end-to-end training. Following the variation-regularization view of
WDRO, we replace it with a local first-order surrogate

    L_tilde_i = ell_i + lambda * rho_i * || grad_{b_i} ell_i ||_2

where the gradient is taken with respect to the fused barycenter ``b_i``.

We compute the gradient norm with :func:`torch.autograd.grad` so that the
surrogate remains fully differentiable through the encoder and the
barycenter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def wasserstein_surrogate_loss(
    logits: Tensor,
    targets: Tensor,
    barycenter: Tensor,
    rho: Tensor,
    *,
    lambda_var: float = 1.0,
    create_graph: bool = True,
    reduction: str = "mean",
) -> dict[str, Tensor]:
    """Compute the surrogate per-record loss.

    Parameters
    ----------
    logits : Tensor of shape ``[B, C]`` -- model output ``g_phi(b_i)``.
    targets : LongTensor of shape ``[B]`` -- gold labels ``y_i``.
    barycenter : Tensor of shape ``[B, H]`` -- fused representation ``b_i``;
        ``barycenter.requires_grad`` must be True.
    rho : Tensor of shape ``[B]`` -- record-specific radius ``rho_i``.
    lambda_var : explicit scalar coefficient on the variation term.
    create_graph : keep the graph alive so that the gradient-norm term is
        differentiable end-to-end (required during training).
    reduction : ``"mean"``, ``"sum"``, or ``"none"``.

    Returns
    -------
    dict with the keys
        ``loss``       : final scalar (or per-record) surrogate loss
        ``ce``         : per-record cross-entropy
        ``var``        : per-record gradient-norm penalty (already scaled by rho)
    """
    if not barycenter.requires_grad:
        raise ValueError("`barycenter` must require gradients for the surrogate term.")

    ce_per_record = F.cross_entropy(logits, targets, reduction="none")  # [B]
    grads = torch.autograd.grad(
        outputs=ce_per_record.sum(),
        inputs=barycenter,
        create_graph=create_graph,
        retain_graph=True,
    )[0]  # [B, H]
    grad_norm = grads.norm(p=2, dim=-1)  # [B]

    var_term = rho * grad_norm  # [B]
    per_record = ce_per_record + lambda_var * var_term

    if reduction == "mean":
        loss = per_record.mean()
    elif reduction == "sum":
        loss = per_record.sum()
    elif reduction == "none":
        loss = per_record
    else:
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    return {
        "loss": loss,
        "ce": ce_per_record.detach(),
        "var": var_term.detach(),
    }
