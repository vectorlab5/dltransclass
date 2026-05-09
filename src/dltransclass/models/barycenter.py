"""Wasserstein metadata barycenter (Eqs. 5–9 of the paper).

Under the Dirac-field construction, the closed-form 2-Wasserstein barycenter of
``M`` Dirac masses ``delta_{h^{(m)}}`` weighted by ``alpha^{(m)}`` reduces to
the weighted Euclidean mean. The implementation also exposes the cross-field
dispersion ``Omega_i`` used to drive the record-specific Wasserstein radius.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class MetadataBarycenter(nn.Module):
    """Field-weighted Wasserstein barycenter of Dirac field embeddings.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the field summaries ``h^{(m)}``.
    attn_dim : int
        Hidden dimension of the additive scoring MLP.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.u = nn.Parameter(torch.empty(attn_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.normal_(self.u, std=0.02)

    @staticmethod
    def _stack_observed(
        field_summaries: Mapping[str, Tensor],
        observed_mask: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Stack per-field summaries into ``[batch, M, H]`` and an observed mask."""
        names = sorted(field_summaries.keys())
        H = torch.stack([field_summaries[n] for n in names], dim=1)
        M = torch.stack([observed_mask[n] for n in names], dim=1).to(H.dtype)
        return H, M

    def forward(
        self,
        field_summaries: Mapping[str, Tensor],
        observed_mask: Mapping[str, Tensor],
        eps: float = 1e-6,
    ) -> dict[str, Tensor]:
        """Compute barycenter, field weights, and cross-field dispersion.

        Returns
        -------
        dict with keys
            ``b``       : barycenter ``[B, H]``         (Eq. 8)
            ``alpha``   : field weights ``[B, M]``      (Eq. 6)
            ``omega``   : cross-field dispersion ``[B]`` (Eq. 9)
            ``H``       : stacked field summaries ``[B, M, H]``
            ``mask``    : observed-field mask ``[B, M]``
        """
        H, mask = self._stack_observed(field_summaries, observed_mask)  # [B, M, H], [B, M]

        # Eq. 5: additive field score s = u^T tanh(W h + b)
        scores = (torch.tanh(self.W(H)) * self.u).sum(dim=-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Eq. 6: softmax restricted to observed fields
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.where(mask.bool(), alpha, torch.zeros_like(alpha))

        # Eq. 8: closed-form barycenter b = sum_m alpha_m h_m
        b = torch.einsum("bm,bmh->bh", alpha, H)

        # Eq. 9: cross-field dispersion Omega
        diff = H - b.unsqueeze(1)
        omega = (alpha * (diff.pow(2).sum(dim=-1))).sum(dim=-1).clamp_min(eps)

        return {"b": b, "alpha": alpha, "omega": omega, "H": H, "mask": mask}
