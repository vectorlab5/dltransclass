"""Field-attribution utilities for the qualitative analysis in Section 4.7.

Given the learned field weights ``alpha`` and the field summaries ``H``, this
module returns per-record cosine-distance margins between the barycenter and
each field embedding. Useful for librarian-facing inspection, and for the
latent-space plots in Figure 7 of the paper.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def field_pull(b: Tensor, H: Tensor, alpha: Tensor) -> dict[str, Tensor]:
    """Compute per-record, per-field pull statistics.

    Parameters
    ----------
    b : Tensor of shape ``[B, H]`` -- fused barycenter.
    H : Tensor of shape ``[B, M, H]`` -- stacked field summaries.
    alpha : Tensor of shape ``[B, M]`` -- learned field weights.

    Returns
    -------
    dict with keys
        ``cosine_to_field`` : ``[B, M]`` -- cosine similarity between the
            barycenter and each field embedding.
        ``cosine_margin`` : ``[B, M]`` -- 1 - cosine_to_field; larger margin
            means the barycenter sits further from that field.
        ``alpha`` : pass-through of the input weights for convenience.
    """
    b_norm = torch.nn.functional.normalize(b, dim=-1).unsqueeze(1)  # [B, 1, H]
    H_norm = torch.nn.functional.normalize(H, dim=-1)               # [B, M, H]
    cos = (b_norm * H_norm).sum(dim=-1)                              # [B, M]
    return {
        "cosine_to_field": cos,
        "cosine_margin": 1.0 - cos,
        "alpha": alpha,
    }
