"""DLTransClass model: shared encoder + barycenter fusion + classifier head.

This module wires together :class:`SharedFieldEncoder` and
:class:`MetadataBarycenter` and exposes a classifier head as well as the
auxiliary tensors required by the missingness-aware Wasserstein surrogate
loss (the radius ``rho_i`` is computed in the training module to keep this
class loss-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from dltransclass.models.barycenter import MetadataBarycenter
from dltransclass.models.field_encoder import FieldBudget, SharedFieldEncoder


@dataclass
class DLTransClassConfig:
    """Hyperparameters for the model. Defaults match the paper."""

    backbone: str = "bert-base-uncased"
    num_classes: int = 18
    attn_dim: int = 128
    rho0: float = 0.08
    beta_m: float = 0.6
    beta_v: float = 0.35
    lambda_var: float = 1.0
    weight_decay: float = 0.01
    pooling: str = "mean"
    budgets: FieldBudget = FieldBudget()


class DLTransClass(nn.Module):
    """End-to-end DLTransClass classifier.

    The forward pass returns a dictionary that includes the logits, the fused
    barycenter ``b``, the field weights ``alpha``, the cross-field dispersion
    ``omega``, and the per-record radius ``rho`` (Eq. 10 of the paper). The
    ``training/surrogate.py`` module consumes this dictionary to compute the
    Wasserstein surrogate loss.
    """

    def __init__(self, cfg: DLTransClassConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedFieldEncoder(
            backbone=cfg.backbone,
            budgets=cfg.budgets,
            pooling=cfg.pooling,
        )
        self.barycenter = MetadataBarycenter(
            hidden_dim=self.encoder.hidden_size,
            attn_dim=cfg.attn_dim,
        )
        self.classifier = nn.Linear(self.encoder.hidden_size, cfg.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        fields: Mapping[str, list[str]],
        observed_mask: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Run the full pipeline.

        Parameters
        ----------
        fields : mapping of field_name -> list of strings (one per record)
        observed_mask : mapping of field_name -> 0/1 tensor of shape [B]
        """
        summaries = self.encoder(fields)
        bary = self.barycenter(summaries, observed_mask)
        logits = self.classifier(bary["b"])

        # Eq. 10: record-specific Wasserstein radius driven jointly by
        # missingness and cross-field disagreement.
        observed = torch.stack([observed_mask[n] for n in sorted(observed_mask)], dim=1).float()
        n_obs = observed.sum(dim=-1)
        n_total = float(observed.shape[-1])
        miss_frac = (n_total - n_obs) / max(n_total, 1.0)
        rho = self.cfg.rho0 * (
            1.0
            + self.cfg.beta_m * miss_frac
            + self.cfg.beta_v * bary["omega"].sqrt()
        )

        return {
            "logits": logits,
            "b": bary["b"],
            "alpha": bary["alpha"],
            "omega": bary["omega"],
            "rho": rho,
        }
