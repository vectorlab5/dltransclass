"""Shared transformer encoder applied to each metadata field.

The encoder is a single BERT-base instance that is shared across the four
bibliographic fields (title, abstract, keywords, subject). Per-field token
budgets are honoured at the tokenization layer; the encoder produces token
states that are mean-pooled into a single Dirac mass per observed field.

See Section 3.1 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class FieldBudget:
    """Per-field token budget calibrated to the empirical length distribution."""

    title: int = 64
    abstract: int = 256
    keywords: int = 48
    subject: int = 48

    def as_dict(self) -> Mapping[str, int]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "subject": self.subject,
        }


class SharedFieldEncoder(nn.Module):
    """Shared BERT-style encoder applied independently to each observed field.

    Parameters
    ----------
    backbone : str
        HuggingFace model identifier (default ``bert-base-uncased``).
    budgets : FieldBudget
        Per-field token budgets used for tail truncation.
    pooling : {"mean", "cls"}
        Strategy for producing the field summary ``h_i^{(m)}``. Mean pooling is
        the default and matches the manuscript description.
    """

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        budgets: FieldBudget = FieldBudget(),
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.budgets = budgets
        self.pooling = pooling
        self.hidden_size: int = self.backbone.config.hidden_size

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def encode_field(self, texts: list[str], field_name: str) -> Tensor:
        """Encode a batch of strings for a single field.

        Returns the field summary ``h_i^{(m)}`` of shape ``[batch, hidden]``.
        """
        max_length = self.budgets.as_dict()[field_name]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(next(self.backbone.parameters()).device)

        out = self.backbone(**enc).last_hidden_state  # [B, L, H]
        mask = enc["attention_mask"].unsqueeze(-1)    # [B, L, 1]

        if self.pooling == "cls":
            return out[:, 0]
        # Mean pooling over valid tokens (Eq. 2 in the paper).
        denom = mask.sum(dim=1).clamp_min(1)
        return (out * mask).sum(dim=1) / denom

    def forward(self, fields: Mapping[str, list[str]]) -> dict[str, Tensor]:
        """Encode all observed fields. Missing fields are skipped silently."""
        return {name: self.encode_field(texts, name) for name, texts in fields.items()}
