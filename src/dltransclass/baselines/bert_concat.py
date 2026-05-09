"""BERT-Concat baseline (Comment R2.1, Table 3 of the paper).

Concatenates Title, Abstract, Keywords, and Subject with [SEP] tokens into a
single sequence, fine-tunes BERT-base end-to-end, and classifies from the
[CLS] embedding. This baseline is included to isolate the contribution of
barycentric fusion from raw transformer capacity. See Section 4.2 of the
paper for the empirical comparison.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class BERTConcat(nn.Module):
    """[SEP]-concatenated BERT classifier."""

    FIELD_ORDER = ("title", "abstract", "keywords", "subject")

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        num_classes: int = 18,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.max_length = max_length
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _join(record: dict[str, str], sep: str = " [SEP] ") -> str:
        parts = [record.get(name, "") for name in BERTConcat.FIELD_ORDER]
        return sep.join(p for p in parts if p)

    def forward(self, fields: Mapping[str, list[str]]) -> Tensor:
        batch = [
            self._join({name: fields[name][i] for name in self.FIELD_ORDER})
            for i in range(len(next(iter(fields.values()))))
        ]
        enc = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(next(self.backbone.parameters()).device)
        cls = self.backbone(**enc).last_hidden_state[:, 0]
        return self.classifier(cls)
