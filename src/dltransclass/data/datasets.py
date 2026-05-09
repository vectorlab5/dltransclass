"""Bibliographic record schema and dataset loaders.

Each dataset is expected to live as a parquet table under ``data/processed/``
with at least the columns ``title``, ``abstract``, ``keywords``, ``subject``,
and ``label``. Per-field observation flags are derived from non-empty strings
unless an explicit ``observed_<field>`` column is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

FIELD_NAMES: tuple[str, ...] = ("title", "abstract", "keywords", "subject")


@dataclass
class BibliographicRecord:
    """Single record with optional fields and a class label."""

    title: str = ""
    abstract: str = ""
    keywords: str = ""
    subject: str = ""
    label: int = -1

    def observed_mask(self) -> dict[str, int]:
        return {name: int(bool(getattr(self, name).strip())) for name in FIELD_NAMES}


class BibliographicDataset(Dataset):
    """Generic dataset wrapper for any of the six benchmarks.

    Parameters
    ----------
    parquet_path : path to the processed parquet file
    label_column : column with integer-encoded gold labels
    field_columns : iterable of column names to treat as bibliographic fields
    """

    def __init__(
        self,
        parquet_path: str | Path,
        label_column: str = "label",
        field_columns: Sequence[str] = FIELD_NAMES,
    ) -> None:
        self._df = pd.read_parquet(parquet_path)
        self._label_column = label_column
        self._field_columns = tuple(field_columns)

        missing = set(self._field_columns) - set(self._df.columns)
        if missing:
            raise ValueError(f"Missing field columns in parquet: {sorted(missing)}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        row = self._df.iloc[idx]
        fields = {name: str(row[name] or "") for name in self._field_columns}
        observed = {
            name: int(bool(row.get(f"observed_{name}", fields[name].strip())))
            for name in self._field_columns
        }
        return {
            "fields": fields,
            "observed": observed,
            "label": int(row[self._label_column]),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """Default collate that stacks fields and observation masks."""
        fields = {name: [b["fields"][name] for b in batch] for name in FIELD_NAMES}
        observed = {
            name: torch.tensor([b["observed"][name] for b in batch], dtype=torch.long)
            for name in FIELD_NAMES
        }
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"fields": fields, "observed": observed, "labels": labels}
