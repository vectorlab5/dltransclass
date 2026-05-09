"""Evaluate a trained DLTransClass checkpoint and print Macro-F1 / Accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dltransclass.data.datasets import BibliographicDataset
from dltransclass.models.dltransclass import DLTransClass, DLTransClassConfig
from dltransclass.utils.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    ds = BibliographicDataset(cfg["data"][args.split])
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=BibliographicDataset.collate,
    )

    model = DLTransClass(DLTransClassConfig(**cfg["model"]))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["fields"], batch["observed"])
            preds.append(out["logits"].argmax(dim=-1).cpu())
            labels.append(batch["labels"])
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    metrics = compute_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
