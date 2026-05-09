"""Train DLTransClass on a single dataset.

Example
-------
::

    python -m dltransclass.scripts.train \
        --config configs/experiments/udc_oldbooks.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from dltransclass.data.datasets import BibliographicDataset
from dltransclass.models.dltransclass import DLTransClass, DLTransClassConfig
from dltransclass.training.trainer import Trainer, TrainingArgs
from dltransclass.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())

    set_global_seed(args.seed)

    train_ds = BibliographicDataset(cfg["data"]["train"])
    val_ds = BibliographicDataset(cfg["data"]["val"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=BibliographicDataset.collate,
        num_workers=cfg["train"].get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=BibliographicDataset.collate,
        num_workers=cfg["train"].get("num_workers", 4),
    )

    model = DLTransClass(DLTransClassConfig(**cfg["model"]))
    trainer = Trainer(model, TrainingArgs(**cfg["train"]["args"]))
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
