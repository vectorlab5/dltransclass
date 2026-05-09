"""Minimal training loop scaffold for DLTransClass."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dltransclass.models.dltransclass import DLTransClass
from dltransclass.training.surrogate import wasserstein_surrogate_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    output_dir: Path = Path("checkpoints")
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 8
    grad_accum_steps: int = 4
    log_every: int = 50
    eval_every_epochs: int = 1
    warmup_ratio: float = 0.1
    fp16: bool = True


class Trainer:
    """Lightweight trainer that wires together the surrogate loss and AdamW.

    This skeleton intentionally omits framework-specific niceties (DeepSpeed,
    distributed launchers, etc.) so that the core training logic remains
    transparent and easy to audit. Production runs should plug in the user's
    preferred orchestration layer.
    """

    def __init__(self, model: DLTransClass, args: TrainingArgs) -> None:
        self.model = model
        self.args = args
        self.optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.global_step = 0

    # ------------------------------------------------------------------
    def train(self, train_loader: Iterable, val_loader: Iterable | None = None) -> None:
        scheduler = self._build_scheduler(len(train_loader))
        for epoch in range(self.args.num_epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    out = self.model(batch["fields"], batch["observed"])
                    out["b"].retain_grad()
                    res = wasserstein_surrogate_loss(
                        out["logits"], batch["labels"], out["b"], out["rho"],
                        lambda_var=self.model.cfg.lambda_var,
                    )
                    loss = res["loss"] / self.args.grad_accum_steps

                self.scaler.scale(loss).backward()
                if (step + 1) % self.args.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    logger.info("step=%d ce=%.4f var=%.4f", self.global_step,
                                res["ce"].mean().item(), res["var"].mean().item())

            if val_loader is not None and (epoch + 1) % self.args.eval_every_epochs == 0:
                self.evaluate(val_loader)

        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.args.output_dir / "model.pt")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: Iterable) -> dict[str, float]:
        self.model.eval()
        # Implementation note: full metric computation lives in
        # `dltransclass.utils.metrics`; this method is a placeholder.
        return {}

    # ------------------------------------------------------------------
    def _build_scheduler(self, steps_per_epoch: int) -> LambdaLR:
        total_steps = max(1, (steps_per_epoch * self.args.num_epochs) // self.args.grad_accum_steps)
        warmup = int(total_steps * self.args.warmup_ratio)

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 1.0 - progress)

        return LambdaLR(self.optimizer, _lr_lambda)
