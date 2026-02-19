"""Base LightningModule for all classification models."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.classification import MulticlassAccuracy

from classifier_training.types import ClassificationBatch


class BaseClassificationModel(L.LightningModule):
    """Abstract base for ResNet classification models.

    Subclasses must set ``self.model`` (nn.Module backbone) in ``__init__``
    and implement ``forward()``.  Do **not** pass ``class_weights`` to
    ``__init__``; call :meth:`set_class_weights` from the training script
    after ``datamodule.setup()``.
    """

    # Declare buffer type explicitly so mypy knows it is always a Tensor.
    class_weights: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
        warmup_start_factor: float = 1e-3,
        cosine_eta_min_factor: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Register class_weights as buffer so .to(device) transfers it
        # automatically.  Default ones; override via set_class_weights().
        self.register_buffer("class_weights", torch.ones(num_classes))
        self._build_loss_fn()

        # Pattern A metrics -- one set per split, auto device placement.
        # top_k_5 guard: MulticlassAccuracy raises ValueError if
        # top_k > num_classes.
        top_k_5 = min(5, num_classes)
        self.train_top1 = MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro"
        )
        self.val_top1 = MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro"
        )
        self.val_top5 = MulticlassAccuracy(
            num_classes=num_classes, top_k=top_k_5, average="micro"
        )
        self.val_per_cls = MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="none"
        )
        self.test_top1 = MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro"
        )
        self.test_top5 = MulticlassAccuracy(
            num_classes=num_classes, top_k=top_k_5, average="micro"
        )
        self.test_per_cls = MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="none"
        )

    def _build_loss_fn(self) -> None:
        """Build CrossEntropyLoss from registered class_weights buffer."""
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.hparams["label_smoothing"],
        )

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Update class weights buffer and rebuild loss function.

        Call this from the training script after ``datamodule.setup()``::

            model.set_class_weights(dm.get_class_weights())
        """
        self.class_weights.copy_(weights.to(self.class_weights.device))
        self._build_loss_fn()

    def training_step(
        self, batch: ClassificationBatch, batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        # Pattern A: update only in step; compute+log+reset in epoch_end
        self.train_top1.update(logits, labels)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc_top1", self.train_top1.compute())
        self.train_top1.reset()

    def validation_step(
        self, batch: ClassificationBatch, batch_idx: int
    ) -> None:
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.val_top1.update(logits, labels)
        self.val_top5.update(logits, labels)
        self.val_per_cls.update(logits, labels)

    def on_validation_epoch_end(self) -> None:
        self.log("val/acc_top1", self.val_top1.compute(), prog_bar=True)
        self.log("val/acc_top5", self.val_top5.compute())
        per_cls: torch.Tensor = self.val_per_cls.compute()
        for i, acc in enumerate(per_cls):
            self.log(f"val/acc_class_{i}", acc)
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_per_cls.reset()

    def test_step(
        self, batch: ClassificationBatch, batch_idx: int
    ) -> None:
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_top1.update(logits, labels)
        self.test_top5.update(logits, labels)
        self.test_per_cls.update(logits, labels)

    def on_test_epoch_end(self) -> None:
        self.log("test/acc_top1", self.test_top1.compute())
        self.log("test/acc_top5", self.test_top5.compute())
        per_cls: torch.Tensor = self.test_per_cls.compute()
        for i, acc in enumerate(per_cls):
            self.log(f"test/acc_class_{i}", acc)
        self.test_top1.reset()
        self.test_top5.reset()
        self.test_per_cls.reset()

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        max_epochs = (
            (self.trainer.max_epochs or 100) if self.trainer else 100
        )
        warmup = int(self.hparams["warmup_epochs"])
        cosine_epochs = max(1, max_epochs - warmup)
        eta_min = (
            self.hparams["learning_rate"]
            * self.hparams["cosine_eta_min_factor"]
        )

        warmup_sched = LinearLR(
            optimizer,
            start_factor=self.hparams["warmup_start_factor"],
            end_factor=1.0,
            total_iters=max(1, warmup),
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
