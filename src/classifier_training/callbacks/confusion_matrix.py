"""Confusion matrix callback for per-epoch heatmap visualization."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from loguru import logger
from torchmetrics.classification import MulticlassConfusionMatrix


class ConfusionMatrixCallback(L.Callback):
    """Save a confusion matrix heatmap PNG after each validation epoch.

    The ``MulticlassConfusionMatrix`` metric is created in ``on_fit_start``
    (not ``__init__``) to ensure it lives on the correct device (GPU/CPU).

    Args:
        output_dir: Root directory for saved artifacts.
        num_classes: Number of target classes.
        class_names: Optional list of human-readable class labels for axes.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        num_classes: int = 43,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_classes = num_classes
        self.class_names = class_names
        self._cm: MulticlassConfusionMatrix | None = None

    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Initialize confusion matrix metric on the model's device."""
        self._cm = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        ).to(pl_module.device)
        logger.info(
            f"ConfusionMatrixCallback: initialized on {pl_module.device}"
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions for the current validation batch."""
        if self._cm is None:
            return

        # batch is a ClassificationBatch dict
        images = batch["images"]  # type: ignore[index]
        labels = batch["labels"]  # type: ignore[index]

        # Run forward pass to get logits
        with torch.no_grad():
            logits = pl_module(images)
        preds = logits.argmax(dim=1)

        self._cm.update(preds, labels)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Compute, plot, and reset the confusion matrix."""
        if self._cm is None:
            return

        cm_tensor = self._cm.compute().cpu()
        self._cm.reset()

        epoch = trainer.current_epoch
        self._plot_and_save(cm_tensor, epoch)

    def _plot_and_save(self, cm: torch.Tensor, epoch: int) -> None:
        """Render confusion matrix as a heatmap and save to disk."""
        matplotlib.use("Agg")

        save_dir = self.output_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        cm_np = cm.numpy()

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_np, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_title(f"Confusion Matrix - Epoch {epoch}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        if self.class_names is not None:
            tick_marks = list(range(len(self.class_names)))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(self.class_names, rotation=45, ha="right")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(self.class_names)

        fig.tight_layout()
        save_path = save_dir / "confusion_matrix.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        logger.info(f"Confusion matrix saved to {save_path}")
