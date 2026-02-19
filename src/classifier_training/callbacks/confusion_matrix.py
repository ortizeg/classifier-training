"""Confusion matrix callback for per-epoch heatmap visualization."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import WandbLogger
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

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Initialize confusion matrix metric on the model's device."""
        self._cm = MulticlassConfusionMatrix(num_classes=self.num_classes).to(
            pl_module.device
        )

        # Auto-detect class names from datamodule if not explicitly provided
        if self.class_names is None:
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is not None and hasattr(datamodule, "class_to_idx"):
                c2i: dict[str, int] = datamodule.class_to_idx
                # Build ordered list: idx -> class name
                self.class_names = [
                    name for name, _ in sorted(c2i.items(), key=lambda x: x[1])
                ]
                logger.info(
                    f"ConfusionMatrixCallback: auto-detected {len(self.class_names)} "
                    "class names from datamodule"
                )

        logger.info(f"ConfusionMatrixCallback: initialized on {pl_module.device}")

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
        png_path = self._plot_and_save(cm_tensor, epoch)

        # Log confusion matrix image to WandB if WandbLogger is active
        if png_path.exists():
            for lgr in trainer.loggers:
                if isinstance(lgr, WandbLogger):
                    lgr.log_image(
                        key="confusion_matrix",
                        images=[str(png_path)],
                        step=trainer.current_epoch,
                    )
                    logger.debug(f"Logged confusion matrix to WandB (epoch {epoch})")

    def _plot_and_save(self, cm: torch.Tensor, epoch: int) -> Path:
        """Render confusion matrix as a heatmap and save to disk."""
        matplotlib.use("Agg")

        save_dir = self.output_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        cm_np = cm.numpy()

        # Scale figure size with number of classes for readability
        n = cm_np.shape[0]
        fig_size = max(12, n * 0.35)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        im = ax.imshow(cm_np, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_title(f"Confusion Matrix - Epoch {epoch}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Always label axes — use class names if available, else integers
        labels = self.class_names or [str(i) for i in range(n)]
        tick_marks = list(range(n))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, fontsize=8)

        fig.tight_layout()
        save_path = save_dir / "confusion_matrix.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path
