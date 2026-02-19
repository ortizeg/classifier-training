"""Sample visualization callback — saves predicted-vs-true overlay grid."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from loguru import logger

# ImageNet normalization constants (duplicated here to avoid circular import)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SampleVisualizationCallback(L.Callback):
    """Save a grid of sample predictions with true-vs-predicted labels.

    Collects the first ``num_samples`` images from validation, runs inference,
    and renders a matplotlib grid with color-coded results (green = correct,
    red = incorrect).

    Args:
        output_dir: Root directory for saved artifacts.
        num_samples: Maximum number of samples to visualize per epoch.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        num_samples: int = 16,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

        # Accumulated during validation
        self._images: list[torch.Tensor] = []
        self._true_labels: list[int] = []
        self._pred_labels: list[int] = []

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Reset sample buffers at epoch start."""
        self._images.clear()
        self._true_labels.clear()
        self._pred_labels.clear()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect samples from early validation batches."""
        if len(self._images) >= self.num_samples:
            return

        images = batch["images"]  # type: ignore[index]
        labels = batch["labels"]  # type: ignore[index]

        with torch.no_grad():
            logits = pl_module(images)
        preds = logits.argmax(dim=1)

        remaining = self.num_samples - len(self._images)
        for i in range(min(remaining, images.size(0))):
            self._images.append(images[i].cpu())
            self._true_labels.append(labels[i].item())
            self._pred_labels.append(preds[i].item())

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Render and save the sample prediction grid."""
        if not self._images:
            return

        epoch = trainer.current_epoch

        # Resolve class names
        datamodule = getattr(trainer, "datamodule", None)
        class_to_idx: dict[str, int] | None = (
            getattr(datamodule, "class_to_idx", None)
            if datamodule is not None
            else None
        )
        idx_to_class: dict[int, str] | None = None
        if class_to_idx is not None:
            idx_to_class = {v: k for k, v in class_to_idx.items()}

        try:
            self._plot_grid(epoch, idx_to_class)
        except Exception as e:
            logger.error(f"Failed to save sample visualization: {e}")

    def _plot_grid(
        self,
        epoch: int,
        idx_to_class: dict[int, str] | None,
    ) -> None:
        """Render image grid with true/predicted labels."""
        matplotlib.use("Agg")

        n = len(self._images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i in range(rows * cols):
            row, col = divmod(i, cols)
            ax = axes[row][col]

            if i < n:
                img = self._denormalize(self._images[i])
                ax.imshow(img)

                true_label = self._true_labels[i]
                pred_label = self._pred_labels[i]

                true_name = (
                    idx_to_class.get(true_label, str(true_label))
                    if idx_to_class
                    else str(true_label)
                )
                pred_name = (
                    idx_to_class.get(pred_label, str(pred_label))
                    if idx_to_class
                    else str(pred_label)
                )

                correct = true_label == pred_label
                color = "green" if correct else "red"

                ax.set_title(
                    f"True: {true_name!r}\nPred: {pred_name!r}",
                    fontsize=9,
                    color=color,
                )

                # Color-coded border
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
            else:
                ax.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Sample Predictions - Epoch {epoch}", fontsize=14)
        fig.tight_layout()

        save_dir = self.output_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "sample_predictions.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        logger.info(f"Sample predictions saved to {save_path}")

    @staticmethod
    def _denormalize(img: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization and convert to HWC [0, 1] for imshow."""
        img = img * _IMAGENET_STD + _IMAGENET_MEAN
        img = img.clamp(0, 1)
        return img.permute(1, 2, 0)
