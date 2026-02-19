"""Training history callback — saves loss and accuracy curve PNGs per epoch."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
from loguru import logger


class TrainingHistoryCallback(L.Callback):
    """Plot and save training/validation loss and accuracy curves.

    After each validation epoch, overwrites two PNG files:
    - ``loss_history.png``: train_loss vs val_loss
    - ``accuracy_history.png``: val_acc_top1, val_acc_top5, train_acc_top1

    Args:
        output_dir: Root directory for saved plots.
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "training_history"
        self.history: dict[str, list[float | None]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc_top1": [],
            "val_acc_top5": [],
            "train_acc_top1": [],
        }
        self.epochs: list[int] = []

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Collect metrics from callback_metrics at end of training epoch."""
        epoch = trainer.current_epoch
        self.epochs.append(epoch)

        metrics = trainer.callback_metrics

        # Collect each metric, falling back to None
        for key, metric_name in [
            ("train_loss", "train/loss_epoch"),
            ("val_loss", "val/loss"),
            ("val_acc_top1", "val/acc_top1"),
            ("val_acc_top5", "val/acc_top5"),
            ("train_acc_top1", "train/acc_top1"),
        ]:
            val = metrics.get(metric_name)
            if val is None and key == "train_loss":
                # Also try without _epoch suffix for train/loss
                val = metrics.get("train/loss")
            self.history[key].append(
                val.item() if val is not None else None
            )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Save updated plots after each validation epoch."""
        if not self.epochs:
            return

        try:
            self._plot_metrics()
        except Exception as e:
            logger.error(f"Failed to plot training history: {e}")

    def _plot_metrics(self) -> None:
        """Draw and save loss + accuracy plots."""
        matplotlib.use("Agg")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Loss plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label, marker in [
            ("train_loss", "Train Loss", "o"),
            ("val_loss", "Val Loss", "s"),
        ]:
            values = self.history[key]
            if any(v is not None for v in values):
                ax.plot(
                    self.epochs,
                    values,  # type: ignore[arg-type]
                    label=label,
                    marker=marker,
                )
        ax.set_title("Training and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_history.png", dpi=150)
        plt.close(fig)

        # --- Accuracy plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label, marker in [
            ("val_acc_top1", "Val Top-1", "s"),
            ("val_acc_top5", "Val Top-5", "^"),
            ("train_acc_top1", "Train Top-1", "o"),
        ]:
            values = self.history[key]
            if any(v is not None for v in values):
                ax.plot(
                    self.epochs,
                    values,  # type: ignore[arg-type]
                    label=label,
                    marker=marker,
                )
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        fig.savefig(self.output_dir / "accuracy_history.png", dpi=150)
        plt.close(fig)

        logger.info(f"Training history plots updated in {self.output_dir}")
