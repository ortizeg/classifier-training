"""Sampler distribution callback.

Logs class sample counts from TrackingWeightedRandomSampler.
"""

from __future__ import annotations

import lightning as L
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from classifier_training.data.sampler import TrackingWeightedRandomSampler


class SamplerDistributionCallback(L.Callback):
    """Log per-class sample counts from TrackingWeightedRandomSampler each epoch.

    At the start of each training epoch (from epoch 1 onward), reads
    ``_last_indices`` from the sampler and maps them to class labels.
    Prints a rich table comparing sampled counts against expected uniform
    distribution.

    Skips silently if the sampler is not a ``TrackingWeightedRandomSampler``.
    """

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Read sampled indices and print class distribution table."""
        train_dl = trainer.train_dataloader
        if train_dl is None:
            return

        sampler = getattr(train_dl, "sampler", None)
        if not isinstance(sampler, TrackingWeightedRandomSampler):
            logger.info(
                "Sampler is not TrackingWeightedRandomSampler. "
                "Skipping distribution logging."
            )
            return

        if not sampler._last_indices:
            # First epoch — no indices yet
            return

        # Get datamodule for class mapping
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        train_dataset = getattr(datamodule, "_train_dataset", None)
        if train_dataset is None:
            return

        samples: list[tuple[str, int]] = train_dataset.samples
        class_to_idx: dict[str, int] = datamodule.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Count sampled classes
        sampled_counts: dict[int, int] = {}
        for idx in sampler._last_indices:
            _, label = samples[idx]
            sampled_counts[label] = sampled_counts.get(label, 0) + 1

        total_sampled = len(sampler._last_indices)
        num_classes = len(class_to_idx)
        expected_per_class = total_sampled / num_classes if num_classes > 0 else 0

        # Print rich table
        console = Console()
        table = Table(
            title=f"Sampler Distribution (Epoch {trainer.current_epoch})",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Class Name", style="cyan")
        table.add_column("Sampled Count", justify="right", style="green")
        table.add_column("Expected Count", justify="right")
        table.add_column("Ratio", justify="right", style="yellow")

        for class_idx in sorted(idx_to_class.keys()):
            name = idx_to_class[class_idx]
            count = sampled_counts.get(class_idx, 0)
            ratio = count / expected_per_class if expected_per_class > 0 else 0.0
            table.add_row(
                repr(name) if name == "" else name,
                str(count),
                f"{expected_per_class:.0f}",
                f"{ratio:.2f}x",
            )

        console.print(table)
        logger.info(
            f"Sampler distribution: {total_sampled} samples across "
            f"{len(sampled_counts)} classes"
        )
