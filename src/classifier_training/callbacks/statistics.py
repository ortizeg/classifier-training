"""Dataset statistics callback — prints class distribution at training start."""

from __future__ import annotations

import lightning as L
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


class DatasetStatisticsCallback(L.Callback):
    """Print a rich table of class distribution from the training dataset.

    Accesses ``trainer.datamodule._train_dataset.samples`` to count labels
    per class and displays them sorted by index.
    """

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute and display class distribution at training start."""
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            logger.warning("No datamodule found. Skipping dataset statistics.")
            return

        train_dataset = getattr(datamodule, "_train_dataset", None)
        if train_dataset is None:
            logger.warning(
                "No _train_dataset found on datamodule. Skipping dataset statistics."
            )
            return

        samples: list[tuple[str, int]] = train_dataset.samples

        # Build idx_to_class from class_to_idx
        class_to_idx: dict[str, int] = datamodule.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Count per class
        counts: dict[int, int] = {}
        for _, label in samples:
            counts[label] = counts.get(label, 0) + 1

        total = len(samples)
        logger.info(f"Training dataset: {total} samples, {len(counts)} classes")

        # Build rich table
        console = Console()
        table = Table(
            title="Dataset Class Distribution",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Class Name", style="cyan")
        table.add_column("Index", justify="right")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        for idx in sorted(counts.keys()):
            name = idx_to_class.get(idx, f"unknown_{idx}")
            count = counts[idx]
            pct = count / total * 100 if total > 0 else 0.0
            table.add_row(
                repr(name) if name == "" else name,
                str(idx),
                str(count),
                f"{pct:.1f}%",
            )

        console.print(table)
