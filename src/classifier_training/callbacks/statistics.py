"""Dataset statistics callback — prints class distribution at training start."""

from __future__ import annotations

from collections import Counter

import lightning as L
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


class DatasetStatisticsCallback(L.Callback):
    """Print a rich table of class distribution from the training dataset.

    Accesses ``trainer.datamodule._train_dataset.samples`` to count labels
    per class and displays them sorted by index.  When metadata is available
    on the dataset, also prints jersey color, number color, and border
    distributions.
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

        # Print metadata distributions if available
        metadata_list: list[dict[str, object] | None] = getattr(
            train_dataset, "metadata", []
        )
        self._print_metadata_stats(console, metadata_list)

    def _print_metadata_stats(
        self,
        console: Console,
        metadata_list: list[dict[str, object] | None],
    ) -> None:
        """Print metadata distribution tables if metadata is present."""
        if not metadata_list:
            return

        entries = [m for m in metadata_list if m is not None]
        if not entries:
            return

        total = len(metadata_list)
        with_meta = len(entries)
        logger.info(
            f"Metadata coverage: {with_meta}/{total} ({with_meta / total * 100:.1f}%)"
        )

        jersey_counts: Counter[str] = Counter()
        number_counts: Counter[str] = Counter()
        border_counts: Counter[bool] = Counter()

        for meta in entries:
            jersey_counts[str(meta.get("jersey_color", "unknown"))] += 1
            number_counts[str(meta.get("number_color", "unknown"))] += 1
            border_val = meta.get("border", False)
            if not isinstance(border_val, bool):
                border_val = str(border_val).lower() in ("true", "yes", "1")
            border_counts[border_val] += 1

        # Jersey color table
        color_table = Table(
            title="Metadata: Color Distributions",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        color_table.add_column("Jersey Color", style="cyan")
        color_table.add_column("Count", justify="right", style="green")
        color_table.add_column("Number Color", style="cyan")
        color_table.add_column("Count", justify="right", style="green")

        jersey_sorted = jersey_counts.most_common()
        number_sorted = number_counts.most_common()
        max_rows = max(len(jersey_sorted), len(number_sorted))

        for i in range(max_rows):
            jc = jersey_sorted[i][0] if i < len(jersey_sorted) else ""
            jn = str(jersey_sorted[i][1]) if i < len(jersey_sorted) else ""
            nc = number_sorted[i][0] if i < len(number_sorted) else ""
            nn = str(number_sorted[i][1]) if i < len(number_sorted) else ""
            color_table.add_row(jc, jn, nc, nn)

        console.print(color_table)

        # Border distribution
        logger.info(f"Border: yes={border_counts[True]}, no={border_counts[False]}")
