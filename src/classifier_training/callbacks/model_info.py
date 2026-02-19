"""Model info callback — reports parameter counts and writes labels_mapping.json."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


class ModelInfoCallback(L.Callback):
    """Compute and display model statistics at training start.

    Reports total parameters, trainable parameters, and model size in MB.
    Optionally writes ``labels_mapping.json`` via the datamodule.

    No FLOPs computation — avoided to keep dependencies minimal (no fvcore).

    Args:
        output_dir: Directory for saved artifacts (labels_mapping.json).
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Compute model stats, print table, write labels_mapping.json."""
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        param_size = sum(
            p.numel() * p.element_size() for p in pl_module.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in pl_module.buffers()
        )
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        # Print rich table
        console = Console()
        table = Table(
            title="Model Information",
            header_style="bold magenta",
            box=box.SQUARE,
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model Class", type(pl_module).__name__)
        table.add_row("Total Parameters", f"{total_params / 1e6:.2f} M")
        table.add_row("Trainable Parameters", f"{trainable_params / 1e6:.2f} M")
        table.add_row("Model Size", f"{model_size_mb:.2f} MB")

        console.print(table)

        logger.info(
            f"Model: {type(pl_module).__name__} | "
            f"Params: {total_params:,} ({trainable_params:,} trainable) | "
            f"Size: {model_size_mb:.2f} MB"
        )

        # Write labels_mapping.json
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "save_labels_mapping"):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            labels_path = self.output_dir / "labels_mapping.json"
            try:
                datamodule.save_labels_mapping(labels_path)
            except Exception as e:
                logger.warning(f"Failed to write labels_mapping.json: {e}")
        else:
            logger.info(
                "No datamodule with save_labels_mapping found. "
                "Skipping labels_mapping.json."
            )
