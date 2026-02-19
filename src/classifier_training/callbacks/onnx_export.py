"""
ONNX export callback for PyTorch Lightning.

Exports the trained model to ONNX format at the end of training,
preferring EMA weights when an EMACallback is present.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import lightning as L
import torch
from loguru import logger


class ONNXExportCallback(L.Callback):
    """Export the model to ONNX at the end of training.

    When an :class:`~classifier_training.callbacks.ema.EMACallback` is found
    among the trainer's callbacks, the ONNX export uses the EMA state dict
    instead of the live model weights.

    Args:
        output_dir: Directory to write the ONNX model into.
        opset_version: ONNX opset version for ``torch.onnx.export``.
        input_height: Expected input image height.
        input_width: Expected input image width.
    """

    def __init__(
        self,
        output_dir: str = "onnx",
        opset_version: int = 17,
        input_height: int = 224,
        input_width: int = 224,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.opset_version = opset_version
        self.input_height = input_height
        self.input_width = input_width

    def on_train_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Export the model to ONNX at end of training."""
        from classifier_training.callbacks.ema import EMACallback

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_path = output_path / "model.onnx"

        # Determine export weights: prefer EMA if available
        export_state_dict: dict[str, Any] | None = None
        for cb in trainer.callbacks:  # type: ignore[attr-defined]
            if isinstance(cb, EMACallback) and cb.ema_state_dict:
                logger.info("Using EMA weights for ONNX export")
                export_state_dict = cb.ema_state_dict
                break

        # Save original state, load export weights, export, restore
        original_state = copy.deepcopy(pl_module.state_dict())
        try:
            if export_state_dict is not None:
                pl_module.load_state_dict(export_state_dict)

            self._export_to_onnx(pl_module, onnx_path)
        finally:
            pl_module.load_state_dict(original_state)

        # Write labels_mapping.json sidecar if datamodule is available
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(
            datamodule, "save_labels_mapping"
        ):
            labels_path = output_path / "labels_mapping.json"
            datamodule.save_labels_mapping(labels_path)
            logger.info(f"Labels mapping saved to {labels_path}")

    def _export_to_onnx(
        self, pl_module: L.LightningModule, output_path: Path
    ) -> None:
        """Export a deep copy of the module to ONNX using the legacy exporter."""
        # Deep copy and move to CPU for export
        model_copy = copy.deepcopy(pl_module).cpu().eval()

        dummy_input = torch.randn(
            1, 3, self.input_height, self.input_width
        )

        # Force legacy exporter path to avoid dynamo issues
        original_env = os.environ.get("TORCH_ONNX_LEGACY_EXPORTER")
        os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"
        try:
            _real_export = torch.onnx.export
            with patch.object(
                torch.onnx,
                "export",
                lambda *a, **kw: _real_export(
                    *a, **{k: v for k, v in kw.items() if k != "dynamo"}, dynamo=False
                ),
            ):
                torch.onnx.export(
                    model_copy,
                    (dummy_input,),
                    str(output_path),
                    input_names=["input"],
                    output_names=["logits"],
                    opset_version=self.opset_version,
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                    },
                )
        finally:
            if original_env is None:
                os.environ.pop("TORCH_ONNX_LEGACY_EXPORTER", None)
            else:
                os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = original_env

        file_size = output_path.stat().st_size
        logger.info(
            f"ONNX model exported to {output_path} ({file_size / 1024:.1f} KB)"
        )
