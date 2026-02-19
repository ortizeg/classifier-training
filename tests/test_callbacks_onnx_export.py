"""Tests for the ONNX export callback."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import lightning as L
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from classifier_training.callbacks.ema import EMACallback
from classifier_training.callbacks.onnx_export import ONNXExportCallback

NUM_CLASSES = 5


class _TinyClassifier(L.LightningModule):
    """Minimal LightningModule for ONNX export tests."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return torch.tensor(0.0)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TestONNXExportCallbackInit:
    """Tests for initialization."""

    def test_default_parameters(self) -> None:
        cb = ONNXExportCallback()
        assert cb.output_dir == "onnx"
        assert cb.opset_version == 17
        assert cb.input_height == 224
        assert cb.input_width == 224

    def test_custom_parameters(self) -> None:
        cb = ONNXExportCallback(
            output_dir="custom_onnx",
            opset_version=16,
            input_height=128,
            input_width=128,
        )
        assert cb.output_dir == "custom_onnx"
        assert cb.opset_version == 16
        assert cb.input_height == 128


class TestONNXExportCallbackExport:
    """Tests for ONNX export via on_train_end."""

    def test_onnx_file_created(self, tmp_path: Path) -> None:
        """ONNX file is created in output_dir after on_train_end."""
        pl_module = _TinyClassifier()
        trainer = MagicMock()
        trainer.callbacks = []
        trainer.datamodule = None

        output_dir = str(tmp_path / "onnx_out")
        cb = ONNXExportCallback(
            output_dir=output_dir, input_height=32, input_width=32
        )
        cb.on_train_end(trainer, pl_module)

        onnx_path = tmp_path / "onnx_out" / "model.onnx"
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_onnx_output_name_is_logits(self, tmp_path: Path) -> None:
        """ONNX model output name must be 'logits'."""
        pl_module = _TinyClassifier()
        trainer = MagicMock()
        trainer.callbacks = []
        trainer.datamodule = None

        output_dir = str(tmp_path / "onnx_out")
        cb = ONNXExportCallback(
            output_dir=output_dir, input_height=32, input_width=32
        )
        cb.on_train_end(trainer, pl_module)

        session = ort.InferenceSession(
            str(tmp_path / "onnx_out" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        outputs = session.get_outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "logits"

    def test_onnx_dynamic_batch_size(self, tmp_path: Path) -> None:
        """ONNX model accepts dynamic batch sizes (1 and 2)."""
        pl_module = _TinyClassifier()
        trainer = MagicMock()
        trainer.callbacks = []
        trainer.datamodule = None

        output_dir = str(tmp_path / "onnx_out")
        cb = ONNXExportCallback(
            output_dir=output_dir, input_height=32, input_width=32
        )
        cb.on_train_end(trainer, pl_module)

        session = ort.InferenceSession(
            str(tmp_path / "onnx_out" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

        # Batch size 1
        input_b1 = np.random.randn(1, 3, 32, 32).astype(np.float32)
        result_b1 = session.run(None, {"input": input_b1})
        assert result_b1[0].shape == (1, NUM_CLASSES)

        # Batch size 2
        input_b2 = np.random.randn(2, 3, 32, 32).astype(np.float32)
        result_b2 = session.run(None, {"input": input_b2})
        assert result_b2[0].shape == (2, NUM_CLASSES)

    def test_onnx_output_shape(self, tmp_path: Path) -> None:
        """ONNX model output shape is (batch_size, num_classes)."""
        pl_module = _TinyClassifier(num_classes=10)
        trainer = MagicMock()
        trainer.callbacks = []
        trainer.datamodule = None

        output_dir = str(tmp_path / "onnx_out")
        cb = ONNXExportCallback(
            output_dir=output_dir, input_height=32, input_width=32
        )
        cb.on_train_end(trainer, pl_module)

        session = ort.InferenceSession(
            str(tmp_path / "onnx_out" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        input_data = np.random.randn(3, 3, 32, 32).astype(np.float32)
        result = session.run(None, {"input": input_data})
        assert result[0].shape == (3, 10)


class TestONNXExportWithEMA:
    """Tests for ONNX export using EMA weights."""

    def test_uses_ema_weights_when_present(self, tmp_path: Path) -> None:
        """ONNX export uses EMA weights when EMACallback is among trainer callbacks."""
        model = _TinyClassifier()

        # Create EMA callback with different weights
        ema_cb = EMACallback(decay=0.9, warmup_steps=0)
        ema_cb.ema_state_dict = copy.deepcopy(model.state_dict())
        # Shift EMA weights so they differ from model
        for key in ema_cb.ema_state_dict:
            if ema_cb.ema_state_dict[key].dtype.is_floating_point:
                ema_cb.ema_state_dict[key] = (
                    ema_cb.ema_state_dict[key] + 10.0
                )

        trainer = MagicMock()
        trainer.callbacks = [ema_cb]
        trainer.datamodule = None

        output_dir = str(tmp_path / "onnx_ema")
        cb = ONNXExportCallback(
            output_dir=output_dir, input_height=32, input_width=32
        )
        cb.on_train_end(trainer, model)

        # Verify ONNX file was created
        onnx_path = tmp_path / "onnx_ema" / "model.onnx"
        assert onnx_path.exists()

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        ema_result = session.run(None, {"input": input_data})[0]

        # Export again without EMA to compare
        trainer_no_ema = MagicMock()
        trainer_no_ema.callbacks = []
        trainer_no_ema.datamodule = None
        output_dir_no_ema = str(tmp_path / "onnx_no_ema")
        cb2 = ONNXExportCallback(
            output_dir=output_dir_no_ema, input_height=32, input_width=32
        )
        cb2.on_train_end(trainer_no_ema, model)

        session2 = ort.InferenceSession(
            str(tmp_path / "onnx_no_ema" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        no_ema_result = session2.run(None, {"input": input_data})[0]

        # EMA and non-EMA outputs should differ
        assert not np.allclose(ema_result, no_ema_result, atol=1e-5)
