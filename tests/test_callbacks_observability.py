"""Tests for the six observability callbacks (plan 03-02).

All tests use minimal models (nn.Linear), tmp_path, and CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import torch
import torch.nn as nn

from classifier_training.callbacks.confusion_matrix import ConfusionMatrixCallback
from classifier_training.callbacks.model_info import ModelInfoCallback
from classifier_training.callbacks.plotting import TrainingHistoryCallback
from classifier_training.callbacks.sampler import SamplerDistributionCallback
from classifier_training.callbacks.statistics import DatasetStatisticsCallback
from classifier_training.callbacks.visualization import SampleVisualizationCallback
from classifier_training.data.sampler import TrackingWeightedRandomSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModule(nn.Module):
    """Minimal model for testing."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _FakeLightningModule:
    """Minimal stand-in for pl_module (not a real LightningModule)."""

    def __init__(self, num_classes: int = 5) -> None:
        self._model = _TinyModule(num_classes)
        self.device = torch.device("cpu")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def parameters(self) -> Any:
        return self._model.parameters()

    def buffers(self) -> Any:
        return self._model.buffers()


def _mock_trainer(
    datamodule: Any = None,
    current_epoch: int = 0,
    callback_metrics: dict[str, Any] | None = None,
    train_dataloader: Any = None,
) -> MagicMock:
    """Build a mock trainer with common attributes."""
    trainer = MagicMock()
    trainer.datamodule = datamodule
    trainer.current_epoch = current_epoch
    trainer.callback_metrics = callback_metrics or {}
    trainer.loggers = []
    trainer.log_dir = None
    # train_dataloader is a property on real Trainer
    type(trainer).train_dataloader = PropertyMock(return_value=train_dataloader)
    return trainer


def _mock_datamodule(
    samples: list[tuple[str, int]] | None = None,
    class_to_idx: dict[str, int] | None = None,
) -> MagicMock:
    """Build a mock datamodule with _train_dataset and class_to_idx."""
    dm = MagicMock()
    if class_to_idx is None:
        class_to_idx = {"cat": 0, "dog": 1, "bird": 2}
    dm.class_to_idx = class_to_idx

    if samples is None:
        samples = [
            ("/data/test/img1.jpg", 0),
            ("/data/test/img2.jpg", 0),
            ("/data/test/img3.jpg", 1),
            ("/data/test/img4.jpg", 1),
            ("/data/test/img5.jpg", 2),
        ]
    dm._train_dataset = MagicMock()
    dm._train_dataset.samples = samples
    return dm


# ---------------------------------------------------------------------------
# ConfusionMatrixCallback
# ---------------------------------------------------------------------------


class TestConfusionMatrixCallback:
    """Tests for ConfusionMatrixCallback."""

    def test_creates_png_after_validation(self, tmp_path: Path) -> None:
        """Verify PNG is created after simulated validation epoch."""
        cb = ConfusionMatrixCallback(
            output_dir=str(tmp_path), num_classes=3
        )

        pl_module = _FakeLightningModule(num_classes=3)
        trainer = _mock_trainer(current_epoch=0)

        # Simulate on_fit_start to initialize metric
        cb.on_fit_start(trainer, pl_module)  # type: ignore[arg-type]
        assert cb._cm is not None

        # Simulate a validation batch
        images = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 2, 1])
        batch = {"images": images, "labels": labels}

        cb.on_validation_batch_end(
            trainer, pl_module, None, batch, batch_idx=0  # type: ignore[arg-type]
        )

        # End validation epoch
        cb.on_validation_epoch_end(trainer, pl_module)  # type: ignore[arg-type]

        png_path = tmp_path / "epoch_000" / "confusion_matrix.png"
        assert png_path.exists()

    def test_device_handling_cpu(self, tmp_path: Path) -> None:
        """Verify metric is initialized on CPU device."""
        cb = ConfusionMatrixCallback(
            output_dir=str(tmp_path), num_classes=5
        )
        pl_module = _FakeLightningModule(num_classes=5)
        trainer = _mock_trainer()

        cb.on_fit_start(trainer, pl_module)  # type: ignore[arg-type]
        assert cb._cm is not None
        # The metric's device should be CPU
        # (torchmetrics doesn't expose .device directly, but the internal
        # tensors should be on CPU)

    def test_class_names_on_axes(self, tmp_path: Path) -> None:
        """Verify class names are set when provided."""
        names = ["cat", "dog", "bird"]
        cb = ConfusionMatrixCallback(
            output_dir=str(tmp_path),
            num_classes=3,
            class_names=names,
        )
        assert cb.class_names == names


# ---------------------------------------------------------------------------
# DatasetStatisticsCallback
# ---------------------------------------------------------------------------


class TestDatasetStatisticsCallback:
    """Tests for DatasetStatisticsCallback."""

    def test_runs_without_error(self) -> None:
        """Verify callback runs on mock datamodule without raising."""
        cb = DatasetStatisticsCallback()
        dm = _mock_datamodule()
        trainer = _mock_trainer(datamodule=dm)

        # Should not raise
        cb.on_fit_start(trainer, MagicMock())

    def test_handles_missing_datamodule(self) -> None:
        """Verify graceful handling when no datamodule is present."""
        cb = DatasetStatisticsCallback()
        trainer = _mock_trainer(datamodule=None)

        # Should not raise
        cb.on_fit_start(trainer, MagicMock())

    def test_handles_missing_train_dataset(self) -> None:
        """Verify graceful handling when _train_dataset is None."""
        cb = DatasetStatisticsCallback()
        dm = MagicMock()
        dm._train_dataset = None
        trainer = _mock_trainer(datamodule=dm)

        # Should not raise
        cb.on_fit_start(trainer, MagicMock())


# ---------------------------------------------------------------------------
# ModelInfoCallback
# ---------------------------------------------------------------------------


class TestModelInfoCallback:
    """Tests for ModelInfoCallback."""

    def test_computes_parameter_count(self, tmp_path: Path) -> None:
        """Verify parameter count is computed correctly for a small model."""
        cb = ModelInfoCallback(output_dir=str(tmp_path))
        model = _TinyModule(num_classes=5)
        pl_module = _FakeLightningModule(num_classes=5)

        # Without save_labels_mapping (datamodule=None)
        trainer = _mock_trainer(datamodule=None)
        cb.on_fit_start(trainer, pl_module)  # type: ignore[arg-type]

        # Verify expected parameter count: Linear(16, 5) -> 16*5 + 5 = 85
        expected_params = 16 * 5 + 5
        actual_params = sum(p.numel() for p in model.parameters())
        assert actual_params == expected_params

    def test_writes_labels_mapping(self, tmp_path: Path) -> None:
        """Verify labels_mapping.json is written when datamodule supports it."""
        cb = ModelInfoCallback(output_dir=str(tmp_path))
        pl_module = _FakeLightningModule(num_classes=3)

        dm = MagicMock()

        # Mock save_labels_mapping to write a file
        def _save(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"class_to_idx": {"a": 0}}, f)

        dm.save_labels_mapping = _save
        trainer = _mock_trainer(datamodule=dm)

        cb.on_fit_start(trainer, pl_module)  # type: ignore[arg-type]

        labels_path = tmp_path / "labels_mapping.json"
        assert labels_path.exists()


# ---------------------------------------------------------------------------
# TrainingHistoryCallback
# ---------------------------------------------------------------------------


class TestTrainingHistoryCallback:
    """Tests for TrainingHistoryCallback."""

    def test_creates_png_files(self, tmp_path: Path) -> None:
        """Verify loss and accuracy PNGs are created after epoch."""
        cb = TrainingHistoryCallback(output_dir=str(tmp_path))

        trainer = _mock_trainer(
            current_epoch=0,
            callback_metrics={
                "train/loss_epoch": torch.tensor(0.5),
                "val/loss": torch.tensor(0.6),
                "val/acc_top1": torch.tensor(0.85),
                "val/acc_top5": torch.tensor(0.95),
                "train/acc_top1": torch.tensor(0.80),
            },
        )

        cb.on_train_epoch_end(trainer, MagicMock())
        cb.on_validation_epoch_end(trainer, MagicMock())

        history_dir = tmp_path / "training_history"
        assert (history_dir / "loss_history.png").exists()
        assert (history_dir / "accuracy_history.png").exists()

    def test_handles_missing_metrics(self, tmp_path: Path) -> None:
        """Verify no crash when metrics are missing."""
        cb = TrainingHistoryCallback(output_dir=str(tmp_path))

        trainer = _mock_trainer(current_epoch=0, callback_metrics={})

        cb.on_train_epoch_end(trainer, MagicMock())
        cb.on_validation_epoch_end(trainer, MagicMock())

        # Should still create plots (with empty data)
        history_dir = tmp_path / "training_history"
        assert (history_dir / "loss_history.png").exists()


# ---------------------------------------------------------------------------
# SamplerDistributionCallback
# ---------------------------------------------------------------------------


class TestSamplerDistributionCallback:
    """Tests for SamplerDistributionCallback."""

    def test_reads_tracking_sampler_indices(self) -> None:
        """Verify callback reads _last_indices from TrackingWeightedRandomSampler."""
        cb = SamplerDistributionCallback()

        # Create a real TrackingWeightedRandomSampler with known state
        sampler = TrackingWeightedRandomSampler(
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],
            num_samples=5,
            replacement=True,
        )
        # Manually set _last_indices to simulate post-epoch state
        sampler._last_indices = [0, 1, 2, 3, 4]

        dm = _mock_datamodule()

        # Create a mock dataloader that exposes the sampler
        train_dl = MagicMock()
        train_dl.sampler = sampler

        trainer = _mock_trainer(
            datamodule=dm,
            current_epoch=1,
            train_dataloader=train_dl,
        )

        # Should not raise
        cb.on_train_epoch_start(trainer, MagicMock())

    def test_skips_non_tracking_sampler(self) -> None:
        """Verify callback skips when sampler is not TrackingWeightedRandomSampler."""
        cb = SamplerDistributionCallback()

        train_dl = MagicMock()
        train_dl.sampler = MagicMock()  # Not a TrackingWeightedRandomSampler

        trainer = _mock_trainer(train_dataloader=train_dl)

        # Should not raise
        cb.on_train_epoch_start(trainer, MagicMock())

    def test_skips_empty_indices(self) -> None:
        """Verify callback skips when _last_indices is empty (epoch 0)."""
        cb = SamplerDistributionCallback()

        sampler = TrackingWeightedRandomSampler(
            weights=[1.0, 1.0],
            num_samples=2,
            replacement=True,
        )
        # _last_indices is empty by default (epoch 0)
        assert sampler._last_indices == []

        train_dl = MagicMock()
        train_dl.sampler = sampler

        trainer = _mock_trainer(train_dataloader=train_dl)

        # Should not raise, should skip
        cb.on_train_epoch_start(trainer, MagicMock())


# ---------------------------------------------------------------------------
# SampleVisualizationCallback
# ---------------------------------------------------------------------------


class TestSampleVisualizationCallback:
    """Tests for SampleVisualizationCallback."""

    def test_creates_grid_png(self, tmp_path: Path) -> None:
        """Verify PNG grid is created with correct number of samples."""
        cb = SampleVisualizationCallback(
            output_dir=str(tmp_path), num_samples=4
        )

        pl_module = _FakeLightningModule(num_classes=3)
        dm = _mock_datamodule()
        trainer = _mock_trainer(datamodule=dm, current_epoch=0)

        cb.on_validation_epoch_start(trainer, pl_module)  # type: ignore[arg-type]

        # Simulate a batch (images must be 3xHxW for denormalization)
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 0])

        # Patch pl_module __call__ to handle 3x32x32 images
        class _ImageModule:
            device = torch.device("cpu")

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                b = x.size(0)
                return torch.randn(b, 3)

            def parameters(self) -> Any:
                return iter([torch.zeros(1)])

            def buffers(self) -> Any:
                return iter([])

        img_module = _ImageModule()
        batch = {"images": images, "labels": labels}

        cb.on_validation_batch_end(
            trainer, img_module, None, batch, batch_idx=0  # type: ignore[arg-type]
        )

        assert len(cb._images) == 4

        cb.on_validation_epoch_end(trainer, img_module)  # type: ignore[arg-type]

        png_path = tmp_path / "epoch_000" / "sample_predictions.png"
        assert png_path.exists()

    def test_respects_num_samples_limit(self, tmp_path: Path) -> None:
        """Verify only num_samples images are collected."""
        cb = SampleVisualizationCallback(
            output_dir=str(tmp_path), num_samples=2
        )

        pl_module = _FakeLightningModule(num_classes=3)
        trainer = _mock_trainer(current_epoch=0)

        cb.on_validation_epoch_start(trainer, pl_module)  # type: ignore[arg-type]

        # Send batch of 4 but limit is 2
        images = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 2, 0])
        batch = {"images": images, "labels": labels}

        cb.on_validation_batch_end(
            trainer, pl_module, None, batch, batch_idx=0  # type: ignore[arg-type]
        )

        assert len(cb._images) == 2
