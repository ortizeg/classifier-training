"""Integration tests for the full callback stack.

Tests verify that:
1. All Hydra _target_ paths in default.yaml resolve to importable classes
2. A 2-epoch smoke training loop with multiple callbacks completes without error
3. EMA + ONNX export integration produces valid ONNX output
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import lightning as L
import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classifier_training.callbacks.confusion_matrix import ConfusionMatrixCallback
from classifier_training.callbacks.ema import EMACallback
from classifier_training.callbacks.model_info import ModelInfoCallback
from classifier_training.callbacks.onnx_export import ONNXExportCallback
from classifier_training.callbacks.plotting import TrainingHistoryCallback

NUM_CLASSES = 10
INPUT_SIZE = 32


class _DictDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset returning ClassificationBatch-style dicts."""

    def __init__(self, num_samples: int = 32) -> None:
        self.images = torch.randn(num_samples, 3, INPUT_SIZE, INPUT_SIZE)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"images": self.images[idx], "labels": self.labels[idx]}


class _MinimalModule(L.LightningModule):
    """Tiny LightningModule for integration smoke tests."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * INPUT_SIZE * INPUT_SIZE, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(x.flatten(1))

    def training_step(  # type: ignore[override]
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch["images"])
        loss = nn.functional.cross_entropy(logits, batch["labels"])
        self.log("train/loss", loss)
        return loss

    def validation_step(  # type: ignore[override]
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        logits = self(batch["images"])
        loss = nn.functional.cross_entropy(logits, batch["labels"])
        preds = logits.argmax(dim=1)
        acc = (preds == batch["labels"]).float().mean()
        self.log("val/loss", loss)
        self.log("val/acc_top1", acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _MockDataModule(L.LightningDataModule):
    """Mock datamodule providing save_labels_mapping for ONNXExportCallback."""

    def save_labels_mapping(self, save_path: Path) -> None:
        mapping = {str(i): f"class_{i}" for i in range(NUM_CLASSES)}
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(mapping))


def _make_dataloaders(
    batch_size: int = 16, num_samples: int = 32
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Create synthetic train and val dataloaders with dict batches."""
    ds = _DictDataset(num_samples=num_samples)
    train_dl: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        ds, batch_size=batch_size
    )
    val_dl: DataLoader[dict[str, torch.Tensor]] = DataLoader(ds, batch_size=batch_size)
    return train_dl, val_dl


class TestDefaultYamlTargets:
    """Verify all _target_ paths in default.yaml are importable."""

    def test_default_yaml_all_targets_importable(self) -> None:
        yaml_path = (
            Path(__file__).parent.parent
            / "src"
            / "classifier_training"
            / "conf"
            / "callbacks"
            / "default.yaml"
        )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert len(config) == 12, f"Expected 12 callbacks, got {len(config)}"

        for key, entry in config.items():
            target = entry.get("_target_")
            assert target is not None, f"Callback '{key}' missing _target_"

            # Split module path and class name
            parts = target.rsplit(".", 1)
            assert len(parts) == 2, f"Invalid _target_ format: {target}"
            module_path, class_name = parts

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            assert cls is not None, (
                f"Class '{class_name}' not found in module '{module_path}' "
                f"for callback '{key}'"
            )


class TestSmokeTrainingLoop:
    """Run a 2-epoch training loop with multiple callbacks."""

    def test_smoke_training_loop_with_callbacks(self, tmp_path: Path) -> None:
        model = _MinimalModule()
        train_dl, val_dl = _make_dataloaders()

        callbacks: list[L.Callback] = [
            EMACallback(decay=0.999, warmup_steps=5),
            ONNXExportCallback(
                output_dir=str(tmp_path),
                opset_version=17,
                input_height=INPUT_SIZE,
                input_width=INPUT_SIZE,
            ),
            ConfusionMatrixCallback(
                output_dir=str(tmp_path),
                num_classes=NUM_CLASSES,
            ),
            TrainingHistoryCallback(output_dir=str(tmp_path)),
            ModelInfoCallback(output_dir=str(tmp_path)),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(tmp_path / "checkpoints"),
                monitor="val/acc_top1",
                mode="max",
            ),
        ]

        trainer = L.Trainer(
            max_epochs=2,
            callbacks=callbacks,
            enable_checkpointing=True,
            logger=False,
            accelerator="cpu",
            devices=1,
            default_root_dir=str(tmp_path),
        )

        # Attach mock datamodule for labels_mapping support
        mock_dm = _MockDataModule()
        trainer.datamodule = mock_dm  # type: ignore[assignment]

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Checkpoint exists
        checkpoint_dir = tmp_path / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory not created"
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        assert len(ckpt_files) > 0, "No checkpoint files saved"

        # ONNX model exists
        onnx_path = tmp_path / "model.onnx"
        assert onnx_path.exists(), "model.onnx not created"

        # ONNX model output name is "logits"
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        outputs = session.get_outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "logits"

        # Confusion matrix PNG exists (epoch 0 or epoch 1)
        cm_files = list(tmp_path.rglob("confusion_matrix.png"))
        assert len(cm_files) > 0, "No confusion_matrix.png saved"

        # labels_mapping.json exists (written by ONNX export callback)
        labels_path = tmp_path / "labels_mapping.json"
        assert labels_path.exists(), "labels_mapping.json not created"
        labels_data = json.loads(labels_path.read_text())
        assert len(labels_data) == NUM_CLASSES


class TestOnnxExportUsesEmaWeights:
    """End-to-end test: EMA + ONNX export integration."""

    def test_onnx_export_uses_ema_weights(self, tmp_path: Path) -> None:
        model = _MinimalModule()
        train_dl, val_dl = _make_dataloaders()

        ema_cb = EMACallback(decay=0.999, warmup_steps=0)
        onnx_cb = ONNXExportCallback(
            output_dir=str(tmp_path),
            opset_version=17,
            input_height=INPUT_SIZE,
            input_width=INPUT_SIZE,
        )

        trainer = L.Trainer(
            max_epochs=2,
            callbacks=[ema_cb, onnx_cb],
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            devices=1,
            default_root_dir=str(tmp_path),
        )

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # ONNX file exists
        onnx_path = tmp_path / "model.onnx"
        assert onnx_path.exists()

        # Load and run inference
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy_input = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        result = session.run(None, {"input": dummy_input})

        # Output shape is (1, NUM_CLASSES)
        assert result[0].shape == (1, NUM_CLASSES)
