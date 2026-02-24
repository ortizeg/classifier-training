"""Unit tests for SmolVLM2ClassificationModel."""

from __future__ import annotations

import torch

from classifier_training.models.smolvlm2 import SmolVLM2ClassificationModel


class TestSmolVLM2ClassificationModel:
    def test_construction_defaults(self) -> None:
        model = SmolVLM2ClassificationModel()
        assert model.num_classes == 43
        assert model.hparams["learning_rate"] == 2e-4
        assert model.hparams["lora_r"] == 16
        assert model.hparams["lora_alpha"] == 32

    def test_construction_custom_params(self) -> None:
        model = SmolVLM2ClassificationModel(
            num_classes=10,
            learning_rate=1e-3,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        assert model.num_classes == 10
        assert model.hparams["learning_rate"] == 1e-3
        assert model.hparams["lora_r"] == 8

    def test_set_class_weights_is_noop(self) -> None:
        model = SmolVLM2ClassificationModel()
        # Should not raise
        model.set_class_weights(torch.ones(43))

    def test_set_class_mappings(self) -> None:
        model = SmolVLM2ClassificationModel()
        class_to_idx = {"0": 0, "1": 1, "23": 2}
        idx_to_class = {0: "0", 1: "1", 2: "23"}
        model.set_class_mappings(class_to_idx, idx_to_class)
        assert model._class_to_idx == class_to_idx
        assert model._idx_to_class == idx_to_class

    def test_num_classes_property(self) -> None:
        model = SmolVLM2ClassificationModel(num_classes=10)
        assert model.num_classes == 10

    def test_hparams_saved(self) -> None:
        model = SmolVLM2ClassificationModel(
            model_name="test-model",
            num_classes=5,
            learning_rate=3e-4,
            weight_decay=0.02,
            lora_r=32,
        )
        assert model.hparams["model_name"] == "test-model"
        assert model.hparams["num_classes"] == 5
        assert model.hparams["weight_decay"] == 0.02
        assert model.hparams["lora_r"] == 32

    def test_validation_counters_reset(self) -> None:
        model = SmolVLM2ClassificationModel()
        model._val_correct = 5
        model._val_total = 10
        model.on_validation_epoch_end()
        assert model._val_correct == 0
        assert model._val_total == 0
