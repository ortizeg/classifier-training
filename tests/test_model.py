"""Tests for ResNet classification models."""

from __future__ import annotations

import classifier_training.models  # noqa: F401
import lightning as L
import pytest
import torch
from hydra.core.config_store import ConfigStore

from classifier_training.models import (
    ResNet18ClassificationModel,
    ResNet34ClassificationModel,
    ResNet50ClassificationModel,
)
from classifier_training.types import ClassificationBatch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def batch_3cls() -> ClassificationBatch:
    """Small 3-class batch: B=4, 3-channel 64x64 images."""
    return {
        "images": torch.randn(4, 3, 64, 64),
        "labels": torch.randint(0, 3, (4,)),
    }


@pytest.fixture()
def batch_43cls() -> ClassificationBatch:
    """Full 43-class batch: B=4, 3-channel 224x224 images."""
    return {
        "images": torch.randn(4, 3, 224, 224),
        "labels": torch.randint(0, 43, (4,)),
    }


@pytest.fixture()
def resnet18_3cls() -> ResNet18ClassificationModel:
    return ResNet18ClassificationModel(num_classes=3, pretrained=False)


@pytest.fixture()
def resnet34_3cls() -> ResNet34ClassificationModel:
    return ResNet34ClassificationModel(num_classes=3, pretrained=False)


@pytest.fixture()
def resnet50_3cls() -> ResNet50ClassificationModel:
    return ResNet50ClassificationModel(num_classes=3, pretrained=False)


@pytest.fixture()
def resnet18_43cls() -> ResNet18ClassificationModel:
    return ResNet18ClassificationModel(num_classes=43, pretrained=False)


@pytest.fixture()
def resnet50_43cls() -> ResNet50ClassificationModel:
    return ResNet50ClassificationModel(num_classes=43, pretrained=False)


# ---------------------------------------------------------------------------
# BaseClassificationModel: metric guard
# ---------------------------------------------------------------------------


class TestBaseMetricGuard:
    def test_top5_guard_3_classes(self) -> None:
        """top_k_5 = min(5, 3) = 3."""
        m = ResNet18ClassificationModel(
            num_classes=3, pretrained=False
        )
        assert m.val_top5.top_k == 3

    def test_top5_guard_43_classes(self) -> None:
        """top_k_5 = min(5, 43) = 5."""
        m = ResNet18ClassificationModel(
            num_classes=43, pretrained=False
        )
        assert m.val_top5.top_k == 5

    def test_class_weights_buffer_shape(
        self, resnet18_3cls: ResNet18ClassificationModel
    ) -> None:
        """class_weights defaults to ones with correct shape."""
        assert resnet18_3cls.class_weights.shape == (3,)
        assert torch.allclose(
            resnet18_3cls.class_weights, torch.ones(3)
        )

    def test_set_class_weights_updates_buffer(
        self, resnet18_3cls: ResNet18ClassificationModel
    ) -> None:
        """set_class_weights copies tensor and rebuilds loss_fn."""
        new_weights = torch.tensor([1.0, 2.0, 3.0])
        resnet18_3cls.set_class_weights(new_weights)
        assert torch.allclose(
            resnet18_3cls.class_weights, new_weights
        )

    def test_hparams_saved(
        self, resnet18_3cls: ResNet18ClassificationModel
    ) -> None:
        """save_hyperparameters stores scalar params."""
        assert resnet18_3cls.hparams["num_classes"] == 3
        assert resnet18_3cls.hparams["learning_rate"] == pytest.approx(
            1e-4
        )
        assert "class_weights" not in resnet18_3cls.hparams


# ---------------------------------------------------------------------------
# ResNet18: forward pass and loss
# ---------------------------------------------------------------------------


class TestResNet18ForwardPass:
    def test_forward_shape_3cls(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet18_3cls(batch_3cls["images"])
        assert logits.shape == (4, 3)

    def test_forward_shape_43cls(
        self,
        resnet18_43cls: ResNet18ClassificationModel,
        batch_43cls: ClassificationBatch,
    ) -> None:
        logits = resnet18_43cls(batch_43cls["images"])
        assert logits.shape == (4, 43)

    def test_loss_is_finite(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet18_3cls(batch_3cls["images"])
        loss = resnet18_3cls.loss_fn(logits, batch_3cls["labels"])
        assert torch.isfinite(loss).all()

    def test_forward_produces_logits(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        """Forward pass + loss produces a finite scalar loss."""
        logits = resnet18_3cls(batch_3cls["images"])
        loss = resnet18_3cls.loss_fn(logits, batch_3cls["labels"])
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ResNet34: forward pass and loss
# ---------------------------------------------------------------------------


class TestResNet34ForwardPass:
    def test_forward_shape_3cls(
        self,
        resnet34_3cls: ResNet34ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet34_3cls(batch_3cls["images"])
        assert logits.shape == (4, 3)

    def test_loss_is_finite(
        self,
        resnet34_3cls: ResNet34ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet34_3cls(batch_3cls["images"])
        loss = resnet34_3cls.loss_fn(logits, batch_3cls["labels"])
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# ResNet50: forward pass and loss
# ---------------------------------------------------------------------------


class TestResNet50ForwardPass:
    def test_forward_shape_3cls(
        self,
        resnet50_3cls: ResNet50ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet50_3cls(batch_3cls["images"])
        assert logits.shape == (4, 3)

    def test_forward_shape_43cls(
        self,
        resnet50_43cls: ResNet50ClassificationModel,
        batch_43cls: ClassificationBatch,
    ) -> None:
        logits = resnet50_43cls(batch_43cls["images"])
        assert logits.shape == (4, 43)

    def test_loss_is_finite(
        self,
        resnet50_3cls: ResNet50ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet50_3cls(batch_3cls["images"])
        loss = resnet50_3cls.loss_fn(logits, batch_3cls["labels"])
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Pattern A metrics: update/compute/reset discipline
# ---------------------------------------------------------------------------


class TestPatternAMetrics:
    """Verify Pattern A: update in step, compute+log+reset in epoch_end."""

    def test_val_top1_no_nan_after_update(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        """After update(), compute() returns finite value."""
        logits = resnet18_3cls(batch_3cls["images"])
        resnet18_3cls.val_top1.update(logits, batch_3cls["labels"])
        result = resnet18_3cls.val_top1.compute()
        assert torch.isfinite(result), f"val_top1 NaN: {result}"
        resnet18_3cls.val_top1.reset()

    def test_val_top5_no_nan_after_update(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        """val_top5 uses top_k=min(5,3)=3."""
        logits = resnet18_3cls(batch_3cls["images"])
        resnet18_3cls.val_top5.update(logits, batch_3cls["labels"])
        result = resnet18_3cls.val_top5.compute()
        assert torch.isfinite(result), f"val_top5 NaN: {result}"
        resnet18_3cls.val_top5.reset()

    def test_val_per_cls_shape_and_finite(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        """val_per_cls returns shape (num_classes,) with finite vals."""
        logits = resnet18_3cls(batch_3cls["images"])
        resnet18_3cls.val_per_cls.update(logits, batch_3cls["labels"])
        result = resnet18_3cls.val_per_cls.compute()
        assert result.shape == (3,), f"per_cls shape: {result.shape}"
        assert torch.isfinite(result).all(), f"per_cls NaN: {result}"
        resnet18_3cls.val_per_cls.reset()

    def test_reset_clears_state(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        """After reset(), second cycle gives consistent result."""
        logits = resnet18_3cls(batch_3cls["images"])
        # First cycle
        resnet18_3cls.val_top1.update(logits, batch_3cls["labels"])
        result1 = resnet18_3cls.val_top1.compute()
        resnet18_3cls.val_top1.reset()
        # Second cycle -- same data, should match
        resnet18_3cls.val_top1.update(logits, batch_3cls["labels"])
        result2 = resnet18_3cls.val_top1.compute()
        resnet18_3cls.val_top1.reset()
        assert torch.allclose(result1, result2), (
            f"reset() broken: {result1} != {result2}"
        )

    def test_train_top1_update_compute_reset(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
        batch_3cls: ClassificationBatch,
    ) -> None:
        logits = resnet18_3cls(batch_3cls["images"])
        resnet18_3cls.train_top1.update(logits, batch_3cls["labels"])
        result = resnet18_3cls.train_top1.compute()
        assert torch.isfinite(result)
        resnet18_3cls.train_top1.reset()


# ---------------------------------------------------------------------------
# Optimizer and scheduler structure
# ---------------------------------------------------------------------------


class TestOptimizerScheduler:
    def test_configure_optimizers_structure(
        self,
        resnet18_3cls: ResNet18ClassificationModel,
    ) -> None:
        """configure_optimizers returns correct structure."""
        trainer = L.Trainer(
            max_epochs=20,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        resnet18_3cls.trainer = trainer  # type: ignore[assignment]
        result = resnet18_3cls.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], torch.optim.AdamW)
        assert result["lr_scheduler"]["interval"] == "epoch"

    def test_optimizer_uses_adamw(
        self, resnet18_43cls: ResNet18ClassificationModel
    ) -> None:
        """AdamW is the optimizer; initial lr is set by hparams."""
        trainer = L.Trainer(
            max_epochs=20,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        resnet18_43cls.trainer = trainer  # type: ignore[assignment]
        result = resnet18_43cls.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)
        # SequentialLR(LinearLR warmup) modifies the param group lr
        # at step 0, so check the hparams-stored learning_rate instead
        assert resnet18_43cls.hparams["learning_rate"] == pytest.approx(
            1e-4
        )


# ---------------------------------------------------------------------------
# Hydra ConfigStore registration
# ---------------------------------------------------------------------------


class TestHydraRegistration:
    def test_resnet18_registered(self) -> None:
        """@register stores ResNet18 in ConfigStore model."""
        cs = ConfigStore.instance()
        models_group = cs.repo.get("model", {})
        names = [k.replace(".yaml", "") for k in models_group]
        assert "resnet18" in names, (
            f"resnet18 not in ConfigStore model: {names}"
        )

    def test_resnet34_registered(self) -> None:
        """@register stores ResNet34 in ConfigStore model."""
        cs = ConfigStore.instance()
        models_group = cs.repo.get("model", {})
        names = [k.replace(".yaml", "") for k in models_group]
        assert "resnet34" in names, (
            f"resnet34 not in ConfigStore model: {names}"
        )

    def test_resnet50_registered(self) -> None:
        """@register stores ResNet50 in ConfigStore model."""
        cs = ConfigStore.instance()
        models_group = cs.repo.get("model", {})
        names = [k.replace(".yaml", "") for k in models_group]
        assert "resnet50" in names, (
            f"resnet50 not in ConfigStore model: {names}"
        )
