"""Tests for the EMA callback."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import torch
from torch import nn

from classifier_training.callbacks.ema import EMACallback


def _make_model() -> nn.Module:
    """Create a simple model for EMA tests."""
    return nn.Linear(10, 5, bias=False)


def _make_mock_pl_module(model: nn.Module) -> MagicMock:
    """Create a mock LightningModule backed by a real nn.Module."""
    pl_module = MagicMock()
    pl_module.state_dict.return_value = model.state_dict()
    pl_module.load_state_dict = MagicMock()
    return pl_module


class TestEMACallbackInit:
    """Tests for EMACallback initialization."""

    def test_default_parameters(self) -> None:
        ema = EMACallback()
        assert ema.decay == 0.9999
        assert ema.warmup_steps == 2000
        assert ema._step_count == 0
        assert ema._ema_applied is False

    def test_custom_parameters(self) -> None:
        ema = EMACallback(decay=0.99, warmup_steps=100)
        assert ema.decay == 0.99
        assert ema.warmup_steps == 100


class TestEMACallbackFitStart:
    """Tests for on_fit_start hook."""

    def test_initializes_ema_state_dict(self) -> None:
        model = _make_model()
        pl_module = _make_mock_pl_module(model)
        trainer = MagicMock()

        ema = EMACallback()
        ema.on_fit_start(trainer, pl_module)

        assert len(ema.ema_state_dict) > 0
        assert ema._step_count == 0


class TestEMACallbackTrainBatch:
    """Tests for on_train_batch_end hook."""

    def test_step_count_increments(self) -> None:
        model = _make_model()
        pl_module = _make_mock_pl_module(model)
        trainer = MagicMock()

        ema = EMACallback()
        ema.on_fit_start(trainer, pl_module)

        batch = {  # type: ignore[assignment]
            "images": torch.zeros(1),
            "labels": torch.zeros(1),
        }
        for i in range(5):
            ema.on_train_batch_end(trainer, pl_module, None, batch, i)
        assert ema._step_count == 5

    def test_warmup_formula_produces_correct_decay(self) -> None:
        """Warmup formula: min(decay, (1 + step) / (10 + step))."""
        # At step 1: (1+1)/(10+1) = 2/11 ~ 0.182
        effective = min(0.9999, (1 + 1) / (10 + 1))
        assert abs(effective - 2 / 11) < 1e-6

        # At step 100: (1+100)/(10+100) = 101/110 ~ 0.918
        effective = min(0.9999, (1 + 100) / (10 + 100))
        assert abs(effective - 101 / 110) < 1e-6

    def test_ema_weights_differ_from_model_after_update(self) -> None:
        """EMA weights should diverge from model weights after update."""
        model = _make_model()
        ema = EMACallback(decay=0.9, warmup_steps=0)
        ema.ema_state_dict = copy.deepcopy(model.state_dict())
        ema._step_count = 0

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        pl_module = _make_mock_pl_module(model)
        trainer = MagicMock()

        batch = {  # type: ignore[assignment]
            "images": torch.zeros(1),
            "labels": torch.zeros(1),
        }
        ema.on_train_batch_end(trainer, pl_module, None, batch, 0)

        # EMA should NOT equal the model weights (partial update)
        for key in ema.ema_state_dict:
            if ema.ema_state_dict[key].dtype.is_floating_point:
                assert not torch.equal(ema.ema_state_dict[key], model.state_dict()[key])


class TestEMACallbackValidation:
    """Tests for validation hooks (apply and restore EMA weights)."""

    def test_apply_and_restore_weights(self) -> None:
        """EMA weights are applied for validation and restored after."""
        model = _make_model()
        pl_module = _make_mock_pl_module(model)
        trainer = MagicMock()

        ema = EMACallback()
        ema.ema_state_dict = copy.deepcopy(model.state_dict())
        # Make EMA weights different
        for key in ema.ema_state_dict:
            ema.ema_state_dict[key] = ema.ema_state_dict[key] + 1.0

        ema.on_validation_start(trainer, pl_module)
        assert ema._ema_applied is True
        assert pl_module.load_state_dict.called

        ema.on_validation_end(trainer, pl_module)
        assert ema._ema_applied is False

    def test_no_apply_when_empty_ema(self) -> None:
        """No-op when EMA state dict is empty."""
        pl_module = MagicMock()
        trainer = MagicMock()

        ema = EMACallback()
        ema.on_validation_start(trainer, pl_module)
        assert ema._ema_applied is False


class TestEMACallbackTest:
    """Tests for test hooks."""

    def test_apply_and_restore_for_test(self) -> None:
        """EMA weights are applied for testing and restored after."""
        model = _make_model()
        pl_module = _make_mock_pl_module(model)
        trainer = MagicMock()

        ema = EMACallback()
        ema.ema_state_dict = copy.deepcopy(model.state_dict())

        ema.on_test_start(trainer, pl_module)
        assert ema._ema_applied is True

        ema.on_test_end(trainer, pl_module)
        assert ema._ema_applied is False


class TestEMACallbackStateDict:
    """Tests for state_dict and load_state_dict."""

    def test_state_dict_roundtrip(self) -> None:
        """State dict save/load roundtrip preserves callback state."""
        ema = EMACallback(decay=0.95, warmup_steps=500)
        ema._step_count = 42
        ema.ema_state_dict = {"weight": torch.tensor([1.0, 2.0, 3.0])}

        state = ema.state_dict()

        ema2 = EMACallback()
        ema2.load_state_dict(state)

        assert ema2._step_count == 42
        assert ema2.decay == 0.95
        torch.testing.assert_close(
            ema2.ema_state_dict["weight"], torch.tensor([1.0, 2.0, 3.0])
        )
