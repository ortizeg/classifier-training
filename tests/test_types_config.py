"""Unit tests for classifier_training.types and classifier_training.config."""

import pytest
import torch
from pydantic import ValidationError

from classifier_training.config import DataModuleConfig
from classifier_training.types import ClassificationBatch

# Dummy path: satisfies data_root: str type; never accessed as a real filesystem path.
_DUMMY_ROOT = "/data/test"


class TestDataModuleConfig:
    def test_defaults(self) -> None:
        cfg = DataModuleConfig(data_root=_DUMMY_ROOT)
        assert cfg.data_root == _DUMMY_ROOT
        assert cfg.batch_size == 32
        assert cfg.num_workers == 4
        assert cfg.pin_memory is True
        assert cfg.persistent_workers is True
        assert cfg.image_size == 224

    def test_frozen_raises_on_mutation(self) -> None:
        cfg = DataModuleConfig(data_root=_DUMMY_ROOT)
        with pytest.raises(ValidationError):
            cfg.batch_size = 64  # type: ignore[misc]

    def test_persistent_workers_auto_corrected_when_num_workers_zero(self) -> None:
        cfg = DataModuleConfig(
            data_root=_DUMMY_ROOT, num_workers=0, persistent_workers=True
        )
        assert cfg.persistent_workers is False, (
            "persistent_workers must be False when num_workers=0"
        )

    def test_persistent_workers_preserved_when_num_workers_nonzero(self) -> None:
        cfg = DataModuleConfig(
            data_root=_DUMMY_ROOT, num_workers=4, persistent_workers=True
        )
        assert cfg.persistent_workers is True

    def test_persistent_workers_false_stays_false(self) -> None:
        cfg = DataModuleConfig(
            data_root=_DUMMY_ROOT, num_workers=4, persistent_workers=False
        )
        assert cfg.persistent_workers is False

    def test_custom_values(self) -> None:
        cfg = DataModuleConfig(
            data_root="/data/jersey",
            batch_size=64,
            num_workers=8,
            pin_memory=False,
            image_size=256,
        )
        assert cfg.batch_size == 64
        assert cfg.num_workers == 8
        assert cfg.pin_memory is False
        assert cfg.image_size == 256


class TestClassificationBatchType:
    def test_typed_dict_keys(self) -> None:
        """ClassificationBatch has exactly 'images' and 'labels' keys."""
        batch: ClassificationBatch = {
            "images": torch.zeros(4, 3, 224, 224),
            "labels": torch.zeros(4, dtype=torch.long),
        }
        assert "images" in batch
        assert "labels" in batch
        assert batch["images"].shape == (4, 3, 224, 224)
        assert batch["labels"].shape == (4,)
