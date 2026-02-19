"""Tests for SamplerConfig and build_sampler factory."""

from pathlib import Path

import pytest

from classifier_training.config import DataModuleConfig
from classifier_training.data import ImageFolderDataModule
from classifier_training.data.sampler import (
    SamplerConfig,
    TrackingWeightedRandomSampler,
    build_sampler,
)


class TestSamplerConfig:
    def test_defaults_to_disabled(self) -> None:
        cfg = SamplerConfig()
        assert cfg.mode == "disabled"
        assert cfg.class_weights is None
        assert cfg.num_samples is None
        assert cfg.replacement is True

    def test_auto_mode(self) -> None:
        cfg = SamplerConfig(mode="auto")
        assert cfg.mode == "auto"

    def test_manual_mode_with_weights(self) -> None:
        cfg = SamplerConfig(mode="manual", class_weights={"0": 1.0, "1": 2.0})
        assert cfg.mode == "manual"
        assert cfg.class_weights == {"0": 1.0, "1": 2.0}

    def test_frozen(self) -> None:
        cfg = SamplerConfig()
        with pytest.raises(Exception):
            cfg.mode = "auto"  # type: ignore[misc]


class TestBuildSampler:
    def test_disabled_returns_none(self) -> None:
        cfg = SamplerConfig(mode="disabled")
        result = build_sampler(
            cfg, labels=[0, 1, 2], class_to_idx={"a": 0, "b": 1, "c": 2}
        )
        assert result is None

    def test_auto_returns_tracking_sampler(self) -> None:
        cfg = SamplerConfig(mode="auto")
        labels = [0, 0, 0, 1, 1, 2]
        class_to_idx = {"a": 0, "b": 1, "c": 2}
        result = build_sampler(cfg, labels, class_to_idx)
        assert isinstance(result, TrackingWeightedRandomSampler)
        assert result.num_samples == len(labels)

    def test_auto_custom_num_samples(self) -> None:
        cfg = SamplerConfig(mode="auto", num_samples=100)
        labels = [0, 1, 2]
        class_to_idx = {"a": 0, "b": 1, "c": 2}
        result = build_sampler(cfg, labels, class_to_idx)
        assert result is not None
        assert result.num_samples == 100

    def test_manual_returns_tracking_sampler(self) -> None:
        cfg = SamplerConfig(
            mode="manual", class_weights={"a": 1.0, "b": 5.0, "c": 10.0}
        )
        labels = [0, 0, 1, 2]
        class_to_idx = {"a": 0, "b": 1, "c": 2}
        result = build_sampler(cfg, labels, class_to_idx)
        assert isinstance(result, TrackingWeightedRandomSampler)

    def test_manual_missing_weights_raises(self) -> None:
        cfg = SamplerConfig(mode="manual")
        with pytest.raises(ValueError, match="manual mode requires class_weights"):
            build_sampler(cfg, labels=[0], class_to_idx={"a": 0})

    def test_manual_unknown_class_raises(self) -> None:
        cfg = SamplerConfig(mode="manual", class_weights={"unknown": 1.0})
        with pytest.raises(ValueError, match="not in class_to_idx"):
            build_sampler(cfg, labels=[0], class_to_idx={"a": 0})


class TestDataModuleSamplerIntegration:
    def test_disabled_sampler_uses_shuffle(self, tmp_dataset_dir: Path) -> None:
        """When sampler is disabled, train_dataloader should use shuffle=True."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg, sampler={"mode": "disabled"})
        dm.setup("fit")
        loader = dm.train_dataloader()
        # No WeightedRandomSampler — should use RandomSampler (shuffle=True)
        assert not isinstance(loader.sampler, TrackingWeightedRandomSampler)

    def test_auto_sampler_uses_weighted_sampler(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg, sampler={"mode": "auto"})
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, TrackingWeightedRandomSampler)

    def test_default_sampler_is_disabled(self, tmp_dataset_dir: Path) -> None:
        """Default (no sampler kwarg) should behave as disabled."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert not isinstance(loader.sampler, TrackingWeightedRandomSampler)
