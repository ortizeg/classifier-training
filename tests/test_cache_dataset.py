"""Tests for CacheDataset wrapper."""

from pathlib import Path

import pytest
import torch
from torchvision.transforms import v2

from classifier_training.config import DataModuleConfig
from classifier_training.data import ImageFolderDataModule
from classifier_training.data.cache_dataset import CacheDataset
from classifier_training.data.dataset import JerseyNumberDataset


@pytest.fixture()
def train_dataset(tmp_dataset_dir: Path) -> JerseyNumberDataset:
    """Raw train dataset without transforms for cache testing."""
    class_to_idx = {"0": 0, "1": 1, "2": 2}
    return JerseyNumberDataset(
        root=tmp_dataset_dir / "train",
        class_to_idx=class_to_idx,
        transform=None,
    )


@pytest.fixture()
def simple_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(64),
            v2.CenterCrop(64),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


class TestCacheDatasetRAM:
    def test_ram_cache_len_matches_dataset(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="ram")
        assert len(cached) == len(train_dataset)

    def test_ram_cache_returns_pil_without_transforms(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="ram")
        img, label = cached[0]
        # Without transforms, returns PIL Image (deepcopy of cached)
        from PIL import Image

        assert isinstance(img, Image.Image)
        assert isinstance(label, int)

    def test_ram_cache_applies_transforms(
        self,
        train_dataset: JerseyNumberDataset,
        simple_transform: v2.Compose,
    ) -> None:
        cached = CacheDataset(
            train_dataset, cache_type="ram", transforms=simple_transform
        )
        img, label = cached[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(label, int)

    def test_ram_cache_preserves_labels(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="ram")
        for i in range(len(cached)):
            _, cached_label = cached[i]
            _, original_label = train_dataset.samples[i]
            assert cached_label == original_label

    def test_ram_cache_samples_proxy(self, train_dataset: JerseyNumberDataset) -> None:
        """CacheDataset.samples should proxy to underlying dataset.samples."""
        cached = CacheDataset(train_dataset, cache_type="ram")
        assert cached.samples is train_dataset.samples

    def test_ram_cache_class_to_idx_proxy(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="ram")
        assert cached.class_to_idx == train_dataset.class_to_idx


class TestCacheDatasetDisk:
    def test_disk_cache_len_matches_dataset(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="disk")
        assert len(cached) == len(train_dataset)

    def test_disk_cache_applies_transforms(
        self,
        train_dataset: JerseyNumberDataset,
        simple_transform: v2.Compose,
    ) -> None:
        cached = CacheDataset(
            train_dataset, cache_type="disk", transforms=simple_transform
        )
        img, label = cached[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(label, int)

    def test_disk_cache_preserves_labels(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="disk")
        for i in range(len(cached)):
            _, cached_label = cached[i]
            _, original_label = train_dataset.samples[i]
            assert cached_label == original_label

    def test_disk_cache_creates_db_file(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        cached = CacheDataset(train_dataset, cache_type="disk")
        assert cached._db_path is not None
        assert cached._db_path.exists()

    def test_disk_cache_reuses_existing(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        """Second instantiation should hit the existing cache."""
        c1 = CacheDataset(train_dataset, cache_type="disk")
        db_path = c1._db_path
        c1._close()

        # Second instance should detect existing cache
        c2 = CacheDataset(train_dataset, cache_type="disk")
        assert c2._db_path == db_path
        assert len(c2) == len(train_dataset)
        c2._close()


class TestCacheDatasetAuto:
    def test_auto_selects_ram_for_small_dataset(
        self, train_dataset: JerseyNumberDataset
    ) -> None:
        """Small synthetic dataset should always fit in RAM."""
        cached = CacheDataset(train_dataset, cache_type="auto")
        assert cached._cache_type == "ram"


class TestDataModuleCacheIntegration:
    def test_cache_enabled_produces_valid_batches(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg, use_cache=True, cache_type="ram")
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "images" in batch
        assert "labels" in batch
        assert isinstance(batch["images"], torch.Tensor)
        assert batch["images"].dtype == torch.float32
        assert batch["images"].shape[1:] == (3, 224, 224)

    def test_cache_disabled_by_default(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        assert not isinstance(dm._train_dataset, CacheDataset)

    def test_cache_enabled_wraps_dataset(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg, use_cache=True, cache_type="ram")
        dm.setup("fit")
        assert isinstance(dm._train_dataset, CacheDataset)
        assert isinstance(dm._val_dataset, CacheDataset)

    def test_cache_with_sampler_auto(self, tmp_dataset_dir: Path) -> None:
        """Cache + auto sampler should work together."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(
            cfg,
            use_cache=True,
            cache_type="ram",
            sampler={"mode": "auto"},
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["images"].shape[1:] == (3, 224, 224)
