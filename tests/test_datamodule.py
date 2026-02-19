"""Tests for ImageFolderDataModule."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler

from classifier_training.config import DataModuleConfig
from classifier_training.data import ImageFolderDataModule
from classifier_training.data.datamodule import IMAGENET_MEAN, IMAGENET_STD

REAL_DATASET = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-jersey-numbers-ocr"
)


# ---------------------------------------------------------------------------
# Synthetic dataset tests (fast, no real data dependency)
# ---------------------------------------------------------------------------


class TestImageFolderDataModuleSynthetic:
    def test_import_succeeds(self) -> None:
        """Phase 1 success criterion 1: importable without error."""
        import classifier_training.data as data_pkg

        assert hasattr(data_pkg, "ImageFolderDataModule")

    def test_setup_fit_creates_train_and_val_datasets(
        self, tmp_dataset_dir: Path
    ) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert dm._test_dataset is None  # not setup for test stage

    def test_setup_test_creates_test_dataset(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("test")
        assert dm._test_dataset is not None
        assert dm._train_dataset is None

    def test_train_dataloader_default_disabled_uses_shuffle(
        self, tmp_dataset_dir: Path
    ) -> None:
        """Default sampler mode is disabled — DataLoader uses shuffle=True."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, RandomSampler)

    def test_train_dataloader_auto_sampler_uses_weighted_sampler(
        self, tmp_dataset_dir: Path
    ) -> None:
        """When sampler mode is auto, WeightedRandomSampler must be used."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg, sampler={"mode": "auto"})
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert loader.sampler is not None
        assert isinstance(loader.sampler, WeightedRandomSampler)

    def test_val_dataloader_has_no_sampler(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        loader = dm.val_dataloader()
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_val_batch_is_deterministic(self, tmp_dataset_dir: Path) -> None:
        """Val transforms must not include augmentation.

        Two forward passes over the same val sample must produce identical tensors.
        This verifies DATA-03: strict transform separation.
        """
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        assert dm._val_dataset is not None
        img1, lbl1 = dm._val_dataset[0]
        img2, lbl2 = dm._val_dataset[0]
        assert isinstance(img1, torch.Tensor)
        assert torch.allclose(img1, img2), (
            "Val transform is not deterministic — augmentation leak"
        )
        assert lbl1 == lbl2

    def test_class_to_idx_built_from_train_only(self, tmp_dataset_dir: Path) -> None:
        """class_to_idx must be derived from train split, not val/test."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        # Access before setup() — lazy build from train annotations
        c2i = dm.class_to_idx
        # conftest fixture has 3 classes: "0", "1", "2"
        assert c2i == {"0": 0, "1": 1, "2": 2}

    def test_class_to_idx_alphabetically_sorted(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        c2i = dm.class_to_idx
        keys = list(c2i.keys())
        assert keys == sorted(keys), "class_to_idx keys must be alphabetically sorted"
        indices = list(c2i.values())
        assert indices == list(range(len(indices))), "Indices must be 0..N-1"

    def test_save_labels_mapping_writes_json(
        self, tmp_dataset_dir: Path, tmp_path: Path
    ) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        out_path = tmp_path / "exports" / "labels_mapping.json"
        dm.save_labels_mapping(out_path)
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert "class_to_idx" in data
        assert "idx_to_class" in data
        assert "num_classes" in data
        assert "normalization" in data
        assert data["normalization"]["mean"] == IMAGENET_MEAN
        assert data["normalization"]["std"] == IMAGENET_STD
        assert data["num_classes"] == 3  # conftest fixture has 3 classes
        assert data["class_to_idx"] == {"0": 0, "1": 1, "2": 2}

    def test_num_classes_property(self, tmp_dataset_dir: Path) -> None:
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        assert dm.num_classes == 3

    def test_get_class_weights_shape(self, tmp_dataset_dir: Path) -> None:
        """Class weights tensor must have shape (num_classes,)."""
        cfg = DataModuleConfig(data_root=str(tmp_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        weights = dm.get_class_weights()
        assert weights.shape == (3,)
        assert (weights > 0).all(), "All class weights must be positive"


# ---------------------------------------------------------------------------
# Real dataset integration tests (marked integration — skipped in fast runs)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_DATASET.exists(),
    reason=f"Real dataset not found at {REAL_DATASET}",
)
class TestImageFolderDataModuleRealDataset:
    """Integration tests against the real basketball-jersey-numbers-ocr dataset.

    Verifies Phase 1 success criteria 4 and 5 against actual data.
    These tests are slow (disk I/O) but required for Phase 1 sign-off.
    """

    def _make_dm(self) -> ImageFolderDataModule:
        cfg = DataModuleConfig(data_root=str(REAL_DATASET), num_workers=0)
        return ImageFolderDataModule(cfg)

    def test_class_to_idx_has_43_classes(self) -> None:
        """Phase 1 success criterion 5: all 43 classes present."""
        dm = self._make_dm()
        assert dm.num_classes == 43, (
            f"Expected 43 classes, got {dm.num_classes}. "
            "Check that '' (empty string) is included."
        )

    def test_empty_string_class_at_index_zero(self) -> None:
        """'' class must sort first (index 0) — alphabetical sort invariant."""
        dm = self._make_dm()
        assert dm.class_to_idx[""] == 0, (
            "Empty-string class must be at index 0 (sorts before all numeric strings)"
        )

    def test_train_dataset_len_equals_annotation_rows(self) -> None:
        """len(train_dataset) must be 2930 (annotation rows), not 2891 (images)."""
        dm = self._make_dm()
        dm.setup("fit")
        assert dm._train_dataset is not None
        assert len(dm._train_dataset) == 2930, (
            f"Expected 2930 annotation rows, got {len(dm._train_dataset)}. "
            "Do not deduplicate by image filename."
        )

    def test_val_dataset_len_equals_annotation_rows(self) -> None:
        dm = self._make_dm()
        dm.setup("fit")
        assert dm._val_dataset is not None
        assert len(dm._val_dataset) == 372

    def test_test_dataset_len_equals_annotation_rows(self) -> None:
        dm = self._make_dm()
        dm.setup("test")
        assert dm._test_dataset is not None
        assert len(dm._test_dataset) == 365

    def test_save_labels_mapping_43_classes(self, tmp_path: Path) -> None:
        """Phase 1 success criterion 5: labels_mapping.json has 43 classes."""
        dm = self._make_dm()
        out = tmp_path / "labels_mapping.json"
        dm.save_labels_mapping(out)
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert data["num_classes"] == 43
        assert len(data["class_to_idx"]) == 43
        assert len(data["idx_to_class"]) == 43
        assert data["class_to_idx"][""] == 0

    def test_train_dataloader_batch_shape(self) -> None:
        """Train batch must be ClassificationBatch dict with (B, 3, 224, 224) float32."""
        dm = self._make_dm()
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        assert "images" in batch and "labels" in batch
        assert isinstance(batch["images"], torch.Tensor)
        assert batch["images"].dtype == torch.float32
        assert batch["images"].shape[1:] == (3, 224, 224)
        assert isinstance(batch["labels"], torch.Tensor)

    def test_val_dataloader_batch_shape(self) -> None:
        dm = self._make_dm()
        dm.setup("fit")
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        assert batch["images"].shape[1:] == (3, 224, 224)
        assert batch["images"].dtype == torch.float32

    def test_class_weights_shape_and_positivity(self) -> None:
        dm = self._make_dm()
        dm.setup("fit")
        weights = dm.get_class_weights()
        assert weights.shape == (43,)
        assert (weights > 0).all()
