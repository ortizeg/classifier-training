"""Unit tests for JerseyNumberDataset."""

from pathlib import Path

import pytest
import torch
from torchvision.transforms import v2

from classifier_training.data.dataset import JerseyNumberDataset


@pytest.fixture()
def class_to_idx() -> dict[str, int]:
    return {"0": 0, "1": 1, "2": 2}


class TestJerseyNumberDataset:
    def test_len_equals_annotation_rows_not_image_count(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        """len(dataset) must equal annotation rows, not unique image count.

        The real dataset has 2930 annotation rows but only 2891 unique images.
        Our fixture: 3 classes x 2 images/class = 6 annotation rows per split.
        """
        ds = JerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        # conftest creates 3 classes * 2 images each = 6 annotation rows
        assert len(ds) == 6

    def test_getitem_returns_tensor_and_int(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = JerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        # Without transform, returns PIL Image — but we confirm it's not None
        _img, label = item
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_tensor(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        ds = JerseyNumberDataset(
            tmp_dataset_dir / "train", class_to_idx, transform=transform
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_label_indices_within_range(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = JerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        for _, label in ds.samples:
            assert 0 <= label < len(class_to_idx)

    def test_all_splits_load(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        for split in ("train", "valid", "test"):
            ds = JerseyNumberDataset(tmp_dataset_dir / split, class_to_idx)
            assert len(ds) == 6  # 3 classes * 2 images per conftest fixture

    def test_class_to_idx_stored_on_instance(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = JerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        assert ds.class_to_idx == class_to_idx
