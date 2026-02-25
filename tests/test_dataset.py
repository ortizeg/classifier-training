"""Unit tests for JerseyNumberDataset."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
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
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
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

    def test_metadata_none_when_absent(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        """Without metadata in JSONL, all entries should be None."""
        ds = JerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        assert len(ds.metadata) == len(ds.samples)
        assert all(m is None for m in ds.metadata)

    def test_metadata_loaded_when_present(
        self, tmp_path: Path, class_to_idx: dict[str, int]
    ) -> None:
        """Records with metadata field should populate dataset.metadata."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        img.save(split_dir / "img.jpg")
        meta = {"jersey_color": "white", "number_color": "red", "border": True}
        record = json.dumps(
            {
                "image": "img.jpg",
                "prefix": "Read the number.",
                "suffix": "0",
                "metadata": meta,
            }
        )
        (split_dir / "annotations.jsonl").write_text(record + "\n")
        ds = JerseyNumberDataset(split_dir, class_to_idx)
        assert len(ds.metadata) == 1
        assert ds.metadata[0] == meta

    def test_metadata_mixed_present_and_absent(
        self, tmp_path: Path, class_to_idx: dict[str, int]
    ) -> None:
        """Mix of records with and without metadata."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        for i in range(2):
            img = Image.new("RGB", (224, 224), color=(100, 100, 100))
            img.save(split_dir / f"img_{i}.jpg")
        lines = [
            json.dumps(
                {
                    "image": "img_0.jpg",
                    "prefix": "Read the number.",
                    "suffix": "0",
                    "metadata": {
                        "jersey_color": "blue",
                        "number_color": "white",
                        "border": False,
                    },
                }
            ),
            json.dumps(
                {
                    "image": "img_1.jpg",
                    "prefix": "Read the number.",
                    "suffix": "1",
                }
            ),
        ]
        (split_dir / "annotations.jsonl").write_text("\n".join(lines) + "\n")
        ds = JerseyNumberDataset(split_dir, class_to_idx)
        assert len(ds.metadata) == 2
        assert ds.metadata[0] is not None
        assert ds.metadata[0]["jersey_color"] == "blue"
        assert ds.metadata[1] is None
