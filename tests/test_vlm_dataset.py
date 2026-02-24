"""Unit tests for VLMJerseyNumberDataset."""

from pathlib import Path

import pytest
from PIL import Image

from classifier_training.data.vlm_dataset import VLMJerseyNumberDataset
from classifier_training.inference.vlm_inferencer import VLM_PROMPT


@pytest.fixture()
def class_to_idx() -> dict[str, int]:
    return {"0": 0, "1": 1, "2": 2}


class TestVLMJerseyNumberDataset:
    def test_len_matches_annotation_rows(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = VLMJerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        assert len(ds) == 6  # 3 classes * 2 images

    def test_getitem_returns_image_prompt_answer(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = VLMJerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        image, prompt, answer = ds[0]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert isinstance(answer, str)

    def test_default_prompt_is_vlm_prompt(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = VLMJerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        _, prompt, _ = ds[0]
        assert prompt == VLM_PROMPT

    def test_custom_prompt(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        custom = "What number?"
        ds = VLMJerseyNumberDataset(
            tmp_dataset_dir / "train", class_to_idx, prompt=custom
        )
        _, prompt, _ = ds[0]
        assert prompt == custom

    def test_answer_is_class_label_string(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = VLMJerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        answers = set()
        for i in range(len(ds)):
            _, _, answer = ds[i]
            answers.add(answer)
        assert answers == {"0", "1", "2"}

    def test_all_splits_load(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        for split in ("train", "valid", "test"):
            ds = VLMJerseyNumberDataset(tmp_dataset_dir / split, class_to_idx)
            assert len(ds) == 6

    def test_no_transforms_applied(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        """VLM dataset returns PIL Images, not tensors."""
        ds = VLMJerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        image, _, _ = ds[0]
        assert isinstance(image, Image.Image)
        assert image.size == (224, 224)

    def test_idx_to_class_built_from_class_to_idx(
        self, tmp_dataset_dir: Path, class_to_idx: dict[str, int]
    ) -> None:
        ds = VLMJerseyNumberDataset(tmp_dataset_dir / "train", class_to_idx)
        assert ds.idx_to_class == {0: "0", 1: "1", 2: "2"}
