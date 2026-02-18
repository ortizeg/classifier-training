"""JSONL-annotated image dataset for jersey number classification."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


class JerseyNumberDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset for flat-directory JSONL-annotated image classification.

    The basketball-jersey-numbers-ocr dataset is NOT ImageFolder-compatible:
    images are in flat directories (no class subdirs). Each split has a single
    annotations.jsonl where each line is:
        {"image": "filename.jpg", "prefix": "Read the number.", "suffix": "<label>"}

    One annotation row = one training sample. Some images appear in multiple rows
    (different label per crop annotation). Build samples from annotation rows,
    NOT from the image file list — len(dataset) = len(annotation_rows).

    Args:
        root: Split directory containing flat image files and annotations.jsonl.
        class_to_idx: Alphabetically-ordered mapping from label string to integer.
            MUST be built from train split only and shared across val/test.
        transform: Optional callable applied to PIL Image, returns torch.Tensor.
    """

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        ann_path = root / "annotations.jsonl"
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                img_path = root / record["image"]
                label_idx = class_to_idx[record["suffix"]]
                self.samples.append((img_path, label_idx))

        logger.debug(
            f"JerseyNumberDataset: loaded {len(self.samples)} samples from {ann_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)  # type: ignore[assignment]
        return img, label  # type: ignore[return-value]
