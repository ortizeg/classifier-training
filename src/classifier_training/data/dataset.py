"""JSONL-annotated image dataset for jersey number classification."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from classifier_training.data.utils import get_files


class JerseyNumberDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset for JSONL-annotated image classification.

    Recursively discovers all ``annotations.jsonl`` files under ``root`` and
    merges their entries.  Image paths are resolved relative to each annotation
    file's parent directory, so real and synthetic data can coexist in
    subdirectories under the same split root.

    Each annotation row is one training sample.  Some images may appear in
    multiple rows (different label per crop annotation).  ``len(dataset)``
    equals the total number of annotation rows across all discovered files.

    Args:
        root: Split directory to search recursively for ``.jsonl`` files.
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
        self.metadata: list[dict[str, object] | None] = []

        ann_files = get_files(root, (".jsonl",))
        skipped = 0
        for ann_path in ann_files:
            ann_dir = ann_path.parent
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    suffix = record["suffix"]
                    if suffix not in class_to_idx:
                        skipped += 1
                        continue
                    img_path = ann_dir / record["image"]
                    label_idx = class_to_idx[suffix]
                    self.samples.append((img_path, label_idx))
                    self.metadata.append(record.get("metadata"))
        if skipped:
            logger.warning(
                f"Skipped {skipped} annotation(s) with unknown labels "
                f"under {root}"
            )

        logger.debug(
            f"JerseyNumberDataset: loaded {len(self.samples)} samples "
            f"from {len(ann_files)} annotation file(s) under {root}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)  # type: ignore[assignment]
        return img, label  # type: ignore[return-value]
