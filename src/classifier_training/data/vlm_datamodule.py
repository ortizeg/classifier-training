"""LightningDataModule for VLM fine-tuning on jersey number classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from classifier_training.data.utils import get_files
from classifier_training.data.vlm_collator import VLMCollator
from classifier_training.data.vlm_dataset import VLMJerseyNumberDataset


class VLMDataModule(L.LightningDataModule):
    """Lightning DataModule for VLM fine-tuning.

    Loads train/val splits, creates :class:`VLMJerseyNumberDataset` instances,
    and uses :class:`VLMCollator` for batching via the HF processor.

    Args:
        data_root: Path to dataset root containing train/valid/test splits.
        model_name: HuggingFace model ID for loading the processor.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader workers.
        max_length: Maximum sequence length for tokenization.
        **kwargs: Absorbs extra Hydra-injected keys (_target_, etc.).
    """

    def __init__(
        self,
        data_root: str = "",
        model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 384,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._data_root = Path(data_root)
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length

        # MPS guard: multiprocessing DataLoader workers crash on Apple Silicon.
        if torch.backends.mps.is_available() and num_workers > 0:
            logger.warning(
                "MPS detected: setting num_workers=0 to avoid multiprocessing crash."
            )
            num_workers = 0
        self._num_workers = num_workers

        self._class_to_idx: dict[str, int] | None = None
        self._train_dataset: VLMJerseyNumberDataset | None = None
        self._val_dataset: VLMJerseyNumberDataset | None = None

        self._processor: AutoProcessor | None = None
        self._collator: VLMCollator | None = None

    # ------------------------------------------------------------------
    # Class mapping — same logic as ImageFolderDataModule
    # ------------------------------------------------------------------

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Alphabetically-sorted class_to_idx built from train annotations."""
        if self._class_to_idx is None:
            self._build_class_to_idx()
        return self._class_to_idx  # type: ignore[return-value]

    @property
    def idx_to_class(self) -> dict[int, str]:
        """Inverse mapping from integer index to class label string."""
        return {v: k for k, v in self.class_to_idx.items()}

    @property
    def num_classes(self) -> int:
        """Total number of classes."""
        return len(self.class_to_idx)

    def _build_class_to_idx(self) -> None:
        """Build alphabetically-sorted class_to_idx from train split annotations."""
        train_root = self._data_root / "train"
        ann_files = get_files(train_root, (".jsonl",))
        classes: set[str] = set()
        for ann_path in ann_files:
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    suffix = json.loads(line)["suffix"]
                    if suffix:  # skip empty-string labels
                        classes.add(suffix)
        self._class_to_idx = {cls: i for i, cls in enumerate(sorted(classes))}
        logger.info(
            f"Built class_to_idx: {len(self._class_to_idx)} classes "
            f"from {len(ann_files)} annotation file(s) under {train_root}"
        )

    # ------------------------------------------------------------------
    # Processor & collator
    # ------------------------------------------------------------------

    @property
    def processor(self) -> Any:
        """Lazily load the HF processor.

        Sets ``size={"longest_edge": 384}`` to limit image tiling.
        Default (1536) creates a 4x4 grid → 17 tiles × 81 = 1377 tokens/image.
        With 384, only 1 tile + overview → 162 tokens/image, which fits on L4.
        Jersey number crops are small so high-res tiling is unnecessary.
        """
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self._model_name,
                size={"longest_edge": 384},
            )
        return self._processor

    @property
    def collator(self) -> VLMCollator:
        """Lazily create the VLM collator."""
        if self._collator is None:
            self._collator = VLMCollator(
                processor=self.processor, max_length=self._max_length
            )
        return self._collator

    # ------------------------------------------------------------------
    # LightningDataModule lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets for the given stage."""
        if stage in ("fit", None):
            self._train_dataset = VLMJerseyNumberDataset(
                root=self._data_root / "train",
                class_to_idx=self.class_to_idx,
            )
            self._val_dataset = VLMJerseyNumberDataset(
                root=self._data_root / "valid",
                class_to_idx=self.class_to_idx,
            )
            logger.info(
                f"VLM setup fit: train={len(self._train_dataset)}, "
                f"val={len(self._val_dataset)} samples"
            )

    # ------------------------------------------------------------------
    # Class weights (compatibility with train.py)
    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """Return uniform weights (VLM uses causal LM loss, not weighted CE)."""
        return torch.ones(self.num_classes)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader[tuple[Any, ...]]:
        """Return training DataLoader."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup('fit') first")
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[Any, ...]]:
        """Return validation DataLoader."""
        if self._val_dataset is None:
            raise RuntimeError("Call setup('fit') first")
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
