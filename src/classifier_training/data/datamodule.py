"""LightningDataModule for the basketball jersey numbers OCR dataset."""

import json
from pathlib import Path

import lightning as L
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from classifier_training.config import DataModuleConfig
from classifier_training.data.dataset import JerseyNumberDataset
from classifier_training.data.sampler import TrackingWeightedRandomSampler

# ImageNet normalization statistics — stored here and written to labels_mapping.json.
# The ONNX inference pipeline in basketball-2d-to-3d reads normalization params
# from labels_mapping.json and applies them at inference time.
# Do NOT bake normalization into the ONNX graph (user decision, per STATE.md).
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class ImageFolderDataModule(L.LightningDataModule):
    """DataModule for the basketball jersey numbers OCR dataset.

    Reads flat JSONL-annotated splits from data_root/{train,valid,test}/.
    Applies strictly separate transform pipelines:
      - Train: RandomResizedCrop + RandomHorizontalFlip + ColorJitter + Normalize
      - Val/Test: Resize(256) + CenterCrop + Normalize (deterministic only)

    class_to_idx is built from train split only, sorted alphabetically. The
    empty-string class ("") is a legitimate class at index 0. All 43 classes
    must be in the mapping.

    Args:
        config: DataModuleConfig frozen model with all DataLoader parameters.
    """

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self._config = config
        self._data_root = Path(config.data_root)

        # MPS guard: multiprocessing DataLoader workers crash on Apple Silicon.
        # Detect at construction time and override num_workers.
        # Per research/sibling pattern — must set num_workers=0 on MPS.
        num_workers = config.num_workers
        if torch.backends.mps.is_available() and num_workers > 0:
            logger.warning(
                "MPS detected: setting num_workers=0 to avoid multiprocessing "
                "crash. Use linux-64 / CUDA for multi-worker DataLoading."
            )
            num_workers = 0

        self._num_workers = num_workers
        self._pin_memory = config.pin_memory
        # persistent_workers is meaningless (and silently ignored) with 0 workers
        self._persistent_workers = config.persistent_workers and num_workers > 0
        self._batch_size = config.batch_size
        self._image_size = config.image_size

        self._class_to_idx: dict[str, int] | None = None
        self._train_dataset: JerseyNumberDataset | None = None
        self._val_dataset: JerseyNumberDataset | None = None
        self._test_dataset: JerseyNumberDataset | None = None

    # ------------------------------------------------------------------
    # Class mapping — built from train only, alphabetical, deterministic
    # ------------------------------------------------------------------

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Alphabetically-sorted class_to_idx built from train annotations.

        Built lazily on first access. Safe to call before setup().
        """
        if self._class_to_idx is None:
            self._build_class_to_idx()
        return self._class_to_idx  # type: ignore[return-value]

    @property
    def num_classes(self) -> int:
        """Total number of classes (43 for basketball jersey numbers dataset)."""
        return len(self.class_to_idx)

    def _build_class_to_idx(self) -> None:
        """Build alphabetically-sorted class_to_idx from train split annotations.

        CRITICAL invariants:
        - Built from train only (val/test may not cover all classes).
        - Alphabetical sort: "" < "0" < "00" < "1" < "10" < ... (lexicographic).
        - "" (empty string) gets index 0 — represents unreadable jersey numbers.
        - Must have exactly 43 classes for basketball jersey numbers dataset.
        """
        ann_path = self._data_root / "train" / "annotations.jsonl"
        classes: set[str] = set()
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                classes.add(json.loads(line)["suffix"])
        self._class_to_idx = {cls: i for i, cls in enumerate(sorted(classes))}
        logger.info(
            f"Built class_to_idx: {len(self._class_to_idx)} classes from {ann_path}"
        )
        logger.debug(f"class_to_idx: {self._class_to_idx}")

    # ------------------------------------------------------------------
    # Transform pipelines — strictly separated at construction time
    # ------------------------------------------------------------------

    def _build_train_transforms(self) -> v2.Compose:
        """Train transforms: augmentation + normalization.

        RandomResizedCrop and ColorJitter are ONLY applied to the train dataset.
        These transforms are assigned at Dataset construction time, not
        conditionally in __getitem__ — this is the only safe pattern (DATA-03).
        """
        return v2.Compose([
            v2.RandomResizedCrop(self._image_size),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _build_val_test_transforms(self) -> v2.Compose:
        """Val/test transforms: deterministic resize + normalize only.

        No augmentation. Two passes over the same image produce identical tensors.
        Uses v2.ToImage() + v2.ToDtype() — ToTensor() is deprecated in v2.
        """
        return v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(self._image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # LightningDataModule lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets for the given stage.

        Args:
            stage: "fit", "test", or None (all stages).
                   "fit" instantiates train + val.
                   "test" instantiates test only.
                   None instantiates all three.
        """
        if stage in ("fit", None):
            self._train_dataset = JerseyNumberDataset(
                root=self._data_root / "train",
                class_to_idx=self.class_to_idx,
                transform=self._build_train_transforms(),
            )
            self._val_dataset = JerseyNumberDataset(
                root=self._data_root / "valid",
                class_to_idx=self.class_to_idx,
                transform=self._build_val_test_transforms(),
            )
            logger.info(
                f"Setup fit: train={len(self._train_dataset)}, "
                f"val={len(self._val_dataset)} samples"
            )

        if stage in ("test", None):
            self._test_dataset = JerseyNumberDataset(
                root=self._data_root / "test",
                class_to_idx=self.class_to_idx,
                transform=self._build_val_test_transforms(),
            )
            logger.info(f"Setup test: {len(self._test_dataset)} samples")

    # ------------------------------------------------------------------
    # Class imbalance handling
    # ------------------------------------------------------------------

    def _compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weight tensor for weighted CrossEntropyLoss.

        Returns tensor of shape (num_classes,) where rare classes get higher weight.
        Normalized so weights sum to num_classes (preserves scale vs unweighted).
        """
        if self._train_dataset is None:
            raise RuntimeError("Call setup('fit') before _compute_class_weights")
        counts = torch.zeros(self.num_classes)
        for _, label in self._train_dataset.samples:
            counts[label] += 1.0
        weights = 1.0 / counts.clamp(min=1.0)
        return weights / weights.sum() * self.num_classes

    def get_class_weights(self) -> torch.Tensor:
        """Public accessor for class weights tensor (for weighted CrossEntropyLoss).

        Requires setup('fit') to have been called first.
        """
        return self._compute_class_weights()

    def _build_sampler(self) -> TrackingWeightedRandomSampler:
        """Per-sample weight sampler for WeightedRandomSampler.

        Uses inverse class frequency: rare classes get higher per-sample weight.
        replacement=True required — samples may repeat within an epoch.

        IMPORTANT: DataLoader using this sampler MUST set shuffle=False.
        shuffle=True + sampler raises ValueError in PyTorch DataLoader.
        """
        if self._train_dataset is None:
            raise RuntimeError("Call setup('fit') before _build_sampler")
        class_weights = self._compute_class_weights()
        sample_weights = [
            class_weights[label].item()
            for _, label in self._train_dataset.samples
        ]
        return TrackingWeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return training DataLoader with WeightedRandomSampler."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup('fit') first")
        sampler = self._build_sampler()
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            sampler=sampler,
            shuffle=False,  # MUST be False when sampler provided — mutually exclusive
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return validation DataLoader (deterministic, no sampler)."""
        if self._val_dataset is None:
            raise RuntimeError("Call setup('fit') first")
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return test DataLoader (deterministic, no sampler)."""
        if self._test_dataset is None:
            raise RuntimeError("Call setup('test') first")
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    # ------------------------------------------------------------------
    # labels_mapping.json serialization
    # ------------------------------------------------------------------

    def save_labels_mapping(self, save_path: Path) -> None:
        """Persist class_to_idx and normalization stats as labels_mapping.json.

        Written alongside ONNX model exports so the basketball-2d-to-3d inference
        pipeline can apply correct preprocessing and map logits to class labels.

        Normalization is documented here (not baked into ONNX graph) per user
        decision.

        Args:
            save_path: Destination path for labels_mapping.json.
        """
        mapping = {
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": {str(v): k for k, v in self.class_to_idx.items()},
            "normalization": {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"Saved labels_mapping.json to {save_path}")
