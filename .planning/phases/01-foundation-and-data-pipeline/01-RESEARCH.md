# Phase 1: Foundation and Data Pipeline - Research

**Researched:** 2026-02-18
**Domain:** Python project scaffold + PyTorch Lightning DataModule for JSONL-annotated image classification
**Confidence:** HIGH

---

## Summary

This phase builds a green-field Python project that mirrors the sibling `object-detection-training` repo in tooling (pixi, flit_core, ruff, mypy, pre-commit, loguru, pydantic v2) and introduces a custom `ImageFolderDataModule` that reads a flat JSONL-annotated dataset — not a standard ImageFolder layout — with strictly separated augmentation pipelines and class-imbalance handling via `WeightedRandomSampler`.

The dataset (`basketball-jersey-numbers-ocr`) has 3,615 images across three pre-split directories (`train/`, `valid/`, `test/`). Each split contains flat JPEG files plus a single `annotations.jsonl`. Images are NOT organized in class subdirectories, so `torchvision.datasets.ImageFolder` cannot be used directly. A custom `Dataset` that parses `annotations.jsonl` and maps `suffix` strings to integer indices is the only viable approach. The 43 classes include an empty-string class (`""`) and show severe imbalance (class `8` has 257 samples, class `6` has only 4 in train).

The sibling repo provides authoritative, copy-paste-ready patterns for every infrastructure concern: pixi.toml structure, pyproject.toml (flit_core + ruff + mypy), pre-commit hooks, pydantic frozen config models, loguru usage, WeightedRandomSampler, seed utilities, and labels_mapping.json serialization. All patterns below are verified by direct inspection of sibling source files.

**Primary recommendation:** Directly mirror sibling repo tooling config; write a custom `JerseyNumberDataset(Dataset)` that builds an index from JSONL, then wrap it in a `L.LightningDataModule` that applies strictly separate train vs. val/test transform pipelines, computes class weights, and persists `labels_mapping.json` with alphabetically-ordered `class_to_idx`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytorch | 2.7.1 | Tensor ops, DataLoader, WeightedRandomSampler | Locked by sibling repo's pixi env |
| torchvision | 0.24.0 | transforms.v2, image I/O | Locked; provides all needed transforms |
| lightning | 2.6.0 | LightningDataModule base class | Locked by sibling |
| pydantic | 2.12.5 | Frozen config models | Locked; v2 API required |
| loguru | 0.7.3 | Structured logging | Locked by sibling |
| pillow | (conda) | Image loading from JPEG | Already in sibling deps |

### Supporting (Dev/Build)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 9.0.2 | Test runner | All tests |
| pytest-cov | 7.0.0 | Coverage reporting | `pixi run test-cov` |
| ruff | 0.14.13 | Lint + format | `pixi run lint` / `pixi run format` |
| mypy | 1.19.1 | Strict type checking | `pixi run typecheck` |
| pre-commit | (conda) | Git hooks | Block bad commits |
| flit_core | >=3.2,<4 | Build backend | Editable installs |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom JSONL Dataset | torchvision.datasets.ImageFolder | ImageFolder needs class subdirs; dataset is flat — not applicable |
| torchvision.transforms.v2 | albumentations | sibling repo uses v2 throughout; no reason to diverge |
| stdlib json | orjson | orjson is faster; sibling uses it for COCO JSON; for JSONL line-by-line either works, stdlib json is fine |
| lightning.seed_everything | Custom seed util | Sibling implements its own (see Code Examples); FOUND-09 says "via lightning.seed_everything" — use lightning's |

### Installation (pixi.toml dependencies)
```toml
[dependencies]
python = "3.11.*"
lightning = "*"
loguru = "*"
numpy = "<2.0.0"
pydantic = ">=2.0"
pytorch = "*"
torchvision = "*"
pillow = "*"
rich = "*"

[feature.dev.dependencies]
pytest = "*"
pytest-cov = "*"
ruff = "*"
mypy = "*"
pre-commit = "*"

[pypi-dependencies]
classifier-training = { path = ".", editable = true }
```

---

## Architecture Patterns

### Recommended Project Structure
```
src/
└── classifier_training/
    ├── __init__.py
    ├── types.py              # TypedDicts, type aliases (ClassificationBatch etc.)
    ├── data/
    │   ├── __init__.py       # exports ImageFolderDataModule
    │   ├── dataset.py        # JerseyNumberDataset(Dataset[tuple[Image,int]])
    │   └── datamodule.py     # ImageFolderDataModule(L.LightningDataModule)
    └── utils/
        ├── __init__.py
        └── seed.py           # seed_everything(seed: int) -> None

tests/
├── __init__.py
├── conftest.py               # shared fixtures (sample_image, tmp_dataset_dir)
├── test_dataset.py           # JerseyNumberDataset unit tests
├── test_datamodule.py        # ImageFolderDataModule integration tests
└── test_utils_seed.py        # seed reproducibility tests
```

### Pattern 1: pixi.toml Structure (mirror sibling exactly)
**What:** Workspace-level pixi.toml with conda-forge channel, osx-arm64 + linux-64, dev feature
**When to use:** Always — this is the locked environment manager
```toml
# Source: /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/pixi.toml
[workspace]
name = "classifier-training"
version = "0.1.0"
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64"]

[tasks]
test = "pytest"
test-cov = "pytest --cov=src --cov-report=term --cov-report=xml -v"
lint = "ruff check ."
format = "ruff format ."
format-check = "ruff format --check ."
typecheck = "mypy src/"
precommit = "pre-commit run --all-files"

[dependencies]
python = "3.11.*"
# ... (see Installation section)
```

### Pattern 2: pyproject.toml (flit_core + ruff + mypy)
**What:** flit_core build backend, ruff select rules matching sibling, mypy strict with third-party ignore overrides
**When to use:** Always — defines editable install and linting config
```toml
# Source: /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/pyproject.toml
[build-system]
requires = ["flit_core >=3.2,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "classifier_training"
version = "0.0.1"
# ...

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "S", "A", "C4", "RUF"]
ignore = ["N812"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]

[tool.ruff.lint.isort]
known-first-party = ["classifier_training"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = ["torch.*", "torchvision.*", "lightning.*", "pydantic.*", "PIL.*", "loguru.*", "numpy.*"]
ignore_missing_imports = true
```

### Pattern 3: Custom JSONL Dataset
**What:** A `Dataset` that reads `annotations.jsonl` line-by-line, builds a list of `(image_path, label_idx)` tuples, and applies transforms on `__getitem__`
**When to use:** Required — the dataset is not ImageFolder-compatible (flat dir, no class subdirs)
```python
# Source: verified against dataset structure at
# /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-jersey-numbers-ocr/
import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class JerseyNumberDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset for JSONL-annotated jersey number classification.

    Reads flat directory + annotations.jsonl. The 'suffix' field in each
    JSONL record is the class label (jersey number as string).
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
                record = json.loads(line)
                img_path = root / record["image"]
                label_idx = class_to_idx[record["suffix"]]
                self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)  # type: ignore[assignment]
        return img, label  # type: ignore[return-value]
```

### Pattern 4: LightningDataModule with Strict Transform Separation
**What:** `ImageFolderDataModule` builds class_to_idx once from train split, assigns train transforms only to train dataset, val/test get inference-only transforms
**When to use:** Always — DATA-03 is a hard requirement

```python
# Source: pattern from sibling coco_data_module.py, adapted for classification
import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2

from classifier_training.data.dataset import JerseyNumberDataset

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.image_size = image_size

        self._class_to_idx: dict[str, int] | None = None
        self._train_dataset: JerseyNumberDataset | None = None
        self._val_dataset: JerseyNumberDataset | None = None
        self._test_dataset: JerseyNumberDataset | None = None

    @property
    def class_to_idx(self) -> dict[str, int]:
        if self._class_to_idx is None:
            self._build_class_to_idx()
        return self._class_to_idx  # type: ignore[return-value]

    def _build_class_to_idx(self) -> None:
        """Build alphabetically-sorted class_to_idx from train annotations."""
        ann_path = self.data_root / "train" / "annotations.jsonl"
        classes: set[str] = set()
        with open(ann_path) as f:
            for line in f:
                classes.add(json.loads(line)["suffix"])
        # CRITICAL: alphabetical sort for deterministic, reproducible mapping
        self._class_to_idx = {cls: i for i, cls in enumerate(sorted(classes))}

    def _build_train_transforms(self) -> v2.Compose:
        return v2.Compose([
            v2.RandomResizedCrop(self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _build_val_test_transforms(self) -> v2.Compose:
        return v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self._train_dataset = JerseyNumberDataset(
                self.data_root / "train", self.class_to_idx, self._build_train_transforms()
            )
            self._val_dataset = JerseyNumberDataset(
                self.data_root / "valid", self.class_to_idx, self._build_val_test_transforms()
            )
        if stage in ("test", None):
            self._test_dataset = JerseyNumberDataset(
                self.data_root / "test", self.class_to_idx, self._build_val_test_transforms()
            )

    def _compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weight tensor for weighted loss."""
        assert self._train_dataset is not None
        counts = torch.zeros(len(self.class_to_idx))
        for _, label in self._train_dataset.samples:
            counts[label] += 1
        weights = 1.0 / counts.clamp(min=1.0)
        return weights / weights.sum() * len(self.class_to_idx)

    def _build_sampler(self) -> WeightedRandomSampler:
        """Per-sample weight for WeightedRandomSampler from class frequencies."""
        assert self._train_dataset is not None
        class_weights = self._compute_class_weights()
        sample_weights = [class_weights[label].item() for _, label in self._train_dataset.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        assert self._train_dataset is not None
        sampler = self._build_sampler()
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,          # sampler and shuffle are mutually exclusive
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def save_labels_mapping(self, save_path: Path) -> None:
        """Persist class_to_idx as labels_mapping.json with metadata."""
        mapping = {
            "num_classes": len(self.class_to_idx),
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
```

### Pattern 5: Pydantic Frozen Config Model (sibling pattern)
**What:** All config objects use `BaseModel` with `frozen=True`; validated at construction time
**When to use:** Any config class (sampler config, datamodule config)
```python
# Source: /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/data/sampler.py
from pydantic import BaseModel, model_validator

class DataModuleConfig(BaseModel, frozen=True):
    data_root: str
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    image_size: int = 224
```

### Pattern 6: seed_everything (use lightning's, per FOUND-09)
**What:** `lightning.pytorch.seed_everything` sets all RNG seeds including Python, NumPy, PyTorch, CUDA
**When to use:** At training start, and in tests requiring determinism

```python
# Source: lightning 2.6.0 API
import lightning as L

L.seed_everything(42, workers=True)
# workers=True also seeds DataLoader workers via worker_init_fn
```

NOTE: The sibling repo implements its own `seed_everything` in `utils/seed.py` (verified). FOUND-09 explicitly says "Reproducibility seed via lightning.seed_everything" — use lightning's built-in, not a custom implementation.

### Pattern 7: pre-commit config (mirror sibling)
```yaml
# Source: /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/.pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

### Anti-Patterns to Avoid
- **Using `torchvision.datasets.ImageFolder` directly:** The dataset is flat (no class subdirs). It cannot be used without restructuring the entire dataset on disk.
- **Building class_to_idx from the union of all splits:** Build it from train only; val/test use the same mapping. Avoid dynamic discovery at val/test time.
- **Applying augmentation to val or test sets:** The transform pipeline must be branched at construction time, not conditionally in `__getitem__`. Assign different `transform` objects to train vs val/test Dataset instances.
- **Setting `shuffle=True` with `WeightedRandomSampler`:** These are mutually exclusive in PyTorch DataLoader. Setting both raises a RuntimeError.
- **Non-deterministic class ordering:** Sort class names alphabetically before building class_to_idx. Do not rely on set iteration order or annotation file order.
- **MPS + multiprocessing:** The sibling repo detects MPS and sets `num_workers=0`. For osx-arm64, enforce this guard in DataModule `__init__`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Class-imbalanced sampling | Custom sampling loop | `torch.utils.data.WeightedRandomSampler` | Handles replacement, num_samples, integrates with DataLoader |
| Image augmentation pipeline | Custom transforms | `torchvision.transforms.v2.Compose` | Handles PIL and tensor types, handles dtype, correct composition |
| Seed reproducibility | Custom seed function | `lightning.pytorch.seed_everything(seed, workers=True)` | Handles Python/NumPy/PyTorch/CUDA + worker seeds in one call |
| Config validation | Dict + assert | `pydantic.BaseModel(frozen=True)` | Type validation, immutability, IDE support |
| Test coverage | Manual tracking | `pytest-cov` with `--cov-report` | Integrated with pytest, outputs XML for CI |

**Key insight:** The sibling repo has solved all of these correctly. Copy the patterns, don't invent.

---

## Common Pitfalls

### Pitfall 1: Empty-String Class Label
**What goes wrong:** 39 train annotations have `suffix = ""` (empty string). Naively filtering or ignoring them corrupts class_to_idx for downstream splits that also contain this class.
**Why it happens:** The dataset has a legitimate "unknown/unreadable number" class encoded as empty string.
**How to avoid:** Include `""` in class_to_idx. Alphabetically, `""` sorts before all numeric strings, so it gets index 0. Verify this in tests.
**Warning signs:** `len(class_to_idx) == 42` instead of 43.

### Pitfall 2: class_to_idx Ordering Inconsistency
**What goes wrong:** If class_to_idx is built from a `set` without sorting, Python's set iteration is non-deterministic across interpreter runs. The model trains with one mapping but inference uses another.
**Why it happens:** `set()` guarantees no order. Dataset has 43 string class names that include numeric strings like `"1"`, `"10"`, `"11"` — lexicographic sort is the right choice (not numeric sort), because the label names are arbitrary strings.
**How to avoid:** `{cls: i for i, cls in enumerate(sorted(all_classes))}`. Persist to `labels_mapping.json` immediately in `setup()` or after `_build_class_to_idx()`.
**Warning signs:** Model accuracy fluctuates suspiciously between runs with identical data and seeds.

### Pitfall 3: Val/Test See Train Augmentations
**What goes wrong:** RandomResizedCrop and ColorJitter applied to val/test produce non-reproducible evaluation metrics.
**Why it happens:** Sharing a single transform object across all splits, or applying augmentation via a flag in `__getitem__` that doesn't get reset.
**How to avoid:** Create separate `Compose` instances for train and val/test. Assign at `Dataset` construction time, not at `DataLoader` time. Test explicitly: check that a val batch produced twice from the same index is pixel-identical.
**Warning signs:** Val loss oscillates more than expected; val batch from same index differs between runs.

### Pitfall 4: `shuffle=True` + `WeightedRandomSampler`
**What goes wrong:** `DataLoader(shuffle=True, sampler=sampler)` raises `ValueError: sampler option is mutually exclusive with shuffle`.
**Why it happens:** Easy to forget to set `shuffle=False` when adding a sampler.
**How to avoid:** Always set `shuffle=False` (explicitly) when passing `sampler=` to DataLoader. Document in code comment.
**Warning signs:** Immediate crash at DataLoader construction.

### Pitfall 5: MPS Multiprocessing Crash (macOS arm64)
**What goes wrong:** `RuntimeError: unable to open shared file ...share_filename_cpu` when `num_workers > 0` on MPS device.
**Why it happens:** PyTorch multiprocessing DataLoader workers share memory differently on MPS vs CUDA.
**How to avoid:** Copy sibling pattern exactly:
```python
if torch.backends.mps.is_available() and num_workers > 0:
    logger.warning("MPS detected, setting num_workers=0")
    num_workers = 0
```
**Warning signs:** Crash on first DataLoader iteration on any M-series Mac.

### Pitfall 6: annotations.jsonl Has More Rows Than Images
**What goes wrong:** `len(annotations.jsonl) != len(image files)` — train has 2,891 images but 2,930 annotation rows. Some images have multiple annotations (VQA-style: one annotation per visible jersey in the crop, possibly).
**Why it happens:** This is a VQA/OCR dataset from Roboflow. Each annotation row is one recognition task; a single image file may appear in multiple rows.
**How to avoid:** Build `samples` list from annotation rows (not from image files). Do NOT deduplicate by filename. One annotation row = one training sample (same image, different label). Write a test verifying `len(dataset) == len(annotation_rows)`.
**Warning signs:** `len(dataset) == 2891` (image count) instead of 2930 (annotation count).

### Pitfall 7: flit_core Requires `__version__` in `__init__.py`
**What goes wrong:** `flit_core` build fails with `AttributeError: module has no attribute '__version__'` if the package's `__init__.py` is empty or missing the version attribute.
**Why it happens:** flit_core discovers version from `__init__.py:__version__` or from `pyproject.toml` depending on config. Using `dynamic = ["version"]` in pyproject.toml avoids the issue but sibling uses a hardcoded version.
**How to avoid:** Either pin `version = "0.0.1"` in `[project]` in pyproject.toml (sibling approach), OR add `__version__ = "0.0.1"` to `__init__.py`. Pick one — don't mix.
**Warning signs:** `pixi install` / editable install fails.

---

## Code Examples

### Building class_to_idx Alphabetically from JSONL
```python
# Source: verified against dataset at
# /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-jersey-numbers-ocr/
import json
from pathlib import Path

def build_class_to_idx(ann_path: Path) -> dict[str, int]:
    classes: set[str] = set()
    with open(ann_path) as f:
        for line in f:
            classes.add(json.loads(line)["suffix"])
    return {cls: i for i, cls in enumerate(sorted(classes))}

# Result (43 classes, "" gets index 0, "0" gets 1, ..., "9" gets 42):
# {"": 0, "0": 1, "00": 2, "1": 3, "10": 4, "11": 5, ...}
```

### Saving labels_mapping.json
```python
# Source: pattern from sibling detection_dataset.py + coco_data_module.py
import json
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def save_labels_mapping(class_to_idx: dict[str, int], save_path: Path) -> None:
    """Persist alphabetically-ordered class_to_idx with normalization metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,              # str -> int, alphabetical
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
    }
    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)
```

### WeightedRandomSampler from Class Counts
```python
# Source: pattern from sibling sampler.py, simplified for classification
# (classification has 1 label per sample, no need for "max weight of annotations")
import torch
from torch.utils.data import WeightedRandomSampler

def build_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    class_weights = 1.0 / counts.clamp(min=1.0)
    sample_weights = [class_weights[lbl].item() for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
```

### torchvision.transforms.v2 Pipelines
```python
# Source: sibling coco_data_module.py uses v2; torchvision 0.24.0
from torchvision.transforms import v2
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Train: augmentation included
train_transforms = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Val/Test: deterministic only
val_test_transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

### lightning.seed_everything (FOUND-09)
```python
# Source: lightning 2.6.0 — use this, not a custom implementation
import lightning as L

L.seed_everything(42, workers=True)
# workers=True passes worker_init_fn to DataLoader workers automatically
```

### Loguru Usage Pattern (mirror sibling)
```python
# Source: sibling data/sampler.py, coco_data_module.py
from loguru import logger

logger.info(f"Loaded {len(samples)} samples from {ann_path}")
logger.warning("MPS detected, setting num_workers=0")
logger.debug(f"class_to_idx: {class_to_idx}")
```

### Conftest Fixtures for Tests
```python
# Source: sibling tests/conftest.py pattern, adapted for classification
import pytest
import json
import pathlib
from PIL import Image

@pytest.fixture()
def tmp_dataset_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Minimal dataset structure for testing (3 classes, 6 samples)."""
    for split in ("train", "valid", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        lines = []
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 80, 100, 150))
            fname = f"img_{i:03d}.jpg"
            img.save(split_dir / fname)
            label = str(i)  # classes: "0", "1", "2"
            lines.append(json.dumps({"image": fname, "prefix": "Read the number.", "suffix": label}))
        (split_dir / "annotations.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path
```

---

## Dataset Facts (for planning)

Verified by direct inspection:

| Fact | Value |
|------|-------|
| Train images | 2,891 |
| Train annotation rows | 2,930 (some images appear >1 time) |
| Val images | 364 |
| Val annotation rows | 372 |
| Test images | 360 |
| Test annotation rows | 365 |
| Total unique classes | 43 |
| Classes in train | 43 (includes `""`) |
| Classes in valid | 41 (missing `"35"`, `"6"`) |
| Classes in test | 42 (missing `"20"`) |
| Most frequent (train) | `"8"` = 257, `"2"` = 217 |
| Least frequent (train) | `"6"` = 4, `"46"` = 5 |
| Image size | 224x224 JPEG (pre-resized by Roboflow) |
| Annotation format | `{"image": "...", "prefix": "Read the number.", "suffix": "<label>"}` |

**Critical:** class_to_idx MUST be built from train split and used for ALL splits. Val/test do not cover all 43 classes — the mapping must come from train.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torchvision.transforms` (v1) | `torchvision.transforms.v2` | torchvision 0.15+ | v2 handles PIL and tensors; v1 deprecated |
| `torchvision.transforms.ToTensor()` | `v2.ToImage() + v2.ToDtype(torch.float32, scale=True)` | torchvision 0.18+ | ToTensor deprecated in v2 |
| `lightning.seed_everything` from `pytorch_lightning` | `lightning.seed_everything` from `lightning` | lightning 2.0 | Import path changed |
| Custom WeightedRandomSampler weight loops | Single-pass count + inverse | stable | Pattern from sibling is optimal |

**Deprecated/outdated:**
- `torchvision.transforms.ToTensor()`: Deprecated in torchvision.transforms.v2. Use `v2.ToImage() + v2.ToDtype(torch.float32, scale=True)` instead.
- `pytorch_lightning` package import: Use `import lightning as L` (the unified package since 2.0).

---

## Open Questions

1. **Should `""` (empty-string) class be included or filtered?**
   - What we know: 39 train + some val/test samples have `suffix=""`. They represent "unreadable" jersey numbers.
   - What's unclear: The requirements say "43 classes" — this implies `""` IS included.
   - Recommendation: Include `""` as a legitimate class. It gets index 0 (alphabetical). Verify count in tests.

2. **Should `save_labels_mapping` be called in `setup()` or as a separate method?**
   - What we know: Success criterion 5 requires `labels_mapping.json` written to disk after `DataModule.setup()`. The sibling exports it via callback or explicit call.
   - What's unclear: Whether it should auto-write during `setup()` or require explicit caller.
   - Recommendation: Expose as `save_labels_mapping(path: Path)` method. Auto-writing in `setup()` requires knowing the save path at construction time, which couples DataModule to filesystem layout. Explicit call is cleaner and easier to test.

3. **What is the exact JSON structure required for labels_mapping.json?**
   - What we know: Requirements say "alphabetically-ordered class_to_idx mapping for all 43 classes". No further schema specified.
   - What's unclear: Whether the planner wants exactly the sibling's schema (with `id_to_name`, `name_to_original_id`, etc.) or a simpler classification-specific schema.
   - Recommendation: Use classification-appropriate schema: `{num_classes, class_to_idx, idx_to_class, normalization}`. The sibling's detection schema (with `original_id_to_contiguous_id`) is detection-specific and doesn't apply.

---

## Sources

### Primary (HIGH confidence)
- Direct inspection: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/pixi.toml` — pixi workspace structure, channel, platforms, tasks, dependency versions
- Direct inspection: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/pyproject.toml` — flit_core, ruff rules, mypy strict config, mypy overrides pattern
- Direct inspection: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/.pre-commit-config.yaml` — hook versions and args
- Direct inspection: `src/object_detection_training/data/base.py` — BaseDataModule pattern
- Direct inspection: `src/object_detection_training/data/coco_data_module.py` — DataModule setup/train/val/test pattern, MPS guard, transform injection
- Direct inspection: `src/object_detection_training/data/sampler.py` — WeightedRandomSampler, SamplerConfig, pydantic frozen model
- Direct inspection: `src/object_detection_training/data/detection_dataset.py` — export_labels_mapping, labels_mapping property
- Direct inspection: `src/object_detection_training/utils/seed.py` — seed_everything implementation
- Direct inspection: `src/object_detection_training/schemas/label_mapping.py` — LabelMapping pydantic schema
- Direct inspection: `tests/conftest.py` — fixture patterns
- Direct inspection: `tests/test_sampler.py` — test class organization, fixtures, assertions
- Runtime verification: `pixi run python -c "import torch, torchvision, lightning, pydantic, loguru..."` — exact installed versions
- Dataset inspection: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-jersey-numbers-ocr/` — JSONL structure, class counts, split counts

### Secondary (MEDIUM confidence)
- `lightning 2.6.0` API: `L.seed_everything(seed, workers=True)` — verified by FOUND-09 requirement text; workers parameter presence confirmed via lightning documentation pattern

### Tertiary (LOW confidence)
- None — all claims verified from primary sources above.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified by running `pixi run python` in sibling env
- Architecture patterns: HIGH — directly copied from working sibling repo files
- Dataset facts: HIGH — verified by running Python against actual data
- Pitfalls: HIGH — verified from sibling code (MPS guard exists verbatim) and dataset inspection (empty class, image vs annotation count mismatch)
- `lightning.seed_everything(workers=True)` signature: MEDIUM — not run against the environment, but stable API

**Research date:** 2026-02-18
**Valid until:** 2026-03-20 (pixi env is locked; library versions stable for 30 days)
