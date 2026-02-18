# Architecture Research

**Domain:** Image Classification Training Framework (PyTorch Lightning + Hydra)
**Researched:** 2026-02-18
**Confidence:** HIGH — based on direct inspection of sibling object-detection-training codebase plus torchmetrics/Lightning official documentation

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                          │
│                       task_manager.py                            │
│        (@hydra.main → conf/train.yaml → conf/ groups)           │
└────────────────────────┬─────────────────────────────────────────┘
                         │  instantiates via Hydra
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│   DataModule │  │    Model    │  │  Callbacks   │
│  (ImageFolder│  │(ResNetLight-│  │(EMA, ONNX,   │
│  DataModule) │  │ningModel)  │  │ Stats, CM,   │
│             │  │             │  │ Plotter, etc)│
└──────┬──────┘  └──────┬──────┘  └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌─────────────────┐
              │  L.Trainer      │
              │  (Lightning)    │
              └────────┬────────┘
                       │  .fit() → .test()
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    train loop    val loop     test loop
    (step/epoch) (step/epoch) (step/epoch)
          │            │            │
          └────────────┼────────────┘
                       ▼
          ┌────────────────────────┐
          │  outputs/YYYY-MM-DD/   │
          │  HH-MM-SS/             │
          │  ├── checkpoints/      │
          │  ├── onnx/             │
          │  ├── plots/            │
          │  ├── stats/            │
          │  └── confusion_matrix/ │
          └────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `task_manager.py` | CLI entry point; wires all Hydra-instantiated components; calls `trainer.fit()` | `@hydra.main` decorated function, mirrors detection sibling exactly |
| `tasks/base_task.py` | Abstract Pydantic BaseModel for tasks; `run()` contract | Identical to detection sibling — `BaseTask(BaseModel, ABC)` |
| `tasks/train_task.py` | Orchestrates training: connects model + datamodule + trainer + callbacks | Identical pattern to detection — holds `model`, `data`, `trainer`, `callbacks` fields |
| `tasks/onnx_export_task.py` | Post-training ONNX export as standalone task | Stripped-down task: model only, no datamodule/trainer needed |
| `data/image_folder_data_module.py` | `LightningDataModule` over `torchvision.datasets.ImageFolder`; exposes `num_classes`, `class_names` | Replaces `COCODataModule`; reads folder-per-class structure instead of COCO JSON |
| `models/base.py` | `BaseClassificationModel(L.LightningModule)` abstract base; defines `training_step`, `validation_step`, `test_step`, metrics wiring | Replaces `BaseDetectionModel`; uses `torchmetrics.MulticlassAccuracy` (top-1, top-5) and `MulticlassConfusionMatrix` instead of mAP |
| `models/resnet_lightning.py` | Concrete ResNet wrapper (ResNet18, ResNet50); `configure_optimizers` with AdamW + CosineAnnealingLR + LinearLR warmup | Replaces `RFDETRLightningModel`; uses `torchvision.models.resnet*` with pretrained ImageNet weights |
| `callbacks/ema.py` | EMA weight averaging; swap in for val/test, restore after | **Reuse as-is** from detection sibling — EMA is model-agnostic |
| `callbacks/onnx_export.py` | Export best+final checkpoints to ONNX at training end | **Reuse with minor edit** — change input_names/output_names to `["input"]`/`["logits"]` |
| `callbacks/model_info.py` | Log param count, FLOPs, inference speed to JSON | **Reuse as-is** — model-agnostic via `compute_model_stats()` |
| `callbacks/statistics.py` | Dataset statistics (class distribution, sample counts, image sizes) | **Adapt**: replace `DetectionDataset` protocol with `ClassificationDataset` protocol; remove box-size distribution; add per-class image count bar charts |
| `callbacks/confusion_matrix.py` | Per-epoch validation confusion matrix as PNG | **New** — no detection equivalent; renders `torchmetrics.MulticlassConfusionMatrix` output with seaborn/matplotlib |
| `callbacks/plotting.py` | Training history plots (loss + accuracy curves) | **Adapt**: replace mAP keys (`val/mAP`) with accuracy keys (`val/acc_top1`, `val/acc_top5`) |
| `callbacks/visualization.py` | Sample predictions with confidence scores and class names | **Adapt**: remove box drawing; render image grid with predicted vs true label text overlays |
| `transforms/` | Classification-only augmentation pipelines (no bounding box preservation) | **Simplified**: no `SanitizeBoundingBoxes`, `ConvertBoundingBoxFormat`, `NormalizeBoxCoords` — pure image transforms |
| `types.py` | TypedDicts and aliases for the classification domain | **Replace**: `ClassificationBatch`, `ClassificationPrediction`, updated `EMAState`, `ModelStats` (remove `num_classes: int | None` ambiguity) |
| `utils/hydra.py` | `@register` decorator + `instantiate_*` helpers | **Reuse as-is** — Hydra wiring is domain-agnostic |
| `conf/` | Hydra config groups: `task/`, `models/`, `data/`, `trainer/`, `callbacks/`, `transforms/` | Same group structure as detection; different leaf YAML files |

---

## Recommended Project Structure

```
src/classifier_training/
├── __init__.py
├── task_manager.py              # @hydra.main CLI entry (mirrors detection sibling)
├── types.py                     # ClassificationBatch, ModelStats, EMAState, etc.
│
├── tasks/
│   ├── __init__.py
│   ├── base_task.py             # BaseTask(BaseModel, ABC) — copy from sibling
│   ├── train_task.py            # TrainTask(BaseTask) — copy from sibling (no change)
│   └── onnx_export_task.py      # ONNXExportTask(BaseTask) — adapted for classification
│
├── data/
│   ├── __init__.py
│   ├── image_folder_data_module.py   # LightningDataModule over ImageFolder
│   └── dataset_stats.py              # ClassificationDatasetStatistics (adapted)
│
├── models/
│   ├── __init__.py
│   ├── base.py                       # BaseClassificationModel(L.LightningModule)
│   └── resnet_lightning.py           # ResNet18/50 Lightning wrappers + @register
│
├── callbacks/
│   ├── __init__.py
│   ├── ema.py                        # REUSE from sibling (model-agnostic)
│   ├── onnx_export.py                # ADAPT from sibling (output names)
│   ├── model_info.py                 # REUSE from sibling (model-agnostic)
│   ├── statistics.py                 # ADAPT: ImageFolder-compatible protocol
│   ├── confusion_matrix.py           # NEW: per-epoch confusion matrix PNG callback
│   ├── plotting.py                   # ADAPT: accuracy history instead of mAP history
│   ├── label_mapping.py              # REUSE from sibling (class_names → JSON)
│   └── visualization.py             # ADAPT: no boxes, predicted/true class overlay
│
├── transforms/
│   ├── __init__.py
│   └── conversion.py                # ToFloat32Tensor helper (same as detection)
│
└── conf/
    ├── __init__.py
    ├── train.yaml                    # Root config: defaults for all groups
    ├── train_basketball_resnet18.yaml# Dataset-specific override
    ├── task/
    │   └── train.yaml               # TrainTask config
    ├── models/
    │   ├── base.yaml                # Shared: lr, weight_decay, warmup, image_mean/std
    │   ├── resnet18.yaml            # ResNet18 config
    │   └── resnet50.yaml            # ResNet50 config
    ├── data/
    │   └── imagefolder.yaml         # ImageFolderDataModule config (train/val/test paths)
    ├── trainer/
    │   └── default.yaml             # Lightning Trainer defaults for T4
    ├── callbacks/
    │   └── default.yaml             # All callbacks enabled by default
    └── transforms/
        ├── resnet_train.yaml        # Training augmentation pipeline
        └── resnet_val.yaml          # Val/test preprocessing pipeline
```

### Structure Rationale

- **`tasks/`**: Pydantic-validated task objects invoked by `task_manager.py`. The `TrainTask` is completely domain-agnostic — it holds `model`, `data`, `trainer`, `callbacks` — and can be copied verbatim. Only `ONNXExportTask` needs classification-aware ONNX input/output axis names.
- **`data/`**: Single `ImageFolderDataModule` replaces the detection repo's `COCODataModule`. The folder-per-class ImageFolder format eliminates JSON parsing, label remapping, and box collation. `num_classes` and `class_names` come from `ImageFolder.classes` (the sorted subdirectory list).
- **`models/`**: `BaseClassificationModel` centralizes metrics (`MulticlassAccuracy`, `MulticlassConfusionMatrix`) identical to how `BaseDetectionModel` centralizes mAP. Each ResNet variant is a thin subclass registered with `@register` in Hydra — same pattern as RFDETR variants.
- **`callbacks/`**: The majority can be reused or lightly adapted. Only `confusion_matrix.py` is net-new. The `statistics.py` adaptation is the most complex because it must switch from a `DetectionDataset` protocol to an `ImageFolder`-compatible protocol.
- **`transforms/`**: Dramatically simpler than detection. No box-aware transforms. Standard classification augmentations via `torchvision.transforms.v2` (RandomHorizontalFlip, RandomResizedCrop, ColorJitter, Normalize) are sufficient. The `conversion.py` `ToFloat32Tensor` helper can be copied directly.
- **`conf/`**: Exact same Hydra config group structure. `train.yaml` references the same default group names (`task`, `models`, `data`, `trainer`, `callbacks`). YAML files use flat keys matching the pattern of the detection `conf/callbacks/default.yaml`.

---

## Architectural Patterns

### Pattern 1: Hydra @register for Model Variants

**What:** A `@register` decorator auto-registers classes in Hydra's ConfigStore, allowing YAML files to reference classes by name without hardcoding `_target_` paths in code.
**When to use:** All `LightningModule` subclasses (ResNet18, ResNet50), `LightningDataModule` (`ImageFolderDataModule`), tasks.
**Trade-offs:** Requires side-effect imports (`import classifier_training.models`) in `task_manager.py` to trigger registration; clean separation between config and code.

**Example:**
```python
# In models/resnet_lightning.py
@register(name="ResNet18")
class ResNet18ClassificationModel(ResNetClassificationModel):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", "resnet18")
        super().__init__(**kwargs)

@register(name="ResNet50")
class ResNet50ClassificationModel(ResNetClassificationModel):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", "resnet50")
        super().__init__(**kwargs)
```

```yaml
# conf/models/resnet18.yaml
defaults:
  - ResNet18
  - base
  - _self_

model_name: resnet18
learning_rate: 1e-4
warmup_epochs: 2
```

### Pattern 2: Metric-First BaseClassificationModel

**What:** All metrics (`MulticlassAccuracy`, `MulticlassConfusionMatrix`) are defined as class attributes in `__init__` so Lightning places them on the correct device automatically. The base class implements all `*_step` and `on_*_epoch_end` hooks; subclasses only implement `forward` and `configure_optimizers`.
**When to use:** All classification model variants inherit from this base.
**Trade-offs:** Couples metrics to the base class rather than callbacks, which is the Lightning-recommended pattern. Detection sibling uses the same approach for mAP.

**Example:**
```python
# In models/base.py
import torchmetrics

class BaseClassificationModel(L.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, ...):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # Metrics live on correct device automatically when defined here
        self.val_acc_top1 = torchmetrics.MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc_top5 = torchmetrics.MulticlassAccuracy(num_classes=num_classes, top_k=min(5, num_classes))
        self.val_confusion = torchmetrics.MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_acc_top1 = torchmetrics.MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.save_hyperparameters()

    def validation_step(self, batch: ClassificationBatch, batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc_top1.update(logits, labels)
        self.val_confusion.update(logits, labels)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc_top1.compute()
        self.log("val/acc_top1", acc, prog_bar=True)
        self.val_acc_top1.reset()
        self.val_confusion.reset()
```

### Pattern 3: ImageFolder DataModule with Hydra-Injected Transforms

**What:** `ImageFolderDataModule` receives `train_transforms` and `val_transforms` as `v2.Compose` objects injected by Hydra from `conf/transforms/*.yaml`. `num_classes` and `class_names` are read from `ImageFolder.classes` (alphabetically sorted subdirectories) rather than JSON annotations.
**When to use:** All classification datasets organized in folder-per-class format (Roboflow exports, standard splits).
**Trade-offs:** Alphabetical class ordering is deterministic but must match label indices in the model head. This is automatic with `ImageFolder` and does not require a label remapping step (unlike the detection sibling's COCO ID → contiguous index mapping).

**Example:**
```python
# In data/image_folder_data_module.py
from torchvision.datasets import ImageFolder

class ImageFolderDataModule(L.LightningDataModule):
    def __init__(self, train_path: str, val_path: str,
                 test_path: str | None = None,
                 train_transforms: v2.Compose | None = None,
                 val_transforms: v2.Compose | None = None,
                 batch_size: int = 32, num_workers: int = 4, ...):
        super().__init__()
        ...

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self._train_dataset = ImageFolder(self.train_path, transform=self.train_transforms)
            self._val_dataset = ImageFolder(self.val_path, transform=self.val_transforms)
        if stage in ("test", None) and self.test_path:
            self._test_dataset = ImageFolder(self.test_path, transform=self.val_transforms)

    @property
    def num_classes(self) -> int:
        return len(self._train_dataset.classes)

    @property
    def class_names(self) -> list[str]:
        return self._train_dataset.classes

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)
```

### Pattern 4: Confusion Matrix as a Standalone Callback

**What:** A dedicated `ConfusionMatrixCallback` renders the confusion matrix from `MulticlassConfusionMatrix` computed in the model's `on_validation_epoch_end`, saved as a PNG per epoch. This keeps visualization logic out of the model.
**When to use:** Always enabled in the default callbacks YAML.
**Trade-offs:** The callback must read the confusion matrix data from the model via `trainer.model.val_confusion.compute()` before reset, or the model must store the result as an attribute. The detection sibling solves this by storing `val_preds_storage` on `self`; the same pattern applies here.

**Example:**
```python
# In callbacks/confusion_matrix.py
class ConfusionMatrixCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Read computed matrix before pl_module resets it
        cm = pl_module.val_confusion.compute().cpu().numpy()
        class_names = trainer.datamodule.class_names if trainer.datamodule else None
        epoch = trainer.current_epoch
        self._save_confusion_matrix_png(cm, class_names, epoch)
```

---

## Data Flow

### Training Forward Pass

```
ImageFolder (disk)
    ↓ [image: PIL → Tensor, label: int]
train_transforms (v2.Compose: RandomResizedCrop, Flip, ColorJitter, Normalize)
    ↓ [Tensor[B,3,H,W], Tensor[B] (labels)]
DataLoader (collate_fn=default — no custom collate needed for classification)
    ↓ ClassificationBatch = tuple[Tensor, Tensor]
ResNetClassificationModel.training_step()
    ↓ forward() → logits [B, num_classes]
    ↓ F.cross_entropy(logits, labels) → scalar loss
L.Trainer.self.log("train/loss", ...)
    ↓
Optimizer.step() → EMACallback.on_train_batch_end()
```

### Validation Forward Pass

```
ImageFolder (disk)
    ↓
val_transforms (v2.Compose: Resize, CenterCrop, Normalize)
    ↓ ClassificationBatch
ResNetClassificationModel.validation_step()
    ↓ forward() → logits [B, num_classes]
    ↓ val_acc_top1.update(logits, labels)
    ↓ val_confusion.update(logits, labels)
on_validation_epoch_end()
    ↓ val_acc_top1.compute() → scalar → self.log("val/acc_top1")
    ↓ val_confusion.compute() → [num_classes, num_classes] matrix
    ↓ confusion matrix stored on pl_module (e.g., self._last_val_confusion)
    ↓ val_acc_top1.reset(), val_confusion.reset()
ConfusionMatrixCallback.on_validation_epoch_end()
    ↓ reads pl_module._last_val_confusion → matplotlib figure → PNG saved
TrainingHistoryPlotter.on_train_epoch_end()
    ↓ reads trainer.callback_metrics["val/acc_top1"] → loss_history.png + acc_history.png
```

### Class Discovery Flow

```
ImageFolderDataModule constructed with train_path
    ↓ (lazy — no I/O until setup())
setup("fit") called by Trainer
    ↓ ImageFolder(train_path) reads subdirectory names → sorted list = class_names
    ↓ ImageFolder assigns label indices 0..N-1 alphabetically
task_manager.py:
    ↓ datamodule.num_classes → passed to instantiate_model(cfg.models, num_classes=N)
    ↓ ResNet head reinitialized to N classes (same pattern as RFDETR num_classes injection)
```

### Hydra Config Resolution

```
conf/train.yaml (root)
    ├── defaults: [task: train, models: resnet18, data: imagefolder,
    │              trainer: default, callbacks: default]
    └── global params: input_height, input_width, log_level
          ↓ Hydra resolves defaults
conf/models/resnet18.yaml
    ├── defaults: [ResNet18, base, _self_]
    └── model-specific overrides
conf/data/imagefolder.yaml
    ├── defaults: [/transforms@_here_: resnet_train, _self_]
    └── train_path: ???, val_path: ???, batch_size: 32 ...
          ↓ Hydra instantiates
ImageFolderDataModule(**resolved_params, train_transforms=v2.Compose(...))
```

---

## Key Differences: Classification vs. Detection

| Concern | Detection (sibling) | Classification (this project) |
|---------|---------------------|-------------------------------|
| Annotation format | COCO JSON (`_annotations.coco.json`) | Folder-per-class (`ImageFolder`) |
| Label discovery | Parse JSON categories → remap IDs | `ImageFolder.classes` (sorted dirs) |
| Collate function | Custom `collate_fn` (list of targets) | Default PyTorch collate (tensor batches) |
| Batch type | `tuple[Tensor, list[DetectionTarget]]` | `tuple[Tensor, Tensor]` |
| Primary metric | mAP@50, mAP@50:95 (supervision) | top-1 accuracy, top-5 accuracy (torchmetrics) |
| Secondary metrics | Per-class AP, PR curves | Confusion matrix, per-class accuracy |
| Loss function | Complex weighted multi-loss (focal, GIoU, L1) | `F.cross_entropy` (single scalar) |
| Transforms complexity | Detection-aware (preserve boxes) | Image-only (random crop, flip, color jitter) |
| Custom transforms | MultiScaleRandomResize, NormalizeBoxCoords | None — all from torchvision.transforms.v2 |
| ONNX output names | `["dets", "labels"]` or `["boxes", "scores", "labels"]` | `["logits"]` (single tensor) |
| New callbacks needed | — | `ConfusionMatrixCallback` |
| Callbacks removed | `sampler_distribution` (detection-specific) | — |

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Small dataset (<5K images) | Add stronger augmentation (MixUp, CutMix via `timm` or manual); use smaller batch size; reduce early stopping patience |
| Standard (5K–100K images) | Current architecture sufficient; T4 can handle ResNet50 at batch 64 |
| Large (100K+ images) | Switch to DDP multi-GPU; increase `num_workers`; consider `use_cache: disk` |

### Scaling Priorities

1. **First bottleneck:** Data loading on single GPU — `num_workers=4` baseline, bump to 8 with `persistent_workers=True` on T4. MPS (macOS dev) must use `num_workers=0` (copy from detection sibling's guard).
2. **Second bottleneck:** Confusion matrix memory for many classes — `torchmetrics.MulticlassConfusionMatrix` accumulates in RAM; reset aggressively; avoid storing raw predictions.

---

## Anti-Patterns

### Anti-Pattern 1: Rewriting the Entire EMA Callback

**What people do:** Implement a new EMA callback "for classification" when the detection sibling's `EMACallback` is already fully model-agnostic.
**Why it's wrong:** The EMA callback operates on `pl_module.state_dict()` — it has no knowledge of detection vs. classification. Rewriting it introduces divergence and potential bugs.
**Do this instead:** Copy `callbacks/ema.py` verbatim from the detection sibling. The only import that needs to change is `from object_detection_training.types import DetectionBatch, EMAState` → `from classifier_training.types import ClassificationBatch, EMAState`.

### Anti-Pattern 2: Custom Collate Function for Classification

**What people do:** Write a custom collate function (because the detection sibling has one) when classification doesn't need it.
**Why it's wrong:** `ImageFolder` produces `(Tensor, int)` pairs. Default PyTorch collate stacks images into `[B,3,H,W]` and labels into `[B]` automatically — exactly what `F.cross_entropy` expects. A custom collate adds complexity with no benefit.
**Do this instead:** `DataLoader(..., collate_fn=None)` — rely on the default.

### Anti-Pattern 3: Storing Raw Logits for Epoch-End Metrics

**What people do:** Accumulate all `(logits, labels)` tuples in a list during `validation_step` and compute metrics in `on_validation_epoch_end` from the raw data.
**Why it's wrong:** For classification, `torchmetrics` stateful metrics (`MulticlassAccuracy`, `MulticlassConfusionMatrix`) handle incremental updates internally. Accumulating raw tensors wastes GPU/CPU memory unnecessarily.
**Do this instead:** Call `.update(logits, labels)` in `validation_step`, `.compute()` in `on_validation_epoch_end`, `.reset()` after logging. The detection sibling uses a storage list because it needs raw boxes for PR curve computation — classification has no equivalent need.

### Anti-Pattern 4: Confusion Matrix in the Model's on_validation_epoch_end

**What people do:** Render and save the confusion matrix PNG directly inside `on_validation_epoch_end` of the `LightningModule`.
**Why it's wrong:** I/O and rendering logic inside a `LightningModule` violates the separation of concerns established by the callback architecture. It also makes the model untestable without disk I/O.
**Do this instead:** Store the computed confusion matrix as `self._last_val_confusion` in `on_validation_epoch_end` (after `compute()`, before `reset()`). The `ConfusionMatrixCallback` reads `pl_module._last_val_confusion` in `Callback.on_validation_epoch_end` and handles rendering/saving.

### Anti-Pattern 5: Putting Transforms Inside the DataModule

**What people do:** Hardcode transforms (e.g., `transforms.RandomResizedCrop(224)`) directly in `ImageFolderDataModule.__init__` instead of injecting them from Hydra config.
**Why it's wrong:** It defeats the purpose of Hydra config-driven experiments. Augmentation strategy (crop size, augmentation strength) is a hyperparameter that must be overridable via CLI overrides without code changes.
**Do this instead:** Accept `train_transforms: v2.Compose | None` and `val_transforms: v2.Compose | None` as constructor parameters. Define them in `conf/transforms/resnet_train.yaml` and `conf/transforms/resnet_val.yaml`. The Hydra `@_here_` injection into `data.yaml` is the same pattern used in the detection sibling.

---

## Component Build Order (Dependencies)

Build order matters because later components import earlier ones.

```
Phase 1: Foundation
  types.py                    ← no internal imports
  utils/hydra.py              ← copy from sibling (domain-agnostic)
  tasks/base_task.py          ← copy from sibling (only pydantic + loguru)

Phase 2: Data Layer
  data/image_folder_data_module.py  ← depends on types.py, utils/hydra.py
  conf/data/imagefolder.yaml        ← depends on data module existing
  conf/transforms/resnet_train.yaml ← no code dependencies
  conf/transforms/resnet_val.yaml   ← no code dependencies

Phase 3: Model Layer
  models/base.py              ← depends on types.py, torchmetrics
  models/resnet_lightning.py  ← depends on models/base.py, utils/hydra.py
  conf/models/base.yaml
  conf/models/resnet18.yaml   ← depends on ResNet18 class being registered
  conf/models/resnet50.yaml

Phase 4: Callbacks (in dependency order)
  callbacks/ema.py            ← copy from sibling; depends on types.py
  callbacks/model_info.py     ← copy from sibling; depends on models/base.py
  callbacks/onnx_export.py    ← adapt from sibling; depends on models/base.py
  callbacks/label_mapping.py  ← copy from sibling; depends on data module
  callbacks/statistics.py     ← adapt from sibling; depends on data module
  callbacks/confusion_matrix.py  ← new; depends on models/base.py, matplotlib
  callbacks/plotting.py       ← adapt from sibling; no internal deps
  callbacks/visualization.py  ← adapt from sibling; depends on data module
  conf/callbacks/default.yaml ← depends on all callbacks existing

Phase 5: Tasks + Entry Point
  tasks/train_task.py         ← copy from sibling; depends on models, data, callbacks
  tasks/onnx_export_task.py   ← adapt from sibling; depends on models
  task_manager.py             ← copy from sibling; depends on all of the above
  conf/train.yaml             ← root config; depends on all conf/ groups
```

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GCP Cloud Build | `cloudbuild.yaml` + `cloud-build.sh` script — same as detection sibling | Pixi-based build, Docker image pushed to Artifact Registry |
| GitHub Actions | CI workflows: lint → test → release — copy from sibling | Python 3.11, pixi, ruff, mypy, pytest |
| Roboflow / dataset sources | Standard ImageFolder export format | Dataset lives on disk; paths passed via Hydra config overrides |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `task_manager.py` ↔ `tasks/` | Hydra `instantiate(cfg.task, ...)` | Task receives pre-built model, datamodule, trainer, callbacks |
| `tasks/` ↔ `models/` | Direct object reference (Lightning module) | `TrainTask.model: L.LightningModule` — same as detection |
| `tasks/` ↔ `data/` | Direct object reference (Lightning datamodule) | `TrainTask.data: L.LightningDataModule` |
| `models/` ↔ `callbacks/` | Lightning callback hooks (`on_validation_epoch_end`, etc.) | Callbacks read `pl_module._last_val_confusion`; model does not import callbacks |
| `data/` ↔ `callbacks/statistics.py` | Protocol-based duck typing (`_ClassificationDataModule` protocol) | Same pattern as detection's `_DetectionDataModule` protocol; avoids circular imports |
| `data/` ↔ `task_manager.py` | `datamodule.num_classes` read before model instantiation | `num_classes` injected into model constructor — same flow as detection |

---

## Sources

- Direct inspection of `object-detection-training` sibling codebase (HIGH confidence — primary source)
- [TorchMetrics in PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html) (HIGH confidence — official docs)
- [MulticlassConfusionMatrix](https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html) (HIGH confidence — official docs)
- [MulticlassAccuracy with top_k](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html) (HIGH confidence — official docs)
- [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) (HIGH confidence — official docs)

---

*Architecture research for: Image Classification Training Framework (classifier-training)*
*Researched: 2026-02-18*
