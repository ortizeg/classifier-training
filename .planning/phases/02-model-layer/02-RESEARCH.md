# Phase 2: Model Layer - Research

**Researched:** 2026-02-18
**Domain:** PyTorch Lightning LightningModule, torchvision ResNet, torchmetrics, Hydra ConfigStore
**Confidence:** HIGH

---

## Summary

Phase 2 builds two ResNet classification models (ResNet18 and ResNet50) as PyTorch Lightning
LightningModules, wired to torchmetrics for Top-1, Top-5, and per-class accuracy, with AdamW +
linear warmup + cosine annealing scheduling, and registered for Hydra ConfigStore instantiation.
All APIs were verified against the live pixi environment (Python 3.11, torch 2.10.0, torchvision
0.25.0, lightning 2.6.1, torchmetrics 1.8.2).

One critical infrastructure gap was discovered: `hydra-core` is not installed in the
classifier-training pixi environment. The sibling repo (`object-detection-training`) has
`hydra-core = "*"` in its pixi.toml; the classifier-training pixi.toml is missing it. This must
be added before Hydra ConfigStore registration or YAML config loading can work. The `@register`
decorator pattern from the sibling repo can be copied directly.

The torchmetrics Pattern A (explicit `update()` in step, `compute()` + `log()` + `reset()` in
`on_*_epoch_end()`) was verified end-to-end with `L.Trainer`. It works correctly and avoids the
NaN/0.0 artifacts that Pattern B (passing metric objects directly to `self.log()`) can produce
when compute is called on empty metric state.

**Primary recommendation:** Copy the `@register` decorator verbatim from the sibling repo's
`utils/hydra.py`, add `hydra-core = "*"` to pixi.toml, use `MulticlassAccuracy` from
`torchmetrics.classification` exclusively (not the legacy `Accuracy` wrapper), and follow the
Pattern A logging discipline enforced in the sibling's base model.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `lightning` (pytorch-lightning) | 2.6.1 | LightningModule base, Trainer, logging hooks | Already in pixi.toml; provides training loop, device management, logging |
| `torchvision` | 0.25.0 | ResNet18_Weights.DEFAULT, ResNet50_Weights.DEFAULT, pretrained backbones | Project decision; verified via live env |
| `torchmetrics` | 1.8.2 | MulticlassAccuracy (top-k, per-class) | Already in pixi env; handles distributed-safe metric accumulation |
| `torch` | 2.10.0 | CrossEntropyLoss, AdamW, LinearLR, CosineAnnealingLR, SequentialLR | Core framework |
| `hydra-core` | 1.3.2 (sibling) | ConfigStore registration, YAML config loading | NOT YET INSTALLED — must be added to pixi.toml |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `omegaconf` | 2.3.0 | DictConfig type hints, config composition | Auto-installed with hydra-core |
| `loguru` | * | Structured logging | Already in pixi.toml, used throughout Phase 1 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torchvision.models` ResNets | `timm` backbones | Project decision locked: use torchvision |
| `MulticlassAccuracy` | Legacy `Accuracy(task='multiclass', ...)` | Both work in torchmetrics 1.8.2; use `MulticlassAccuracy` — explicit and cleaner |
| `SequentialLR(warmup + cosine)` | `OneCycleLR` | SequentialLR matches sibling repo pattern exactly |

**Installation (add to pixi.toml):**
```toml
hydra-core = "*"
```

Then also add to `pyproject.toml` dependencies:
```toml
"hydra-core",
"omegaconf",
"torchmetrics",
```

---

## Architecture Patterns

### Recommended Project Structure

```
src/classifier_training/
├── models/
│   ├── __init__.py          # exports BaseClassificationModel, ResNet18Model, ResNet50Model
│   ├── base.py              # BaseClassificationModel (LightningModule ABC)
│   └── resnet.py            # ResNet18Model, ResNet50Model with @register
├── utils/
│   ├── __init__.py
│   └── hydra.py             # @register decorator (copy from sibling)
└── conf/
    └── models/
        ├── resnet18.yaml    # Hydra config for ResNet18
        └── resnet50.yaml    # Hydra config for ResNet50
```

Note: `conf/` should live under `src/classifier_training/conf/` (package-accessible) or at
project root alongside main `conf/` if a root config exists. The sibling uses
`src/object_detection_training/conf/`. Follow the same pattern.

### Pattern 1: BaseClassificationModel (LightningModule)

**What:** Abstract base LightningModule with CrossEntropyLoss, Pattern A torchmetrics logging,
AdamW + SequentialLR. Concrete models subclass and provide `self.model` (backbone nn.Module).

**When to use:** All classification models in this project.

```python
# Source: verified in live env against lightning 2.6.1 + torchmetrics 1.8.2

from __future__ import annotations

import lightning as L
import torch
from loguru import logger
from torchmetrics.classification import MulticlassAccuracy


class BaseClassificationModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
        warmup_start_factor: float = 1e-3,
        cosine_eta_min_factor: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # class_weights: register as buffer so .to(device) transfers them automatically.
        # Default ones — caller sets via set_class_weights() before/during training.
        self.register_buffer("class_weights", torch.ones(num_classes))

        # Build loss; recreated if weights are updated.
        self._build_loss_fn()

        # Pattern A metrics — one set per split.
        # Top-5 only valid when num_classes >= 5.
        top_k_5 = min(5, num_classes)
        self.train_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.val_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.val_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=top_k_5, average="micro")
        self.val_per_cls = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="none")
        self.test_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.test_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=top_k_5, average="micro")
        self.test_per_cls = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="none")

    def _build_loss_fn(self) -> None:
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Update class weights buffer and rebuild loss function."""
        self.class_weights.copy_(weights.to(self.class_weights.device))
        self._build_loss_fn()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_top1.update(logits, labels)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc_top1", self.train_top1.compute())
        self.train_top1.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_top1.update(logits, labels)
        self.val_top5.update(logits, labels)
        self.val_per_cls.update(logits, labels)

    def on_validation_epoch_end(self) -> None:
        self.log("val/acc_top1", self.val_top1.compute(), prog_bar=True)
        self.log("val/acc_top5", self.val_top5.compute())
        per_cls = self.val_per_cls.compute()
        for i, acc in enumerate(per_cls):
            self.log(f"val/acc_class_{i}", acc)
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_per_cls.reset()

    def configure_optimizers(self) -> dict:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        max_epochs = (self.trainer.max_epochs or 100) if self.trainer else 100
        warmup = self.hparams.warmup_epochs
        cosine_epochs = max(1, max_epochs - warmup)

        warmup_sched = LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup),
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.hparams.learning_rate * self.hparams.cosine_eta_min_factor,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
```

### Pattern 2: Concrete ResNet Model with @register

**What:** Subclass of BaseClassificationModel that wires a specific torchvision backbone and
registers it with Hydra ConfigStore via the `@register` decorator.

**When to use:** Each distinct backbone variant (ResNet18, ResNet50).

```python
# Source: verified against torchvision 0.25.0, follows sibling repo's @register pattern

import torchvision.models as tv_models
from classifier_training.utils.hydra import register
from classifier_training.models.base import BaseClassificationModel


@register(name="ResNet18")
class ResNet18Model(BaseClassificationModel):
    """ResNet18 with pretrained ImageNet weights for jersey number classification."""

    def __init__(self, num_classes: int = 43, **kwargs) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        # backbone.fc.in_features == 512 for ResNet18
        self.model = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


@register(name="ResNet50")
class ResNet50Model(BaseClassificationModel):
    """ResNet50 with pretrained ImageNet weights for jersey number classification."""

    def __init__(self, num_classes: int = 43, **kwargs) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        # backbone.fc.in_features == 2048 for ResNet50
        self.model = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
```

### Pattern 3: @register Decorator (copy from sibling)

**What:** Registers a class with Hydra ConfigStore so `hydra.utils.instantiate(cfg)` can
construct the model from YAML. Infers group from module path.

**When to use:** Any concrete model class that needs Hydra instantiation.

```python
# Source: object-detection-training/src/object_detection_training/utils/hydra.py
# Copy verbatim — no changes needed for classifier-training

from __future__ import annotations
from typing import Any
from hydra.core.config_store import ConfigStore
from loguru import logger


def register(
    cls: type[Any] | None = None,
    *,
    group: str | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> type[Any] | Any:
    def _process_class(target_cls: type[Any]) -> type[Any]:
        nonlocal group, name
        target_path = f"{target_cls.__module__}.{target_cls.__name__}"
        config_name = name or target_cls.__name__
        if group is None:
            module_parts = target_cls.__module__.split(".")
            group = module_parts[-2]
        cs = ConfigStore.instance()
        node = {"_target_": target_path}
        node.update(kwargs)
        cs.store(group=group, name=config_name, node=node)
        return target_cls

    if cls is None:
        return _process_class
    return _process_class(cls)
```

### Pattern 4: Hydra YAML Config for ResNet Models

**What:** Flat-key YAML config consumed by `hydra.utils.instantiate()`. Must not double-wrap.

```yaml
# conf/models/resnet18.yaml
# Source: sibling repo pattern (flat keys, no wrapper node)
_target_: classifier_training.models.resnet.ResNet18Model
num_classes: 43
learning_rate: 1e-4
weight_decay: 1e-4
warmup_epochs: 5
label_smoothing: 0.1
warmup_start_factor: 1e-3
cosine_eta_min_factor: 0.05
```

```yaml
# conf/models/resnet50.yaml
_target_: classifier_training.models.resnet.ResNet50Model
num_classes: 43
learning_rate: 5e-5       # ResNet50 larger model, lower LR appropriate
weight_decay: 1e-4
warmup_epochs: 5
label_smoothing: 0.1
warmup_start_factor: 1e-3
cosine_eta_min_factor: 0.05
```

### Anti-Patterns to Avoid

- **Pattern B metric logging:** `self.log("val/acc", self.val_top1)` — passing the metric
  object directly. Lightning calls `compute()` internally but may not call `reset()` at the
  right time, leading to accumulation across epochs (produces inflated accuracy, NaN, or 0.0
  on first epoch before any updates). Use Pattern A exclusively.

- **CrossEntropyLoss weight on wrong device:** Creating `CrossEntropyLoss(weight=tensor)` in
  `__init__` where `tensor` is on CPU, then moving model to MPS/CUDA. The buffer approach
  (`register_buffer("class_weights", ...)`) ensures automatic device transfer.

- **top_k > num_classes in MulticlassAccuracy:** `MulticlassAccuracy(num_classes=3, top_k=5)`
  raises `ValueError`. Always use `top_k = min(5, num_classes)` for the Top-5 metric.
  For the real dataset (43 classes), top_k=5 is safe. But test fixtures use 3 classes, so
  the base model must guard this.

- **pretrained weights download blocking tests:** `ResNet18_Weights.DEFAULT` triggers a
  ~44MB network download on first call. Tests must either use `weights=None` for speed or
  mock the download. Cache lives at `~/.cache/torch/hub/checkpoints/`.

- **Double-nesting in YAML configs:** Sibling repo learned that config group YAMLs use flat
  keys (no wrapper node around params). A config YAML like `model:\n  lr: 1e-4` would
  cause double-nesting when the ConfigStore already has the group.

- **Hydra group inference from module path:** The `@register` decorator infers group from
  `target_cls.__module__.split(".")[-2]`. For `classifier_training.models.resnet.ResNet18Model`,
  group is inferred as `"models"`. Verify this matches where YAML configs are stored.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Metric accumulation across batches | Custom running-sum accuracy counter | `MulticlassAccuracy` from torchmetrics | Handles distributed, dtype edge cases, NaN masking |
| Per-class accuracy breakdown | Loop over class indices, manual averages | `MulticlassAccuracy(average='none')` | Correct handling of zero-sample classes, no ZeroDivision |
| LR warmup schedule | Custom lambda-LR with ramp logic | `torch.optim.lr_scheduler.LinearLR` | Well-tested, SequentialLR composition supported |
| Cosine LR decay | Custom cosine calculation | `torch.optim.lr_scheduler.CosineAnnealingLR` | Numerically correct, handles eta_min |
| Hydra model registration | Custom config dict | `ConfigStore.instance().store(...)` via `@register` | Pattern already proven in sibling repo |

**Key insight:** torchmetrics 1.8.2 handles all edge cases for multi-class accuracy (distributed
sync, dtype promotion, empty batches, top-k validation). The custom implementations almost always
get one of these wrong.

---

## Common Pitfalls

### Pitfall 1: Hydra-core Not Installed

**What goes wrong:** `ModuleNotFoundError: No module named 'hydra'` on import of `utils/hydra.py`
or any model that uses `@register`. Tests fail immediately.

**Why it happens:** `hydra-core` is missing from `classifier-training/pixi.toml`. The sibling
repo has `hydra-core = "*"` as a dependency; this project was not set up with it.

**How to avoid:** Add `hydra-core = "*"` to `[dependencies]` in `pixi.toml` before any model
code that imports `hydra`. Also add to `pyproject.toml` under `dependencies`.

**Warning signs:** `import hydra` fails in the pixi shell; `pixi list | grep hydra` returns empty.

---

### Pitfall 2: class_weights Device Mismatch

**What goes wrong:** `RuntimeError: Expected all tensors to be on the same device, but found at
least two devices` when CrossEntropyLoss tries to apply weight to logits on MPS/CUDA.

**Why it happens:** CrossEntropyLoss stores `weight` as a plain tensor, not a registered buffer.
When the LightningModule is moved to a non-CPU device, the loss weight stays on CPU.

**How to avoid:** Use `self.register_buffer("class_weights", torch.ones(num_classes))` in
`__init__`. Then rebuild `self.loss_fn = CrossEntropyLoss(weight=self.class_weights, ...)`.
Buffers are automatically transferred by `.to(device)`.

**Warning signs:** Tests pass on CPU but fail on MPS or CUDA during training.

---

### Pitfall 3: NaN / 0.0 in Accuracy Metrics (Pattern B)

**What goes wrong:** `val/acc_top1` logs 0.0 for the first epoch or NaN intermittently.

**Why it happens:** If `self.log("val/acc_top1", self.val_top1)` (Pattern B) is used, Lightning
calls `metric.compute()` but the reset timing doesn't align with epoch boundaries in all versions.
Confirmed: calling `compute()` with no prior `update()` returns `tensor(0.)` with a UserWarning.

**How to avoid:** Enforce Pattern A strictly:
1. `validation_step`: call `self.val_top1.update(logits, labels)` only
2. `on_validation_epoch_end`: call `self.val_top1.compute()`, then `self.log(...)`, then `self.val_top1.reset()`

**Warning signs:** Accuracy logs 0.0 or NaN in first epoch; accuracy suspiciously high in later
epochs (accumulating across epochs without reset).

---

### Pitfall 4: top_k=5 with num_classes < 5 in Tests

**What goes wrong:** `ValueError: Expected argument 'top_k' to be smaller or equal to
'num_classes' but got 5 and 3` when test fixtures use 3-class datasets.

**Why it happens:** The test `conftest.py` creates 3-class datasets (`["0", "1", "2"]`). A
hardcoded `MulticlassAccuracy(num_classes=3, top_k=5)` raises ValueError.

**How to avoid:** In `BaseClassificationModel.__init__`, always compute:
```python
top_k_5 = min(5, num_classes)
self.val_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=top_k_5, average="micro")
```

**Warning signs:** Tests with small `num_classes` fail on model init, not on forward pass.

---

### Pitfall 5: Pretrained Weight Downloads in Tests

**What goes wrong:** Tests are slow (10-45 seconds) because ResNet18/ResNet50 weights are
downloaded from PyTorch Hub during test initialization.

**Why it happens:** `ResNet18_Weights.DEFAULT` downloads ~44MB; `ResNet50_Weights.DEFAULT`
downloads ~98MB. This happens once (cached at `~/.cache/torch/hub/checkpoints/`), but first run
is slow in CI.

**How to avoid:** Test the model class with `weights=None` for unit tests (forward pass shape,
loss computation, metric update/compute). Use `weights=DEFAULT` only for an explicit
`test_pretrained_weights_load` test. The model `__init__` should accept `weights` as a parameter
or expose `pretrained: bool = True` so tests can pass `pretrained=False`.

**Alternative:** Accept that pretrained weights test will be slow; annotate with
`@pytest.mark.slow` if the project uses markers.

---

### Pitfall 6: save_hyperparameters with Tensor Arguments

**What goes wrong:** `TypeError: Object of type Tensor is not JSON serializable` when
`save_hyperparameters()` is called and `class_weights` is a tensor parameter.

**Why it happens:** `save_hyperparameters()` serializes all `__init__` args to the checkpoint
hparams dict. Tensors are not serializable by default.

**How to avoid:** Do NOT pass `class_weights` as a constructor argument. Provide it via
`set_class_weights()` method after construction (called from training script after datamodule
setup). Only scalar/primitive hparams should be passed to `__init__`.

---

## Code Examples

Verified patterns from live environment:

### Constructing Metrics (torchmetrics 1.8.2)

```python
# Source: verified in live env (torchmetrics 1.8.2, python 3.11)
from torchmetrics.classification import MulticlassAccuracy

NUM_CLASSES = 43

# Top-1 overall accuracy
top1 = MulticlassAccuracy(num_classes=NUM_CLASSES, top_k=1, average="micro")

# Top-5 overall accuracy (safe for num_classes >= 5)
top5 = MulticlassAccuracy(num_classes=NUM_CLASSES, top_k=5, average="micro")

# Per-class accuracy: shape (NUM_CLASSES,), one value per class
per_cls = MulticlassAccuracy(num_classes=NUM_CLASSES, top_k=1, average="none")
```

### Pattern A Logging (verified with L.Trainer 2.6.1)

```python
# Source: verified end-to-end with L.Trainer max_epochs=1 in live env

def validation_step(self, batch: dict, batch_idx: int) -> None:
    images, labels = batch["images"], batch["labels"]
    logits = self(images)
    loss = self.loss_fn(logits, labels)
    self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    # Only update — never compute() in step
    self.val_top1.update(logits, labels)
    self.val_top5.update(logits, labels)
    self.val_per_cls.update(logits, labels)

def on_validation_epoch_end(self) -> None:
    # compute() once per epoch, then log, then reset
    self.log("val/acc_top1", self.val_top1.compute(), prog_bar=True)
    self.log("val/acc_top5", self.val_top5.compute())
    per_cls = self.val_per_cls.compute()
    for i, acc in enumerate(per_cls):
        self.log(f"val/acc_class_{i}", acc)
    self.val_top1.reset()
    self.val_top5.reset()
    self.val_per_cls.reset()
```

### ResNet FC Replacement (verified in live env)

```python
# Source: verified with torchvision 0.25.0 in live env

import torchvision.models as tv_models

# ResNet18: fc.in_features == 512, params == 11,689,512
backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)  # 512 -> num_classes

# ResNet50: fc.in_features == 2048, params == 25,557,032
backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)  # 2048 -> num_classes
```

### CrossEntropyLoss with class_weight + label_smoothing (verified)

```python
# Source: verified with torch 2.10.0 in live env

loss_fn = torch.nn.CrossEntropyLoss(
    weight=torch.ones(43, dtype=torch.float32),  # will be replaced by actual weights
    label_smoothing=0.1,
)
# class_weights come from ImageFolderDataModule.get_class_weights()
# which returns shape (num_classes,) inverse-frequency tensor
```

### SequentialLR: Linear Warmup + Cosine Annealing (verified)

```python
# Source: verified with torch 2.10.0 in live env, L.Trainer confirmed max_epochs accessible

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
max_epochs = self.trainer.max_epochs or 100  # self.trainer is set when configure_optimizers runs
warmup_epochs = 5
cosine_epochs = max(1, max_epochs - warmup_epochs)

warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_epochs))
cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-4 * 0.05)
scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

return {
    "optimizer": optimizer,
    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
}
```

### ClassificationBatch TypedDict Unpacking (from Phase 1 types.py)

```python
# Source: classifier_training/types.py (Phase 1)

# ClassificationBatch is a TypedDict with keys "images" and "labels"
def training_step(self, batch: ClassificationBatch, batch_idx: int) -> torch.Tensor:
    images = batch["images"]   # Tensor (B, C, H, W)
    labels = batch["labels"]   # Tensor (B,) int64
    logits = self(images)
    return self.loss_fn(logits, labels)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torchvision.models.resnet18(pretrained=True)` | `resnet18(weights=ResNet18_Weights.DEFAULT)` | torchvision 0.13 | `pretrained=True` is deprecated; use `weights=` API |
| `from torchmetrics import Accuracy` (legacy) | `from torchmetrics.classification import MulticlassAccuracy` | torchmetrics 0.10 | Legacy still works in 1.8.2 but wraps the new class; use explicit class directly |
| `torchvision.transforms.ToTensor()` | `torchvision.transforms.v2.ToImage() + ToDtype(float32, scale=True)` | torchvision 0.15 | `ToTensor()` deprecated in v2; Phase 1 already uses v2 correctly |

**Deprecated/outdated:**
- `pretrained=True` in torchvision model constructors: Use `weights=ResNet18_Weights.DEFAULT`
- `torchmetrics.Accuracy` without task argument: Use `MulticlassAccuracy` directly
- `from torch.optim.lr_scheduler import LambdaLR` for warmup: Use `LinearLR` + `SequentialLR`

---

## Open Questions

1. **Hydra ConfigStore group for models**
   - What we know: The `@register` decorator infers group from `module.__name__.split(".")[-2]`.
     For `classifier_training.models.resnet`, this yields group `"models"`.
   - What's unclear: Does the project need a root `conf/config.yaml` with a `defaults` list that
     references `model: resnet18`? The sibling repo has a full Hydra app structure. This phase
     only needs models registerable for config override; the full app entrypoint may be Phase 3+.
   - Recommendation: For Phase 2, only implement `@register` and YAML configs. Don't wire the
     full Hydra app yet — just verify `hydra.utils.instantiate(cfg)` can construct the models.

2. **class_weights source: datamodule vs YAML**
   - What we know: `ImageFolderDataModule.get_class_weights()` returns a computed
     inverse-frequency tensor (shape 43,). This is dynamic — it changes with training data.
   - What's unclear: Phase 2 requirements say "configurable class_weight tensor". Does this mean
     a fixed YAML tensor, or the dynamic datamodule-computed weights?
   - Recommendation: Accept `class_weights: list[float] | None = None` in YAML (default None),
     and provide `set_class_weights()` method for runtime injection from datamodule. Both
     workflows supported.

3. **Model file structure: one file vs separate**
   - What we know: Sibling has `base.py` + `rfdetr_lightning.py`. Phase 2 plan says 02-01
     (base) and 02-02 (ResNet18 + ResNet50) as separate plans.
   - What's unclear: Should ResNet18 and ResNet50 be in separate files or one `resnet.py`?
   - Recommendation: Single `resnet.py` for both (they share >90% structure). Keeps imports
     simple: `from classifier_training.models.resnet import ResNet18Model, ResNet50Model`.

---

## Infrastructure Action Required

**BEFORE writing any model code**, the following pixi.toml changes are needed:

```toml
# In [dependencies] section of pixi.toml:
hydra-core = "*"
torchmetrics = "*"    # Already available via conda, add explicitly for clarity
```

**Note:** `torchmetrics` is already in the pixi environment (1.8.2) but not listed in
`pixi.toml`. Adding it explicitly makes the dependency explicit. `hydra-core` is the critical
missing dependency.

Also update `pyproject.toml` under `[project] dependencies`:
```toml
"hydra-core",
"omegaconf",
"torchmetrics",
```

And add to `[tool.mypy.overrides] module` in `pyproject.toml`:
```toml
"hydra.*",
"omegaconf.*",
"torchmetrics.*",
```

---

## Sources

### Primary (HIGH confidence)

- Live pixi environment (torch 2.10.0, torchvision 0.25.0, lightning 2.6.1, torchmetrics 1.8.2)
  — all API calls verified by execution
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/models/base.py`
  — BaseDetectionModel: Pattern A reference, configure_optimizers with SequentialLR
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/models/rfdetr_lightning.py`
  — @register usage, SequentialLR construction
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/utils/hydra.py`
  — @register decorator source
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/classifier-training/src/classifier_training/types.py`
  — ClassificationBatch TypedDict (Phase 1)
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/classifier-training/src/classifier_training/data/datamodule.py`
  — get_class_weights() and class_weights computation

### Secondary (MEDIUM confidence)

- `pixi list` output: confirmed hydra-core is NOT in classifier-training env
- `pixi list` output: confirmed torchmetrics 1.8.2 IS installed (from conda-forge)
- torchvision 0.25.0 release notes pattern: `weights=ResNet18_Weights.DEFAULT` API

### Tertiary (LOW confidence)

- Recommended LR values (lr=1e-4 for ResNet18 fine-tuning, lr=5e-5 for ResNet50) — these are
  common fine-tuning defaults but specific values depend on dataset; treat as starting point.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified in live env, all APIs executed successfully
- Architecture: HIGH — sibling repo patterns confirmed working, Pattern A verified with Trainer
- Pitfalls: HIGH — pitfalls 1-6 all triggered or verified in live env during research
- Infrastructure gap (hydra-core missing): HIGH — confirmed by `pixi list` and import failure

**Research date:** 2026-02-18
**Valid until:** 2026-03-20 (30 days; torch/torchvision/lightning stable ecosystem)
