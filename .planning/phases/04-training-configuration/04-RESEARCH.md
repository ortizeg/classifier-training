# Phase 4: Training Configuration - Research

**Researched:** 2026-02-18
**Domain:** Hydra hierarchical config composition, WandB logging, Lightning Trainer configuration, checkpoint resume, training entrypoint
**Confidence:** HIGH

---

## Summary

Phase 4 wires the completed codebase (Phases 1-3) into a launchable training pipeline. Three interconnected deliverables are required: (1) a complete Hydra config tree with config groups for model, data, trainer, callbacks, and logging; (2) a `train.py` entrypoint decorated with `@hydra.main` that instantiates all components and runs `trainer.fit()`; (3) a `pixi run train` task in `pixi.toml` that invokes it. The sibling `object-detection-training` repo has all three patterns implemented and tested in production — this phase is a direct port with classification-specific adaptations.

The most important design constraint in this phase is checkpoint resume stability (TRAIN-07). The default Lightning + Hydra combination creates a new timestamped output directory every run (e.g., `outputs/2026-02-18/14-49-09/`), which causes `ModelCheckpoint`'s `dirpath` to change between runs. When `dirpath` changes, `ckpt_path="last"` in `trainer.fit()` cannot find the previous checkpoint because it looks in the logger's `log_dir` subdirectory. The fix is to set `ModelCheckpoint(dirpath=...)` to a fixed, run-independent path (e.g., `checkpoints/` relative to `Trainer(default_root_dir=...)`) — verified by examining Lightning 2.6.1's `__resolve_ckpt_dir` source.

WandB is not currently installed in the `classifier-training` pixi environment (confirmed). It must be added via `pixi.toml` before `WandbLogger` can be used. The confusion matrix callback (already implemented in Phase 3) saves PNG files to disk; Phase 4 must add WandB image logging by checking `isinstance(trainer.logger, WandbLogger)` in `ConfusionMatrixCallback.on_validation_epoch_end` and calling `trainer.logger.log_image()`.

**Primary recommendation:** Port the sibling's `task_manager.py` pattern into `classifier_training/train.py`; add `conf/train_basketball_resnet18.yaml` as the root Hydra config; add `conf/trainer/default.yaml` with T4-tuned defaults; add `conf/data/basketball_jersey_numbers.yaml`; add `conf/logging/wandb.yaml`; add `wandb` to `pixi.toml`; add `train` task to `pixi.toml`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `hydra-core` | 1.3.2 (installed) | Config composition, CLI overrides | Already in pixi.toml; `@hydra.main` decorates train entrypoint |
| `lightning` | 2.6.1 (installed) | `L.Trainer`, callbacks, loggers | Already in pixi.toml; entire training lifecycle |
| `wandb` | `>=0.23.1,<0.24` | WandB experiment logging | In sibling pixi.toml; TRAIN-05 requires WandbLogger |
| `omegaconf` | via hydra-core | OmegaConf DictConfig, interpolations | Transitive from hydra-core; `${hydra:runtime.output_dir}` interpolations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `loguru` | `*` (installed) | Structured training run logs | Already used throughout; add `setup_loguru()` in train.py |
| `python-dotenv` | via pypi | Load `WANDB_API_KEY` from `.env` | Optional; sibling uses `load_dotenv()` in task_manager |
| `lightning.pytorch.loggers.WandbLogger` | via lightning | WandB run management | Part of lightning; `wandb` package must be installed separately |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `WandbLogger` | `TensorBoardLogger` | TensorBoard doesn't require API key; WandB is locked in TRAIN-05 requirement |
| `@hydra.main` in `train.py` | `hydra.initialize()` + `hydra.compose()` | Programmatic initialization only useful for tests, not the production entrypoint |
| Fixed `dirpath` in callbacks.yaml | `dirpath: ${hydra:runtime.output_dir}/checkpoints` | The latter changes every run, breaking resume; fixed path is the correct TRAIN-07 pattern |

**Dependencies to add to pixi.toml:**
```toml
# [dependencies]
wandb = ">=0.23.1,<0.24"

# [pypi-dependencies]  (optional, for .env loading)
# python-dotenv not strictly required; WANDB_API_KEY can be set via env var directly
```

---

## Architecture Patterns

### Recommended Project Structure (new files for Phase 4)
```
src/classifier_training/
├── train.py                    # NEW: @hydra.main entrypoint
└── conf/
    ├── train_basketball_resnet18.yaml   # NEW: root config (defaults list)
    ├── models/                 # EXISTING: resnet18.yaml, resnet50.yaml
    ├── callbacks/              # EXISTING: default.yaml
    ├── data/
    │   └── basketball_jersey_numbers.yaml  # NEW: dataset config
    ├── trainer/
    │   └── default.yaml        # NEW: T4-tuned trainer config
    └── logging/
        └── wandb.yaml          # NEW: WandbLogger config
```

```
pixi.toml                       # MODIFIED: add wandb dep + train task
src/classifier_training/conf/callbacks/default.yaml  # MODIFIED: fix ModelCheckpoint dirpath
```

### Pattern 1: Root Config with Defaults List
**What:** The root YAML (`conf/train_basketball_resnet18.yaml`) declares the defaults list. Each group entry selects the YAML from that config group directory.
**When to use:** The root config is loaded by `@hydra.main(config_name="train_basketball_resnet18")`.

```yaml
# Source: adapted from sibling's src/object_detection_training/conf/train.yaml
# File: src/classifier_training/conf/train_basketball_resnet18.yaml

defaults:
  - _self_
  - model: resnet18
  - data: basketball_jersey_numbers
  - trainer: default
  - callbacks: default
  - logging: wandb

# Global parameters
seed: 42
log_level: INFO

# Hydra output directory configuration
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: outputs/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

**Critical note on config group key naming:** The existing `conf/models/` directory uses key `model` in YAML references but the directory is named `models/`. The root config defaults list uses `model: resnet18` to select from the `models/` group. Verify the ConfigStore group name matches the `@register` decorator's inferred group from the module path.

**Note on flat keys:** Config group YAMLs use flat keys (no wrapper dict) — this is the established pattern from Phases 1-3. The `_target_` key is at the top level.

### Pattern 2: train.py Entrypoint
**What:** A Python module with `@hydra.main` that imports all registered models, instantiates components, and calls `trainer.fit()`. Must seed everything before instantiating.
**Source:** Adapted from `object-detection-training/src/object_detection_training/task_manager.py`.

```python
# Source: sibling task_manager.py adapted for classification
# File: src/classifier_training/train.py

import sys
import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# CRITICAL: import models to trigger @register decorators
import classifier_training.models  # noqa: F401


@hydra.main(version_base=None, config_path="conf", config_name="train_basketball_resnet18")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO"))

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = HydraConfig.get().runtime.output_dir
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Instantiate data module
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # Set class weights on model after setup
    model = hydra.utils.instantiate(cfg.model)
    model.set_class_weights(datamodule.get_class_weights())

    # Instantiate callbacks from DictConfig
    callbacks = [
        hydra.utils.instantiate(v)
        for v in cfg.callbacks.values()
        if v is not None and "_target_" in v
    ]

    # Instantiate logger(s)
    loggers = []
    if cfg.get("logging"):
        for v in cfg.logging.values():
            if v is not None and "_target_" in v:
                loggers.append(hydra.utils.instantiate(v))

    # Instantiate trainer
    trainer_cfg = dict(cfg.trainer)
    trainer_cfg.pop("_target_", None)
    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=loggers or False,
        default_root_dir=output_dir,
    )

    # Run training (ckpt_path="last" enables automatic resume)
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")


if __name__ == "__main__":
    main()
```

**Key points:**
- `import classifier_training.models` must happen before `@hydra.main` runs to register models in ConfigStore
- `ckpt_path="last"` in `trainer.fit()` is the Lightning 2.6.1 magic keyword that finds and resumes from the last checkpoint automatically (confirmed from inspecting `Trainer.fit` signature)
- `datamodule.setup("fit")` must be called before `model.set_class_weights()` so the train dataset is loaded

### Pattern 3: ModelCheckpoint with Fixed dirpath (TRAIN-07)
**What:** The `dirpath` for `ModelCheckpoint` must be a fixed, run-independent path so that `ckpt_path="last"` in `trainer.fit()` can find the previous checkpoint when resuming.

**Why it breaks without this:** When `dirpath` is `${hydra:runtime.output_dir}/checkpoints`, each run gets a new timestamped path (e.g., `outputs/2026-02-18/14-49-09/checkpoints`). The next run's `ckpt_path="last"` looks in `outputs/2026-02-18/15-00-00/checkpoints` (new dir, empty), finds nothing, and starts from scratch.

**Fix — two options:**

Option A: Fixed dirpath in callbacks/default.yaml (preferred for simplicity):
```yaml
# In conf/callbacks/default.yaml — change dirpath to fixed path
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: checkpoints    # Fixed relative path, resolved to default_root_dir/checkpoints
  filename: "epoch={epoch:02d}-val_acc={val/acc_top1:.4f}"
  monitor: val/acc_top1
  mode: max
  save_top_k: 3
  save_last: true
```

Option B: Override dirpath in train.py when constructing trainer by setting `default_root_dir` to a fixed project-level directory and omitting `dirpath` in ModelCheckpoint (so Lightning resolves to `default_root_dir/checkpoints`).

**Verified from Lightning 2.6.1 source (`ModelCheckpoint.__resolve_ckpt_dir`):** When `dirpath` is set, it is used directly. When `dirpath=None` and a WandbLogger is present, Lightning uses `logger.save_dir + name + version + "/checkpoints"` — which includes the WandB run ID and is NOT stable across runs. Setting explicit `dirpath` is the only reliable fix.

**Recommendation:** Use Option A — set `dirpath: checkpoints` (relative) in `conf/callbacks/default.yaml` so the path resolves to `Trainer.default_root_dir/checkpoints` regardless of Hydra output directory.

### Pattern 4: T4 Trainer Config
**What:** The `conf/trainer/default.yaml` contains Lightning Trainer parameters tuned for NVIDIA T4 GPU (16GB VRAM, 2560 CUDA cores). Classification is less memory-intensive than object detection.

```yaml
# Source: adapted from sibling's conf/trainer/default.yaml
# File: src/classifier_training/conf/trainer/default.yaml
# No _target_ key — Trainer is constructed directly in train.py, not via hydra.utils.instantiate

max_epochs: 100
precision: 16-mixed          # TRAIN-02: AMP for T4 GPU
accelerator: auto
devices: auto
strategy: auto
val_check_interval: 1.0      # Validate at end of each epoch
log_every_n_steps: 10
enable_checkpointing: true
enable_progress_bar: true
enable_model_summary: true
gradient_clip_val: 1.0       # TRAIN-03: clip_val=1.0 for classification stability
gradient_clip_algorithm: norm
accumulate_grad_batches: 1   # TRAIN-03: start at 1, can tune
num_sanity_val_steps: 2
```

**Notes on T4 defaults (TRAIN-04):**
- `batch_size=64` is set in `conf/data/basketball_jersey_numbers.yaml`, not in trainer config
- `num_workers=4` is set in the data config
- T4 has 16GB VRAM; ResNet18 + batch_size=64 + `16-mixed` fits comfortably (estimated ~2GB VRAM)
- `gradient_clip_val=1.0` vs sibling's `5.0`: classification with cross-entropy is less sensitive to gradient norms than detection; `1.0` is the locked decision per Phase 4 requirements (empirical testing needed)

### Pattern 5: Data Config for Basketball Jersey Numbers
**What:** `conf/data/basketball_jersey_numbers.yaml` is the Hydra config for `ImageFolderDataModule`. It uses flat keys matching `DataModuleConfig` field names (Pydantic frozen model).

```yaml
# File: src/classifier_training/conf/data/basketball_jersey_numbers.yaml
_target_: classifier_training.data.datamodule.ImageFolderDataModule

# DataModuleConfig fields — passed as kwargs to ImageFolderDataModule.__init__
# IMPORTANT: ImageFolderDataModule takes a DataModuleConfig, not **kwargs directly
# Must check if instantiation passes config= or unpacked kwargs
data_root: /Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-jersey-numbers-ocr
batch_size: 64               # TRAIN-04: T4 GPU default
num_workers: 4               # TRAIN-04: T4 GPU default
pin_memory: true
persistent_workers: true
image_size: 224
```

**CRITICAL ISSUE — DataModule instantiation mismatch:** `ImageFolderDataModule.__init__` takes `config: DataModuleConfig`, not individual kwargs. `hydra.utils.instantiate(cfg.data)` will pass `data_root=...`, `batch_size=...`, etc. as keyword arguments, but `ImageFolderDataModule.__init__` expects a single `config: DataModuleConfig` argument.

**Resolution options:**
1. Change `ImageFolderDataModule.__init__` to accept `**kwargs` and construct `DataModuleConfig` internally (requires modifying Phase 1 code)
2. Add a `_convert_: all` key to the YAML and restructure `__init__` to accept individual fields
3. Create a Hydra-compatible factory function or alternate constructor

**Recommended approach:** Modify `ImageFolderDataModule.__init__` to accept keyword arguments that get forwarded to `DataModuleConfig`. This is a small change and avoids introducing a factory wrapper. The signature becomes `__init__(self, data_root: str, batch_size: int = 32, ...)` with `DataModuleConfig` constructed internally from the kwargs.

### Pattern 6: WandB Logger Config
**What:** `conf/logging/wandb.yaml` configures `WandbLogger`. The logging group is optional (can be `null` to disable WandB).

```yaml
# Source: adapted from sibling's conf/logging/wandb.yaml
# File: src/classifier_training/conf/logging/wandb.yaml
wandb:
  _target_: lightning.pytorch.loggers.WandbLogger
  project: classifier-training
  name: null    # Auto-generated if null — uses wandb default (random name)
  save_dir: ${hydra:runtime.output_dir}
  log_model: false   # Don't auto-upload checkpoints as artifacts (disk space)
  tags: []
  notes: null
```

**WandB image logging for confusion matrix (TRAIN-03 success criterion):** After Phase 3, `ConfusionMatrixCallback` saves PNG files to disk. Phase 4 must extend `ConfusionMatrixCallback.on_validation_epoch_end` to also log the image to WandB when a `WandbLogger` is active:

```python
# In ConfusionMatrixCallback.on_validation_epoch_end:
from lightning.pytorch.loggers import WandbLogger

for lgr in trainer.loggers:
    if isinstance(lgr, WandbLogger):
        lgr.log_image(
            key="confusion_matrix",
            images=[str(png_path)],
            step=trainer.current_epoch,
        )
```

**Note:** `WandbLogger.log_image()` accepts file paths as strings (confirmed from source inspection). No `wandb` import needed in the callback itself — `lgr.log_image()` handles the `wandb.Image` wrapping internally.

### Pattern 7: pixi.toml train Task
**What:** The `pixi run train` task calls the Python module entrypoint.

```toml
# In pixi.toml [tasks]
train = "python -m classifier_training.train"
```

**Alternative using pyproject.toml scripts entry point:**
```toml
# In pyproject.toml [project.scripts]
train = "classifier_training.train:main"
```
Then in `pixi.toml`:
```toml
train = "train"
```

**Recommendation:** Use the `python -m` approach in `pixi.toml` for simplicity — avoids requiring package reinstall after adding the script. The sibling uses a registered console script (`task_manager`) but the `python -m` approach is equally clean and doesn't require a pyproject.toml change.

### Anti-Patterns to Avoid

- **`ckpt_path=None` in trainer.fit():** This disables resume. Use `ckpt_path="last"` to enable automatic resume from the last checkpoint.
- **`dirpath: ${hydra:runtime.output_dir}/checkpoints` in callbacks/default.yaml:** Each run gets a new Hydra output dir, so this path changes every run. `ckpt_path="last"` cannot find the previous run's checkpoints. Use a fixed `dirpath`.
- **Instantiating ImageFolderDataModule without adapting the __init__ signature:** Current `__init__` takes `config: DataModuleConfig`. `hydra.utils.instantiate` passes flat kwargs. These are incompatible without an adapter.
- **Importing wandb directly in callbacks:** Use `isinstance(lgr, WandbLogger)` guard and call `lgr.log_image()` — WandbLogger handles the `import wandb` internally only when needed.
- **Setting `_target_` in trainer YAML if constructing Trainer directly:** If `train.py` constructs `L.Trainer(**trainer_cfg)`, remove `_target_` from the trainer config to avoid the key being passed to Trainer as an unknown kwarg.
- **`logging: null` in root config defaults list:** When `logging` is null, the logger loop must handle `cfg.get("logging") is None` gracefully — do not call `.values()` on None.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config composition | Custom YAML loading | Hydra defaults list | Hydra handles interpolations, overrides, multirun sweeps |
| Checkpoint resume | Custom "find last checkpoint" logic | `ckpt_path="last"` in `trainer.fit()` | Lightning 2.6.1 natively searches for `last.ckpt` in the resolved checkpoint directory |
| WandB run management | `wandb.init()` + `wandb.log()` directly | `WandbLogger` from Lightning | WandbLogger handles step alignment with Lightning's global_step, distributed safety |
| Mixed precision | Manual `torch.cuda.amp.autocast()` | `precision="16-mixed"` in Trainer | Lightning's precision plugin handles scaler, backward, unscaling automatically |
| Gradient clipping | Manual `torch.nn.utils.clip_grad_norm_()` | `gradient_clip_val=1.0` in Trainer | Lightning clips before optimizer step in the correct order |
| Gradient accumulation | Custom step counter + zero_grad logic | `accumulate_grad_batches=N` in Trainer | Lightning handles `global_step` tracking, logging alignment, and scheduler stepping correctly |

---

## Common Pitfalls

### Pitfall 1: ModelCheckpoint dirpath Changes Every Hydra Run — Resume Breaks
**What goes wrong:** `pixi run train` starts a fresh run every time instead of resuming.
**Why it happens:** Hydra creates `outputs/2026-02-18/14-49-09/` for each run. `ModelCheckpoint(dirpath="${hydra:runtime.output_dir}/checkpoints")` points to this timestamped dir. On re-run, Hydra creates `outputs/2026-02-18/15-00-00/` — new dir, no checkpoint found.
**How to avoid:** Set `dirpath: checkpoints` (fixed relative path) in `conf/callbacks/default.yaml`. Lightning resolves relative `dirpath` against `Trainer.default_root_dir`. Set `default_root_dir` in `train.py` to a stable project-level path (e.g., `HydraConfig.get().runtime.cwd` or a fixed `runs/` directory).
**Warning signs:** WandB step counter resets to 0 on restart; training logs say "epoch 0" despite previous runs.

### Pitfall 2: ImageFolderDataModule __init__ Signature Mismatch
**What goes wrong:** `hydra.utils.instantiate(cfg.data)` raises `TypeError: __init__() got unexpected keyword argument 'data_root'`.
**Why it happens:** `ImageFolderDataModule.__init__(self, config: DataModuleConfig)` expects a single `DataModuleConfig` object. Hydra passes individual kwargs from the YAML (e.g., `data_root=...`, `batch_size=...`).
**How to avoid:** Modify `ImageFolderDataModule.__init__` to accept individual keyword arguments matching `DataModuleConfig` fields and construct `DataModuleConfig` internally. Alternative: wrap `DataModuleConfig` construction in a Hydra `_target_`-able factory.
**Warning signs:** `TypeError` on `hydra.utils.instantiate(cfg.data)` at startup.

### Pitfall 3: wandb Not Installed — WandbLogger Import Fails
**What goes wrong:** `from lightning.pytorch.loggers import WandbLogger` succeeds (WandbLogger is defined in Lightning with lazy wandb import), but constructing `WandbLogger(...)` raises `ModuleNotFoundError: No module named 'wandb'`.
**Why it happens:** `wandb` is not in `classifier-training`'s `pixi.toml` (confirmed — not currently installed).
**How to avoid:** Add `wandb = ">=0.23.1,<0.24"` to `[dependencies]` in `pixi.toml`. Run `pixi install`.
**Warning signs:** `ModuleNotFoundError: No module named 'wandb'` when the training script tries to instantiate `WandbLogger`.

### Pitfall 4: models Not Imported Before @hydra.main — ConfigStore Empty
**What goes wrong:** `hydra.utils.instantiate(cfg.model)` raises `hydra.errors.InstantiationException` because the `model` group has no registered configs.
**Why it happens:** `@register` decorators in `models/resnet.py` only fire when the module is imported. If `classifier_training.models` is not imported before Hydra parses the config, the ConfigStore is empty when the defaults list references `model: resnet18`.
**How to avoid:** Add `import classifier_training.models  # noqa: F401` at the top of `train.py`, before `@hydra.main` is invoked.
**Warning signs:** `HydraException: Could not load 'model/resnet18'` or similar config resolution error.

### Pitfall 5: AMP + Gradient Clipping Interaction on T4
**What goes wrong:** Training diverges (loss NaN) in early epochs when using `precision="16-mixed"` with `gradient_clip_val=1.0`.
**Why it happens:** With AMP, gradients can have very large magnitudes in fp16 before the GradScaler adjusts. Clipping at `1.0` is aggressive and may interfere with gradient flow in early epochs before the GradScaler stabilizes.
**How to avoid:** Monitor training loss in the first few epochs. If divergence occurs, try `gradient_clip_val=5.0` (sibling's value for detection). Per the prior decision note, `1.0` vs `5.0` requires empirical testing. The plan should include a validation step that specifically checks for NaN loss.
**Warning signs:** `train/loss` = NaN or inf in WandB after the first batch; GradScaler reduces scale factor to 1 after repeated underflows.

### Pitfall 6: val/acc_top1 Not Logged Before Callbacks Check It
**What goes wrong:** `ModelCheckpoint` and `EarlyStopping` cannot monitor `val/acc_top1` because it's not logged during the sanity check.
**Why it happens:** Lightning runs `num_sanity_val_steps` before training to verify the val dataloader works. During sanity, `on_validation_epoch_end` fires and logs `val/acc_top1` — but `ModelCheckpoint` should work correctly since the metric is logged per epoch. This is NOT a likely issue but is worth noting.
**How to avoid:** Confirmed safe: `BaseClassificationModel.on_validation_epoch_end()` calls `self.log("val/acc_top1", ...)` which Lightning makes available to monitoring callbacks. Verify by looking for `UserWarning: ModelCheckpoint(monitor='val/acc_top1') could not find monitored key` in output.

### Pitfall 7: Hydra config_path Resolves Relative to train.py Location
**What goes wrong:** `@hydra.main(config_path="conf")` cannot find YAML files.
**Why it happens:** Hydra resolves `config_path="conf"` relative to the Python file containing `@hydra.main`. If `train.py` is at `src/classifier_training/train.py`, then `config_path="conf"` looks for `src/classifier_training/conf/` — which is where the conf directory is. This is correct and matches the sibling pattern.
**How to avoid:** Place `train.py` at `src/classifier_training/train.py` (same level as `conf/`). Do not place it at the repo root or elsewhere.
**Warning signs:** `MissingConfigException: Cannot find primary config 'train_basketball_resnet18'`.

---

## Code Examples

Verified patterns from codebase inspection and installed library source:

### ckpt_path="last" for Automatic Resume (Lightning 2.6.1)
```python
# Source: Inspected lightning/pytorch/trainer/trainer.py fit() signature
# "last" keyword: finds last.ckpt in the ModelCheckpoint dirpath

trainer.fit(model, datamodule=datamodule, ckpt_path="last")
# If no previous checkpoint exists: silently starts from scratch
# If last.ckpt exists: resumes from it (restores optimizer state, epoch counter, etc.)
```

### hydra.utils.instantiate for Callbacks DictConfig
```python
# Source: sibling task_manager.py / utils/hydra.py instantiate_callbacks
# cfg.callbacks is a DictConfig with keys like model_checkpoint, early_stopping, etc.
# Each value is a DictConfig with _target_ and params

callbacks = [
    hydra.utils.instantiate(v)
    for k, v in cfg.callbacks.items()
    if v is not None and "_target_" in v
]
# Returns list of L.Callback instances
```

### WandbLogger Instantiation (Lightning 2.6.1 confirmed signature)
```python
# Source: Inspected WandbLogger.__init__ in installed Lightning 2.6.1
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(
    project="classifier-training",
    name=None,           # auto-generated run name
    save_dir="outputs/", # where wandb/ run data is stored
    log_model=False,     # don't upload checkpoints as artifacts
)
# Requires: wandb package installed + WANDB_API_KEY env var set
```

### ModelCheckpoint Fixed dirpath Pattern
```python
# Source: Lightning 2.6.1 __resolve_ckpt_dir source (inspected directly)
# When dirpath is not None, it is used directly (no logger-based path construction)

from lightning.pytorch.callbacks import ModelCheckpoint

ckpt = ModelCheckpoint(
    dirpath="checkpoints",           # Fixed path relative to default_root_dir
    filename="epoch={epoch:02d}-val_acc={val/acc_top1:.4f}",
    monitor="val/acc_top1",
    mode="max",
    save_top_k=3,
    save_last=True,                  # Creates last.ckpt for ckpt_path="last" to find
)
```

### Trainer Construction with Correct Keys
```python
# Source: Inspected L.Trainer.__init__ signature in installed Lightning 2.6.1
import lightning as L

trainer = L.Trainer(
    max_epochs=100,
    precision="16-mixed",            # TRAIN-02: AMP
    gradient_clip_val=1.0,           # TRAIN-03: gradient clipping
    accumulate_grad_batches=1,       # TRAIN-03: gradient accumulation (start at 1)
    accelerator="auto",
    devices="auto",
    strategy="auto",
    val_check_interval=1.0,
    log_every_n_steps=10,
    enable_checkpointing=True,
    num_sanity_val_steps=2,
    default_root_dir="runs/basketball_resnet18",  # Fixed path for resume stability
    callbacks=[...],
    logger=[wandb_logger],
)
```

### DataModuleConfig -> ImageFolderDataModule Adapter Pattern
```python
# Source: Inspected classifier_training/data/datamodule.py and config.py
# Current __init__: ImageFolderDataModule(config: DataModuleConfig)
# hydra.utils.instantiate passes: ImageFolderDataModule(data_root=..., batch_size=..., ...)
# Fix: change __init__ to accept **kwargs and build DataModuleConfig internally

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
        self._config = DataModuleConfig(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            image_size=image_size,
        )
        # ... rest of __init__ uses self._config
```

### Confusion Matrix WandB Image Logging
```python
# Source: Inspected WandbLogger.log_image signature in Lightning 2.6.1
# log_image accepts: list of image paths (str), PIL Images, numpy arrays, or tensors

from lightning.pytorch.loggers import WandbLogger

# In ConfusionMatrixCallback.on_validation_epoch_end:
png_path = save_dir / "confusion_matrix.png"
self._plot_and_save(cm_tensor, trainer.current_epoch, png_path)

for lgr in trainer.loggers:
    if isinstance(lgr, WandbLogger):
        lgr.log_image(
            key="confusion_matrix",
            images=[str(png_path)],   # accepts file path strings
            step=trainer.current_epoch,
        )
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ckpt_path=None` for no resume | `ckpt_path="last"` for auto-resume | Lightning 1.8+ | No custom resume logic needed |
| Manual Trainer construction | Hydra `hydra.utils.instantiate(cfg.trainer)` | Hydra 1.0+ | Cleaner, but requires removing `_target_` from config if not using instantiate |
| `wandb.init()` directly | `WandbLogger` in Lightning | Lightning 1.x | Handles distributed, global_step alignment |
| `precision=16` | `precision="16-mixed"` | Lightning 2.x | String form required in Lightning 2.x; integer `16` deprecated |

**Deprecated/outdated:**
- `precision=16` (integer): Deprecated in Lightning 2.x. Use `"16-mixed"` string form. Confirmed valid from `L.Trainer.__init__` type hint.
- `gpus=N` Trainer parameter: Replaced by `devices=N` + `accelerator="gpu"` in Lightning 1.8+. Use `accelerator="auto"`, `devices="auto"` for portability.

---

## Open Questions

1. **ImageFolderDataModule __init__ signature — modify or factory?**
   - What we know: Current signature takes `config: DataModuleConfig`. Hydra instantiation passes flat kwargs. These are incompatible.
   - What's unclear: Whether the test suite for Phase 1 tests construct DataModule with `config=DataModuleConfig(...)` or with kwargs — changing the signature could break existing tests.
   - Recommendation: Modify `__init__` to accept individual kwargs (matching DataModuleConfig fields) and construct DataModuleConfig internally. Update existing tests if they call with `config=DataModuleConfig(...)` positionally. This is the cleanest Hydra-compatible pattern.

2. **Fixed checkpoint dirpath — relative to what?**
   - What we know: Lightning resolves `dirpath="checkpoints"` (relative path) against... unclear. `ModelCheckpoint.__resolve_ckpt_dir` uses `self.dirpath` directly if set — it does NOT resolve against `default_root_dir` if `dirpath` is a relative path.
   - What's unclear: Whether `dirpath="checkpoints"` resolves relative to CWD or `default_root_dir`.
   - Recommendation: Use an absolute path via `${hydra:runtime.cwd}/checkpoints` in YAML, or pass `dirpath` programmatically in `train.py` as an absolute path computed from `HydraConfig.get().runtime.cwd`. This eliminates ambiguity.

3. **WandB `log_model` — should checkpoints be uploaded as artifacts?**
   - What we know: `WandbLogger(log_model=True)` uploads checkpoint files to WandB as artifacts. The sibling uses `log_model=True`.
   - What's unclear: Whether the basketball jersey number training runs should auto-upload checkpoints (adds storage cost and upload time).
   - Recommendation: Default to `log_model=False` for Phase 4. Can be overridden via `logging.wandb.log_model=true` Hydra override.

4. **AMP stability — clip_val=1.0 or 5.0?**
   - What we know: TRAIN-03 requires `clip_val=1.0`. Prior STATE.md notes "AMP stability on T4 with classification (clip_val=1.0 vs 5.0) needs empirical testing".
   - What's unclear: Whether `clip_val=1.0` is stable with `16-mixed` for ResNet18 on the basketball dataset.
   - Recommendation: Implement with `clip_val=1.0` as specified in requirements. Add a verification step in PLAN.md that checks for NaN loss in the first 5 batches of a dry run. If NaN occurs, the plan must note that `gradient_clip_val=5.0` is the fallback.

5. **How does the `logging` config group interact with null?**
   - What we know: The root config can have `logging: null` to disable WandB. The train.py must handle this case.
   - What's unclear: Whether Hydra's `optional: true` syntax in the defaults list is needed or if checking `cfg.get("logging") is None` suffices.
   - Recommendation: Use `- logging: null` in the defaults list with `optional: true` override (Hydra 1.3 syntax: `- optional logging: wandb`). In `train.py`, check `cfg.get("logging") is None` before iterating loggers.

---

## Sources

### Primary (HIGH confidence)
- Installed `lightning` 2.6.1 source — `Trainer.__init__` signature, `Trainer.fit` `ckpt_path` docs, `ModelCheckpoint.__resolve_ckpt_dir` source, `WandbLogger.__init__` and `log_image` signatures — all inspected directly via `inspect.getsource()`
- `classifier-training` source — `src/classifier_training/data/datamodule.py`, `src/classifier_training/config.py`, `src/classifier_training/models/resnet.py`, `src/classifier_training/utils/hydra.py`, `src/classifier_training/conf/callbacks/default.yaml`, `src/classifier_training/conf/models/resnet18.yaml` — read directly
- Sibling `object-detection-training` source — `src/object_detection_training/task_manager.py`, `src/object_detection_training/tasks/train_task.py`, `src/object_detection_training/conf/train.yaml`, `src/object_detection_training/conf/trainer/default.yaml`, `src/object_detection_training/conf/logging/wandb.yaml` — read directly
- Installed Hydra 1.3.2 — confirmed installed; `config_path="conf"` resolves relative to decorated function's module file
- Direct environment test — `wandb` NOT installed in `classifier-training` pixi env (confirmed `ModuleNotFoundError`); `WandbLogger` importable from Lightning without wandb installed (lazy import)

### Secondary (MEDIUM confidence)
- Sibling `pixi.toml` — `wandb = ">=0.23.1,<0.24"` confirmed working combination; `wandb` 0.23.1 installed in sibling environment
- `classifier-training` `.planning/STATE.md` — prior decisions, AMP stability concern flagged explicitly

### Tertiary (LOW confidence)
- AMP + clip_val=1.0 stability claim — based on general knowledge that classification is less sensitive to gradient magnitudes than detection; not empirically validated on this specific dataset/GPU combination

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library versions confirmed from installed packages; wandb absence confirmed
- Architecture patterns: HIGH — sibling source read directly; Lightning API verified from installed source; Hydra config path behavior verified
- Pitfalls: HIGH — ModelCheckpoint dirpath issue verified from Lightning source; DataModule signature mismatch verified from reading the code; wandb absence confirmed empirically
- AMP stability: LOW — requires empirical validation; flagged as open question

**Research date:** 2026-02-18
**Valid until:** 2026-04-18 (Lightning 2.6.x stable; Hydra 1.3.x stable; wandb 0.23.x stable)

---

## File Inventory: Existing vs New

### Existing files that need modification:
| File | Change Required |
|------|----------------|
| `pixi.toml` | Add `wandb = ">=0.23.1,<0.24"` to `[dependencies]`; add `train = "python -m classifier_training.train"` to `[tasks]` |
| `src/classifier_training/data/datamodule.py` | Change `__init__(self, config: DataModuleConfig)` to accept individual kwargs + construct DataModuleConfig internally |
| `src/classifier_training/conf/callbacks/default.yaml` | Change `model_checkpoint.dirpath` to fixed path (not `${hydra:runtime.output_dir}/checkpoints`) |
| `src/classifier_training/callbacks/confusion_matrix.py` | Add WandB image logging in `on_validation_epoch_end` when `WandbLogger` is present |

### New files to create:
| File | Purpose |
|------|---------|
| `src/classifier_training/train.py` | `@hydra.main` entrypoint — instantiates all components, calls `trainer.fit()` |
| `src/classifier_training/conf/train_basketball_resnet18.yaml` | Root config with defaults list |
| `src/classifier_training/conf/trainer/default.yaml` | T4-tuned Trainer config (precision, clipping, accumulation) |
| `src/classifier_training/conf/data/basketball_jersey_numbers.yaml` | Dataset config for basketball-jersey-numbers-ocr |
| `src/classifier_training/conf/logging/wandb.yaml` | WandbLogger config |
| `src/classifier_training/conf/trainer/__init__.py` | Empty `__init__.py` for Hydra to discover the directory (may be required) |
| `src/classifier_training/conf/data/__init__.py` | Empty `__init__.py` for Hydra to discover the directory (may be required) |
| `src/classifier_training/conf/logging/__init__.py` | Empty `__init__.py` for Hydra to discover the directory (may be required) |
| `tests/test_train_config.py` | Tests: Hydra config composes correctly, train.py entrypoint smoke test, resume test |

### Tests required:
| Test | What It Verifies |
|------|-----------------|
| `test_hydra_config_composes` | Root config loads without error; all defaults resolve |
| `test_trainer_config_values` | `precision`, `gradient_clip_val`, `accumulate_grad_batches` are correct in composed config |
| `test_data_config_instantiates` | `hydra.utils.instantiate(cfg.data)` produces `ImageFolderDataModule` |
| `test_model_config_instantiates` | `hydra.utils.instantiate(cfg.model)` produces `ResNet18ClassificationModel` |
| `test_hydra_override_model` | `model=resnet50` override changes the model class |
| `test_hydra_override_batch_size` | `data.batch_size=32` override propagates to the DataModule |
| `test_resume_creates_last_ckpt` | After a training run, `last.ckpt` exists in the checkpoint directory |
| `test_checkpoint_dirpath_is_stable` | `ModelCheckpoint.dirpath` does not change between two consecutive runs |
