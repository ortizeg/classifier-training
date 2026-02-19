# Phase 3: Callbacks and ONNX Export - Research

**Researched:** 2026-02-18
**Domain:** PyTorch Lightning callbacks, ONNX export, classification visualization
**Confidence:** HIGH

---

## Summary

Phase 3 is a port-and-adapt job, not a build-from-scratch job. The sibling `object-detection-training` repository already contains working implementations of every callback required by this phase. The porting task is well-scoped: replace detection-specific types (`DetectionBatch`, `DetectionDataset`, `TrackingWeightedRandomSampler`) with the classification equivalents already defined in `classifier-training`, and adapt the ONNX export to output `["logits"]` instead of `["dets", "labels"]`.

Three new dependencies must be added to `pixi.toml` before any callback code can be written: `matplotlib` (for training history plots), `onnx` (for `torch.onnx.export`), and `onnxruntime` (for export validation tests). The sibling project uses `onnxscript >= 0.5.6, < 0.6` alongside `onnx` as part of the legacy ONNX exporter path — this pairing is required. The `onnx` package itself is not listed separately in the sibling's `pixi.toml` but is a transitive dependency of `onnxscript`. However, `torch.onnx.export` at opset 17 requires `onnx` to be importable at export time, so it must be declared explicitly.

The critical architectural decision for this phase is the EMA + ModelCheckpoint hook ordering. In Lightning 2.6.1, `on_train_epoch_end` fires for non-monitoring callbacks first (including EMACallback), then calls `pl_module.on_train_epoch_end`, then fires for monitoring callbacks (`ModelCheckpoint` and `EarlyStopping`). When `check_val_every_n_epoch=1` and `val_check_interval=1.0`, `ModelCheckpoint` uses `on_validation_end` — at which point EMA has already swapped in EMA weights for validation and swapped them back. This means the checkpoint saved by `ModelCheckpoint` contains the **original training weights**, not EMA weights. The `ONNXExportCallback` must therefore load EMA weights from `EMACallback.ema_state_dict` directly rather than relying on the checkpoint's `state_dict`.

**Primary recommendation:** Port all seven sibling callbacks directly; add `matplotlib`, `onnx`, `onnxscript`, and `onnxruntime` to `pixi.toml`; implement `ONNXExportCallback` to load EMA state from `EMACallback.ema_state_dict` at `on_train_end`; use `torchmetrics.classification.MulticlassConfusionMatrix` (already installed) to generate confusion matrices without `seaborn`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `lightning` | 2.6.1 (locked) | Callback base class, hook lifecycle | Already in pixi.toml; all callbacks extend `L.Callback` |
| `torch` | 2.10.0 (locked) | ONNX export via `torch.onnx.export` | Already in pixi.toml; provides the export primitive |
| `onnx` | via onnxscript | ONNX graph validation and saving | Required by `torch.onnx.export` at runtime |
| `onnxscript` | `>=0.5.6,<0.6` | Legacy ONNX exporter support | Sibling uses this pinned range; enables `dynamo=False` path |
| `onnxruntime` | `>=1.23.2,<2` | ONNX inference validation in tests | Sibling uses this range; CPUExecutionProvider for tests |
| `matplotlib` | `*` | Training history plots (loss/acc curves) | Sibling uses it; standard for PNG generation |
| `torchmetrics` | 1.8.2 (locked) | `MulticlassConfusionMatrix` for heatmaps | Already installed; avoids seaborn dependency for confusion matrix |
| `rich` | `*` (locked) | Terminal tables for stats callbacks | Already in pixi.toml |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `seaborn` | `*` | Enhanced confusion matrix heatmaps | Optional; NOT needed — `torchmetrics` + `matplotlib` suffices |
| `pandas` | `*` | Tabular data for sampler distribution | Only needed if `SamplerDistributionCallback` uses DataFrame-based reporting like the sibling |
| `wandb` | `*` | WandB logger integration for confusion matrix images | Only needed in Phase 4 (Training Config); callbacks should be WandB-agnostic |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torchmetrics.MulticlassConfusionMatrix` | `seaborn.heatmap` | torchmetrics is already installed, device-aware, and produces the tensor; seaborn adds a new dependency just for plotting |
| `torch.onnx.export` (legacy) | `torch.onnx.dynamo_export` | dynamo exporter requires `onnxscript` + additional setup; legacy is more stable for ResNet and matches sibling pattern |
| Direct EMA state dict injection | Lightning `StochasticWeightAveraging` | SWA is built-in but has different decay scheduling; EMA from sibling is the locked decision |

**Installation — additions to `pixi.toml` required:**
```toml
# [dependencies]
matplotlib = "*"
onnxscript = ">=0.5.6,<0.6"
pandas = "*"

# [pypi-dependencies]
onnxruntime = ">=1.23.2, <2"
```

Note: `onnx` is a transitive dependency of `onnxscript` and will be present after adding `onnxscript`. Verify by running `pixi run python -c "import onnx"` after `pixi install`.

---

## Architecture Patterns

### Recommended Project Structure
```
src/classifier_training/
├── callbacks/
│   ├── __init__.py          # exports all callback classes
│   ├── ema.py               # EMACallback (port from sibling, adapt types)
│   ├── onnx_export.py       # ONNXExportCallback (adapt to logits + EMA state)
│   ├── confusion_matrix.py  # ConfusionMatrixCallback (NEW — not in sibling)
│   ├── model_info.py        # ModelInfoCallback (port from sibling, adapt input dims)
│   ├── statistics.py        # DatasetStatisticsCallback (simplified for classification)
│   ├── plotting.py          # TrainingHistoryCallback (adapt metrics keys)
│   ├── visualization.py     # SampleVisualizationCallback (adapt for classification)
│   └── sampler.py           # SamplerDistributionCallback (adapt for WeightedRandomSampler)
├── conf/
│   └── callbacks/
│       └── default.yaml     # All callbacks wired with _target_ keys
└── models/
    └── base.py              # Add export_onnx() method here OR in callback
```

### Pattern 1: EMA Callback (Direct Port from Sibling)

**What:** Maintains exponential moving average of model weights. Swaps EMA weights in for validation/testing, restores original afterward.

**Key adaptation:** Change `DetectionBatch` type annotation to `ClassificationBatch`. All other logic is identical.

```python
# Source: object-detection-training/src/object_detection_training/callbacks/ema.py
# Adaptation: replace DetectionBatch with ClassificationBatch in on_train_batch_end signature

from classifier_training.types import ClassificationBatch  # replaces DetectionBatch

def on_train_batch_end(
    self,
    trainer: L.Trainer,
    pl_module: L.LightningModule,
    outputs: torch.Tensor | Mapping[str, Any] | None,
    batch: ClassificationBatch,   # <-- changed from DetectionBatch
    batch_idx: int,
) -> None:
    ...
```

**Also add to `classifier_training/types.py`:**
```python
class EMAState(TypedDict, total=False):
    ema_state_dict: dict[str, torch.Tensor]
    step_count: int
    decay: float

class ONNXExportState(TypedDict):
    exported_checkpoints: list[str]
```

### Pattern 2: ONNX Export from EMA State Dict

**What:** At `on_train_end`, load EMA weights from `EMACallback.ema_state_dict` into the model temporarily, export to ONNX with `output_names=["logits"]`, restore original weights.

**Critical:** The checkpoint saved by `ModelCheckpoint` does NOT contain EMA weights (see hook ordering analysis below). The `ONNXExportCallback` must find the `EMACallback` instance in `trainer.callbacks` and access `.ema_state_dict` directly.

```python
# Source: adapted from sibling's ONNXExportCallback and rfdetr_lightning.export_onnx
import os
import copy
import torch

def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    # Find EMA callback if present
    ema_cb = next(
        (cb for cb in trainer.callbacks if isinstance(cb, EMACallback)),
        None
    )

    # Determine which weights to export
    export_state_dict = None
    if ema_cb and ema_cb.ema_state_dict:
        export_state_dict = ema_cb.ema_state_dict
        logger.info("Exporting ONNX from EMA weights")
    else:
        export_state_dict = pl_module.state_dict()
        logger.info("No EMA found; exporting from current model weights")

    # Save original, load export weights
    original_state = copy.deepcopy(pl_module.state_dict())
    pl_module.load_state_dict(export_state_dict)

    try:
        self._export_to_onnx(pl_module, output_path)
    finally:
        pl_module.load_state_dict(original_state)

def _export_to_onnx(
    self,
    pl_module: L.LightningModule,
    output_path: Path,
) -> None:
    # Use legacy exporter path (same pattern as sibling)
    os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"
    original_export = torch.onnx.export

    def patched_export(*args: Any, **kwargs: Any) -> Any:
        kwargs["dynamo"] = False
        return original_export(*args, **kwargs)

    torch.onnx.export = patched_export

    model = copy.deepcopy(pl_module).cpu().eval()
    dummy_input = torch.randn(1, 3, self.input_height, self.input_width)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=["input"],
                output_names=["logits"],          # <-- classification output
                opset_version=self.opset_version,
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "logits": {0: "batch_size"},  # <-- classification axes
                },
            )
    finally:
        torch.onnx.export = original_export
```

### Pattern 3: Confusion Matrix Callback (New — no sibling equivalent)

**What:** Per-epoch validation confusion matrix as PNG heatmap using `torchmetrics.classification.MulticlassConfusionMatrix` + `matplotlib`. Avoids seaborn dependency.

**Key points:**
- Accumulate predictions and labels during `on_validation_batch_end`
- Compute and plot in `on_validation_epoch_end`
- `MulticlassConfusionMatrix` is already installed (torchmetrics 1.8.2)
- Must call `.cpu()` before passing to matplotlib — GPU tensors cannot be plotted directly

```python
# Source: torchmetrics 1.8.2 API (verified)
from torchmetrics.classification import MulticlassConfusionMatrix

class ConfusionMatrixCallback(L.Callback):
    def __init__(self, num_classes: int, output_dir: str = "outputs"):
        super().__init__()
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self._cm: MulticlassConfusionMatrix | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Initialize on correct device after model is placed
        num_classes = getattr(pl_module, "hparams", {}).get("num_classes", self.num_classes)
        self._cm = MulticlassConfusionMatrix(num_classes=num_classes).to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: ClassificationBatch,
        batch_idx: int,
    ) -> None:
        if self._cm is None:
            return
        images, labels = batch["images"], batch["labels"]
        with torch.no_grad():
            logits = pl_module(images.to(pl_module.device))
            preds = logits.argmax(dim=1)
        self._cm.update(preds, labels.to(pl_module.device))

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._cm is None:
            return
        cm_tensor = self._cm.compute().cpu()  # MUST move to CPU before matplotlib
        self._cm.reset()
        self._plot_and_save(cm_tensor, trainer.current_epoch)

    def _plot_and_save(self, cm: torch.Tensor, epoch: int) -> None:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for server/CI
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm.numpy(), aspect="auto", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — Epoch {epoch}")
        save_dir = self.output_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
```

### Pattern 4: DatasetStatisticsCallback (Simplified for Classification)

**What:** Print class distribution table using `rich` at training start. Classification version is much simpler than detection version — no bounding boxes, just label counts.

**Key adaptation:** Access `datamodule._train_dataset.samples` (list of `(path, label_int)` tuples — see `JerseyNumberDataset`) instead of `DetectionDataset`.

```python
def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    dm = getattr(trainer, "datamodule", None)
    if dm is None:
        return
    train_ds = getattr(dm, "_train_dataset", None)
    if train_ds is None:
        return
    # Build count per class index
    counts: dict[int, int] = {}
    for _, label in train_ds.samples:
        counts[label] = counts.get(label, 0) + 1
    # Print rich table using class_to_idx for names
    idx_to_class = {v: k for k, v in dm.class_to_idx.items()}
    ...
```

### Pattern 5: SamplerDistributionCallback (Simplified for Classification)

**What:** Log actual class counts sampled per epoch from `WeightedRandomSampler`. Unlike the sibling (which uses a `TrackingWeightedRandomSampler` that records indices), the `classifier-training` DataModule uses the standard PyTorch `WeightedRandomSampler` which does not expose sampled indices.

**Resolution:** Two options:
1. **Introspect sampler weights** — read `sampler.weights` and `sampler.num_samples` to compute expected distribution without tracking actual samples
2. **Replace sampler with a tracking version** — subclass `WeightedRandomSampler` to record `_last_indices`, matching the sibling pattern

Option 2 is the cleaner approach and matches what the sibling did. Add a `TrackingWeightedRandomSampler` to `classifier_training/data/` and use it in `ImageFolderDataModule._build_sampler()`.

```python
# In classifier_training/data/sampler.py (new file)
from torch.utils.data import WeightedRandomSampler

class TrackingWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler that records the last sampled indices."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        self._last_indices = indices
        return iter(indices)
```

### Pattern 6: Hydra Callbacks Config (conf/callbacks/default.yaml)

**What:** Flat YAML with `_target_` keys for each callback. Uses `${hydra:runtime.output_dir}` for paths.

```yaml
# Source: adapted from sibling's conf/callbacks/default.yaml
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: ${hydra:runtime.output_dir}/checkpoints
  filename: "epoch={epoch:02d}-val_acc={val/acc_top1:.4f}"
  monitor: val/acc_top1
  mode: max
  save_top_k: 3
  save_last: true

early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: val/acc_top1
  mode: max
  patience: 10
  min_delta: 0.001

ema:
  _target_: classifier_training.callbacks.ema.EMACallback
  decay: 0.9999
  warmup_steps: 2000

onnx_export:
  _target_: classifier_training.callbacks.onnx_export.ONNXExportCallback
  output_dir: ${hydra:runtime.output_dir}
  opset_version: 17
  input_height: 224
  input_width: 224

lr_monitor:
  _target_: lightning.pytorch.callbacks.LearningRateMonitor
  logging_interval: epoch

rich_progress:
  _target_: lightning.pytorch.callbacks.RichProgressBar

confusion_matrix:
  _target_: classifier_training.callbacks.confusion_matrix.ConfusionMatrixCallback
  output_dir: ${hydra:runtime.output_dir}/confusion_matrices
```

### Anti-Patterns to Avoid

- **Exporting ONNX from checkpoint file:** `ModelCheckpoint` saves original (non-EMA) weights. Loading the checkpoint `.ckpt` for ONNX export gives wrong weights. Always use `EMACallback.ema_state_dict` directly.
- **Using `matplotlib` in GPU hook without `.cpu()`:** `cm_tensor.numpy()` fails on CUDA tensors. Always call `.cpu()` on torchmetrics output before plotting.
- **Using `matplotlib.use()` at import time:** Call `matplotlib.use("Agg")` inside the method or use `plt.switch_backend("Agg")` to avoid side effects on the global pyplot state.
- **Importing `wandb` in callbacks:** Phase 3 callbacks should work without WandB. WandB logger integration (logging confusion matrix images) belongs in Phase 4 when WandbLogger is wired in.
- **`shuffle=True` with sampler:** The `ImageFolderDataModule._build_sampler()` already sets `shuffle=False` when a sampler is provided. Do not break this invariant when replacing with `TrackingWeightedRandomSampler`.
- **`persistent_workers=True` with `num_workers=0`:** DataModule already guards this. Preserve the guard when making any changes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Confusion matrix computation | Manual loop over predictions | `torchmetrics.classification.MulticlassConfusionMatrix` | Device-aware, handles batched updates, already installed |
| EMA weight averaging | Custom decay math | Port `EMACallback` from sibling | Sibling's `on_validation_start/end` swap pattern is the correct Lightning idiom |
| Model parameter counting | Manual `.numel()` loops | Use in `ModelInfoCallback._compute_basic_stats()` (already in sibling) | The sibling's fallback is already correct; no need for `fvcore` (not in pixi.toml) |
| ONNX graph validation | Parsing ONNX proto manually | `onnxruntime.InferenceSession` with CPUExecutionProvider | Standard validation approach; confirmed by MEMORY.md |
| Rich terminal tables | `print()` formatting | `rich.table.Table` + `rich.console.Console` | Already in pixi.toml; sibling callbacks use this pattern throughout |
| Training history plots | Custom drawing | `matplotlib.pyplot` | Already in sibling; straightforward to port |

---

## Common Pitfalls

### Pitfall 1: EMA Weights Not In Checkpoint — ONNX Exports Wrong Model

**What goes wrong:** Developer loads `best_model_path` from `ModelCheckpoint` and exports that checkpoint's `state_dict` to ONNX. The resulting ONNX model has training weights, not EMA weights. Inference performance is degraded.

**Why it happens:** Lightning's hook ordering for `on_train_epoch_end` (when `val_check_interval=1.0`):
1. Non-monitoring callbacks fire `on_train_epoch_end` — EMACallback runs here (no-op, it operates on batch end)
2. `pl_module.on_train_epoch_end` fires
3. Monitoring callbacks fire `on_train_epoch_end` — NOT where checkpoint saves with val metrics

When validation runs:
1. `on_validation_start` fires — EMACallback swaps in EMA weights
2. Validation runs with EMA weights — metrics computed
3. `on_validation_end` fires — EMACallback restores original weights
4. THEN `ModelCheckpoint.on_validation_end` fires — saves checkpoint with **restored original weights**

**How to avoid:** In `ONNXExportCallback.on_train_end`, find `EMACallback` in `trainer.callbacks` and access `.ema_state_dict` directly. Load those weights into the model before calling `torch.onnx.export`.

**Warning signs:** ONNX export test passes with non-EMA weights; production model underperforms training validation accuracy.

### Pitfall 2: ONNX Export Fails Without `onnx` Package

**What goes wrong:** `torch.onnx.export` raises `OnnxExporterError: Module onnx is not installed!` even when using the legacy exporter path.

**Why it happens:** `torch.onnx.export` imports `onnx` to serialize the graph. The `onnx` package is not installed by default in the classifier-training environment.

**How to avoid:** Add `onnxscript = ">=0.5.6,<0.6"` to `[dependencies]` in `pixi.toml`. This brings in `onnx` as a transitive dependency. Verify with `pixi run python -c "import onnx"` after `pixi install`.

**Warning signs:** `ModuleNotFoundError: No module named 'onnx'` at export time (confirmed in environment check — not currently installed).

### Pitfall 3: matplotlib Not Available — Training History Callback Fails at Import

**What goes wrong:** `TrainingHistoryCallback` (or `ConfusionMatrixCallback`) raises `ModuleNotFoundError: No module named 'matplotlib'` when imported.

**Why it happens:** `matplotlib` is not in `classifier-training`'s `pixi.toml` (confirmed — not currently installed).

**How to avoid:** Add `matplotlib = "*"` to `[dependencies]` in `pixi.toml` before implementing the callbacks that use it.

**Warning signs:** `ModuleNotFoundError` at import time for `matplotlib.pyplot` (confirmed in environment check — not currently installed).

### Pitfall 4: Confusion Matrix Callback GPU Device Mismatch

**What goes wrong:** `MulticlassConfusionMatrix` is initialized on CPU in `__init__`, model runs on GPU, `update(preds, labels)` fails with device mismatch.

**Why it happens:** `MulticlassConfusionMatrix` must be on the same device as its inputs. Initializing in `__init__` happens before the model is placed on GPU.

**How to avoid:** Initialize `MulticlassConfusionMatrix` in `on_fit_start` after the model is placed, using `pl_module.device`:
```python
def on_fit_start(self, trainer, pl_module):
    self._cm = MulticlassConfusionMatrix(num_classes=...).to(pl_module.device)
```

**Warning signs:** `RuntimeError: Expected all tensors to be on the same device` during validation.

### Pitfall 5: SamplerDistributionCallback Finds No Tracked Indices

**What goes wrong:** `SamplerDistributionCallback` attempts `sampler._last_indices` but the standard `WeightedRandomSampler` has no `_last_indices` attribute.

**Why it happens:** Unlike the sibling project (which uses `TrackingWeightedRandomSampler`), the base classifier-training project uses standard `WeightedRandomSampler`. The attribute doesn't exist.

**How to avoid:** Implement `TrackingWeightedRandomSampler` in `classifier_training/data/sampler.py` and update `ImageFolderDataModule._build_sampler()` to use it. `TrackingWeightedRandomSampler` is a one-method subclass that overrides `__iter__` to capture indices.

**Warning signs:** `AttributeError: 'WeightedRandomSampler' object has no attribute '_last_indices'` at epoch end.

### Pitfall 6: ModelCheckpoint `monitor` Key Must Match Logged Metric Name Exactly

**What goes wrong:** `ModelCheckpoint(monitor="val/acc_top1")` raises `MisconfigurationException` or saves nothing because the key isn't found.

**Why it happens:** `BaseClassificationModel.on_validation_epoch_end()` logs `self.log("val/acc_top1", ...)`. The monitor key must exactly match the logged string, including slash separators.

**How to avoid:** Verify the exact metric key in `base.py` before configuring `ModelCheckpoint`. The correct key is `"val/acc_top1"` (confirmed in `base.py` line 118: `self.log("val/acc_top1", ...)`).

**Warning signs:** "ModelCheckpoint(monitor='...') could not find the monitored key" warning in trainer output.

### Pitfall 7: `torch.onnx.export` Dynamo Mode Fails Without `onnxscript`

**What goes wrong:** Default `torch.onnx.export` in PyTorch 2.10 uses the Dynamo exporter, which requires `onnxscript`. Without it, export fails.

**Why it happens:** PyTorch 2.6+ defaults `dynamo=True`. With `onnxscript` installed, this should work, but the `dynamo` path has known issues with some model architectures and produces larger graphs.

**How to avoid:** Use the same monkeypatch pattern as the sibling — force `dynamo=False` via environment variable and keyword override:
```python
os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"
```
and monkeypatch `torch.onnx.export` to inject `dynamo=False`. ResNet is a simple architecture and should work with either path, but the legacy path is more predictable.

---

## Code Examples

Verified patterns from codebase inspection and environment testing:

### EMA State Dict Access (Verified via sibling source)
```python
# Source: object-detection-training/callbacks/ema.py
# In ONNXExportCallback.on_train_end:
from classifier_training.callbacks.ema import EMACallback

ema_cb = next(
    (cb for cb in trainer.callbacks if isinstance(cb, EMACallback)),
    None
)
if ema_cb and ema_cb.ema_state_dict:
    export_weights = ema_cb.ema_state_dict
```

### ONNX Export for Classifier (Verified: logits output, dynamic batch)
```python
# Source: adapted from sibling's rfdetr_lightning.export_onnx
import os, copy
import torch

os.environ["TORCH_ONNX_LEGACY_EXPORTER"] = "1"
original_export = torch.onnx.export
def _patched(*args, **kwargs):
    kwargs["dynamo"] = False
    return original_export(*args, **kwargs)
torch.onnx.export = _patched

model = copy.deepcopy(pl_module).cpu().eval()
dummy = torch.randn(1, 3, input_height, input_width)
try:
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],            # CALL-02: must be "logits"
            opset_version=17,
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
finally:
    torch.onnx.export = original_export
```

### ONNX Validation with onnxruntime (for tests — per MEMORY.md)
```python
# Source: MEMORY.md — always use CPUExecutionProvider for ONNX tests
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]   # NEVER CoreMLExecutionProvider in tests
)
dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = sess.run(None, {"input": dummy_np})
# outputs[0] is logits with shape (1, num_classes)
assert outputs[0].shape == (1, num_classes)
assert sess.get_outputs()[0].name == "logits"  # CALL-02 verification
```

### MulticlassConfusionMatrix Usage (Verified: torchmetrics 1.8.2 installed)
```python
# Source: torchmetrics.classification API — verified available in environment
from torchmetrics.classification import MulticlassConfusionMatrix

cm = MulticlassConfusionMatrix(num_classes=43)
# During validation:
cm.update(preds, labels)   # preds: (B,) int64, labels: (B,) int64
# At epoch end:
matrix = cm.compute()      # (num_classes, num_classes) int64 tensor
cm.reset()
# For plotting:
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(matrix.cpu().numpy(), cmap="Blues")  # .cpu() is required
```

### TrackingWeightedRandomSampler Pattern (Verified: sibling pattern)
```python
# Source: object-detection-training/data/sampler.py pattern
from torch.utils.data import WeightedRandomSampler
from typing import Iterator, Any

class TrackingWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        self._last_indices = indices
        return iter(indices)
```

### EarlyStopping for Accuracy Monitoring (Verified: Lightning 2.6.1 API)
```python
# Source: Lightning 2.6.1 EarlyStopping signature (verified)
from lightning.pytorch.callbacks import EarlyStopping

EarlyStopping(
    monitor="val/acc_top1",  # matches logged metric key exactly
    mode="max",              # stop when accuracy stops increasing
    patience=10,
    min_delta=0.001,
    verbose=True,
)
```

### ModelCheckpoint with Accuracy Monitoring (Verified: Lightning 2.6.1 API)
```python
# Source: Lightning 2.6.1 ModelCheckpoint signature (verified)
from lightning.pytorch.callbacks import ModelCheckpoint

ModelCheckpoint(
    dirpath="${hydra:runtime.output_dir}/checkpoints",  # in YAML
    filename="epoch={epoch:02d}-val_acc={val/acc_top1:.4f}",
    monitor="val/acc_top1",
    mode="max",
    save_top_k=3,
    save_last=True,
    # save_on_train_epoch_end is NOT set → defaults to on_validation_end
    # when check_val_every_n_epoch=1 and val_check_interval=1.0
)
```

### LearningRateMonitor (Verified: Lightning 2.6.1 API)
```python
from lightning.pytorch.callbacks import LearningRateMonitor
LearningRateMonitor(logging_interval="epoch")
# logging_interval=None means log at both step and epoch boundaries
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.onnx.export()` without `dynamo=` | `dynamo=True` default in PyTorch 2.6+ | PyTorch 2.0+ | Must force `dynamo=False` for stable legacy path |
| `ToTensor()` in transforms | `v2.ToImage() + v2.ToDtype(float32)` | torchvision 0.14+ | Phase 1 already uses v2; callbacks don't touch transforms |
| Manual confusion matrix | `torchmetrics.MulticlassConfusionMatrix` | torchmetrics 0.10+ | Device-aware; no sklearn needed |
| SWA for weight averaging | EMA callback (custom) | ongoing | EMA is locked decision per prior decisions section |

**Deprecated/outdated:**
- `torch.onnx.export(dynamo=True)`: While this is now the default in PyTorch 2.x, it is less predictable with custom model patterns. The legacy path (forced via monkeypatch or env var) is more reliable for production models matching the sibling pattern.
- `sklearn.metrics.confusion_matrix`: Not needed; `torchmetrics` version is GPU-native and already installed.

---

## Open Questions

1. **Does `onnx` install automatically as a transitive dependency of `onnxscript`?**
   - What we know: `onnxscript >= 0.5.6, < 0.6` is required; sibling project uses it. The `onnx` package is separately listed in the sibling's PyPI deps.
   - What's unclear: Whether adding `onnxscript` to `pixi.toml` pulls in `onnx` automatically in the pixi solver.
   - Recommendation: Add `onnxscript` first, run `pixi install`, then test `import onnx`. If it fails, add `onnx = "*"` to pypi-dependencies explicitly.

2. **Should `matplotlib.use("Agg")` be called globally or per-method?**
   - What we know: On macOS CI (headless), calling `plt.show()` or using the default interactive backend causes crashes. The sibling never calls `plt.show()`.
   - What's unclear: Whether the pixi environment already sets a non-interactive backend.
   - Recommendation: Set backend inside the plot method (`matplotlib.use("Agg")`) or use `plt.switch_backend("Agg")` before each `plt.figure()`. Do NOT call globally at module import.

3. **Should `SampleVisualizationCallback` use WandB at all in Phase 3?**
   - What we know: WandB (`wandb`) is not in the `classifier-training` `pixi.toml` yet. WandB is Phase 4.
   - What's unclear: Whether the success criteria for Phase 3 requires WandB logging of sample images.
   - Recommendation: Implement `SampleVisualizationCallback` to save images to disk only (no WandB). Add WandB log calls behind a `if isinstance(logger, WandbLogger)` guard in Phase 4.

4. **`fvcore` for FLOPs computation in `ModelInfoCallback`?**
   - What we know: The sibling's `model_info.py` references `compute_model_stats()` which may use `fvcore`. `fvcore` is not in `classifier-training`'s `pixi.toml`.
   - What's unclear: Whether the CALL-04 requirement ("FLOPs, parameters, model size") requires `fvcore` or if basic parameter counting suffices.
   - Recommendation: Use `ModelInfoCallback._compute_basic_stats()` fallback (which only uses `p.numel()` and `p.element_size()`) for Phase 3. FLOPs reporting can be added in Phase 4 if needed.

---

## Critical Hook Ordering Analysis

This is the resolved answer to the prior research decision: "EMA + ModelCheckpoint timing interaction (Lightning issue #11276)."

**Verified for Lightning 2.6.1 with `check_val_every_n_epoch=1`, `val_check_interval=1.0`:**

```
Per epoch (with validation):
1. Training batches run
   └─ on_train_batch_end → EMACallback updates ema_state_dict
2. on_train_epoch_end fires for non-monitoring callbacks (monitoring_callbacks=False)
   └─ EMACallback.on_train_epoch_end (no-op — all work is in on_train_batch_end)
   └─ ConfusionMatrixCallback, TrainingHistoryCallback, etc.
3. pl_module.on_train_epoch_end fires
4. on_train_epoch_end fires for monitoring callbacks (ModelCheckpoint, EarlyStopping)
   └─ BUT: _should_save_on_train_epoch_end() returns False when val runs every epoch
   └─ So ModelCheckpoint does NOT save here
5. Validation loop runs
   └─ on_validation_start → EMACallback swaps EMA weights into model
   └─ Validation steps run (metrics computed with EMA weights)
   └─ on_validation_epoch_end → ConfusionMatrixCallback plots
   └─ on_validation_end → EMACallback RESTORES original weights
   └─ on_validation_end → ModelCheckpoint saves checkpoint (with ORIGINAL weights)
```

**Conclusion:** `ModelCheckpoint` checkpoints always contain original (non-EMA) weights. `ONNXExportCallback.on_train_end` must use `EMACallback.ema_state_dict` directly, not the checkpoint file.

Source: Direct inspection of `/opt/.../lightning/pytorch/loops/fit_loop.py` lines 478–480 (monitoring_callbacks split), `/opt/.../callbacks/model_checkpoint.py` lines 642–664 (`_should_save_on_train_epoch_end`), and `/opt/.../loops/evaluation_loop.py` lines 354–360 (on_validation_end ordering). All verified in the installed Lightning 2.6.1 package.

---

## Sources

### Primary (HIGH confidence)
- Installed Lightning 2.6.1 source: `loops/fit_loop.py`, `callbacks/model_checkpoint.py`, `loops/evaluation_loop.py`, `trainer/call.py` — hook ordering analysis
- Installed torchmetrics 1.8.2: `MulticlassConfusionMatrix` class confirmed available, tested in environment
- Sibling project source: `object-detection-training/src/object_detection_training/callbacks/` — all 6 callback implementations read directly
- `classifier-training` source: `models/base.py`, `data/datamodule.py`, `types.py`, `pixi.toml` — confirmed installed packages and missing dependencies
- Direct environment test: confirmed `matplotlib`, `onnx`, `onnxruntime` NOT installed; `torchmetrics`, `rich`, `lightning` ARE installed

### Secondary (MEDIUM confidence)
- Sibling `pixi.toml` — `onnxscript >= 0.5.6, < 0.6`, `matplotlib = "*"`, `onnxruntime >= 1.23.2, < 2` confirmed working combination for sibling project
- MEMORY.md — CPUExecutionProvider requirement for ONNX tests (carryover from basketball-2d-to-3d experience)

### Tertiary (LOW confidence)
- Prior decisions note "Lightning issue #11276" — this GitHub issue concerns EMA + checkpoint timing; resolved via source inspection rather than issue text

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions confirmed from installed packages and pixi.lock
- Architecture patterns: HIGH — sibling source read directly; hook ordering verified in installed Lightning source
- Pitfalls: HIGH — three pitfalls (ONNX missing, matplotlib missing, hook ordering) confirmed via direct testing

**Research date:** 2026-02-18
**Valid until:** 2026-04-01 (Lightning 2.x stable; torchmetrics API stable; onnx opset 17 stable)

---

## Sibling Source Files Available for Direct Porting

All sibling callback source files confirmed readable at these absolute paths:

| File | Port Status | Key Adaptation |
|------|------------|----------------|
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/ema.py` | Port directly | Change `DetectionBatch` to `ClassificationBatch` |
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/onnx_export.py` | Port and adapt | Add EMA state dict lookup; change output_names to `["logits"]`; remove best-checkpoint loading (replaced by EMA lookup) |
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/model_info.py` | Port and adapt | Change `input_height/width` defaults to 224; simplify `_export_labels_mapping` to use `dm.save_labels_mapping()` |
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/statistics.py` | Rewrite simplified | Replace `DetectionDataset` / `DatasetStatistics` with direct `_train_dataset.samples` iteration |
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/plotting.py` | Port and adapt | Replace mAP metric keys with `val/acc_top1`, `val/acc_top5`; rename plot from `map_history.png` to `accuracy_history.png` |
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/visualization.py` | Rewrite for classification | Remove bounding box logic; add `argmax(logits)` overlay; remove `supervision` dependency |
| `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/sampler_distribution.py` | Port with sampler change | Requires `TrackingWeightedRandomSampler` in `classifier_training/data/sampler.py` |
| *(no sibling equivalent)* | NEW: `confusion_matrix.py` | Use `MulticlassConfusionMatrix` from torchmetrics; plot with matplotlib |

**Test files available for direct porting:**
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/tests/test_callbacks_ema.py`
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/tests/test_callbacks_onnx_export.py`
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/tests/test_callbacks_model_info.py`
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/tests/test_callbacks_plotting.py`
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/tests/test_callbacks_statistics.py`
