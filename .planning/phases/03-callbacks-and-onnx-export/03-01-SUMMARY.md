---
phase: 03-callbacks-and-onnx-export
plan: 01
subsystem: training
tags: [ema, onnx, onnxruntime, callbacks, lightning, weighted-sampler]

# Dependency graph
requires:
  - phase: 02-model-layer
    provides: BaseClassificationModel with forward(), state_dict(), ClassificationBatch TypedDict
  - phase: 01-foundation-and-data-pipeline
    provides: ImageFolderDataModule with save_labels_mapping(), WeightedRandomSampler
provides:
  - EMACallback with configurable decay, warmup, validation/test weight swapping
  - ONNXExportCallback with EMA integration, dynamic batch axis, labels_mapping sidecar
  - TrackingWeightedRandomSampler with _last_indices for distribution monitoring
  - Phase 3 dependencies (matplotlib, onnxscript, onnxruntime, pandas)
affects: [03-02, 03-03, 04-training-pipeline, 05-cloud-deployment]

# Tech tracking
tech-stack:
  added: [matplotlib, onnxscript, onnxruntime, pandas]
  patterns: [EMA weight swap for validation/test, legacy ONNX exporter with dynamo=False, tracking sampler subclass]

key-files:
  created:
    - src/classifier_training/callbacks/__init__.py
    - src/classifier_training/callbacks/ema.py
    - src/classifier_training/callbacks/onnx_export.py
    - src/classifier_training/data/sampler.py
    - tests/test_callbacks_ema.py
    - tests/test_callbacks_onnx_export.py
  modified:
    - pixi.toml
    - src/classifier_training/data/__init__.py
    - src/classifier_training/data/datamodule.py

key-decisions:
  - "Warmup formula: min(decay, (1+step)/(10+step)) -- smooth ramp matching sibling repo"
  - "Legacy ONNX exporter via TORCH_ONNX_LEGACY_EXPORTER=1 + dynamo=False monkeypatch -- avoids dynamo issues with PyTorch 2.9"
  - "getattr(trainer, 'datamodule') for mypy compatibility -- trainer.datamodule not typed in Lightning stubs"
  - "Real LightningModule in ONNX tests (not MagicMock) -- deepcopy compatibility with torch.onnx.export"

patterns-established:
  - "EMA swap pattern: on_validation_start loads EMA weights, on_validation_end restores originals"
  - "ONNX export: deepcopy to CPU, legacy exporter, output_names=['logits'], dynamic_axes on batch dim"
  - "TrackingWeightedRandomSampler: subclass captures _last_indices on each iteration"

# Metrics
duration: 6min
completed: 2026-02-18
---

# Phase 3 Plan 1: EMA + ONNX Export Callbacks Summary

**EMACallback with warmup decay ramp and validation weight swapping, ONNXExportCallback with EMA integration and dynamic batch axis, TrackingWeightedRandomSampler for distribution monitoring**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-19T02:49:54Z
- **Completed:** 2026-02-19T02:56:07Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- EMACallback maintains exponential moving average of model weights with warmup ramp, swaps for validation/test
- ONNXExportCallback exports ONNX model with output name "logits", dynamic batch axis, EMA weight integration
- TrackingWeightedRandomSampler replaces WeightedRandomSampler in datamodule with _last_indices tracking
- 4 new dependencies installed (matplotlib, onnxscript, onnxruntime, pandas) for Phase 3 callbacks
- 17 new tests (10 EMA + 7 ONNX), full suite 72 tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Phase 3 dependencies and create TrackingWeightedRandomSampler** - `9d466f4` (feat)
2. **Task 2: Implement EMACallback and ONNXExportCallback with tests** - `2a5c810` (feat)

## Files Created/Modified
- `src/classifier_training/callbacks/__init__.py` - Package init, exports EMACallback and ONNXExportCallback
- `src/classifier_training/callbacks/ema.py` - EMA callback with decay, warmup, validation/test swap
- `src/classifier_training/callbacks/onnx_export.py` - ONNX export with EMA lookup, legacy exporter, labels_mapping sidecar
- `src/classifier_training/data/sampler.py` - TrackingWeightedRandomSampler with _last_indices
- `src/classifier_training/data/__init__.py` - Added TrackingWeightedRandomSampler export
- `src/classifier_training/data/datamodule.py` - Swapped WeightedRandomSampler for TrackingWeightedRandomSampler
- `pixi.toml` - Added matplotlib, onnxscript, onnxruntime, pandas dependencies
- `tests/test_callbacks_ema.py` - 10 tests for EMA callback
- `tests/test_callbacks_onnx_export.py` - 7 tests for ONNX export callback

## Decisions Made
- Warmup formula `min(decay, (1+step)/(10+step))` matches sibling repo exactly
- Legacy ONNX exporter forced via env var + monkeypatch to avoid PyTorch 2.9 dynamo issues
- Used `getattr(trainer, 'datamodule')` instead of direct attribute access for mypy compatibility
- Used real LightningModule in ONNX tests instead of MagicMock for deepcopy compatibility
- `Sequence[float]` type for sampler weights parameter to satisfy mypy strict mode

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed onnxscript version constraint**
- **Found during:** Task 1 (pixi install)
- **Issue:** Plan specified `>=0.5.6,<0.6` but I initially tried `>=0.1.0,<0.2` which had no candidates
- **Fix:** Corrected to `>=0.5.6,<0.6` matching the plan specification
- **Files modified:** pixi.toml
- **Verification:** pixi install succeeded
- **Committed in:** 9d466f4

**2. [Rule 1 - Bug] Fixed mypy Sequence[float] type for sampler weights**
- **Found during:** Task 1 (typecheck)
- **Issue:** `list[float] | torch.Tensor` not compatible with WeightedRandomSampler's `Sequence[float]` parameter
- **Fix:** Changed to `Sequence[float]` import from collections.abc
- **Files modified:** src/classifier_training/data/sampler.py
- **Verification:** pixi run typecheck passes
- **Committed in:** 9d466f4

**3. [Rule 1 - Bug] Fixed mypy trainer attribute access**
- **Found during:** Task 2 (typecheck)
- **Issue:** `trainer.callbacks` and `trainer.datamodule` not typed in Lightning stubs
- **Fix:** Used `type: ignore[attr-defined]` for callbacks, `getattr()` for datamodule
- **Files modified:** src/classifier_training/callbacks/onnx_export.py
- **Verification:** pixi run typecheck passes with 0 errors
- **Committed in:** 2a5c810

---

**Total deviations:** 3 auto-fixed (3 bug fixes)
**Impact on plan:** All fixes necessary for correctness and type safety. No scope creep.

## Issues Encountered
- MagicMock `__deepcopy__` protocol incompatible with `copy.deepcopy` -- resolved by using real LightningModule in ONNX tests

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- EMA and ONNX export callbacks ready for integration in training pipeline
- TrackingWeightedRandomSampler ready for SamplerDistributionCallback (plan 03-02)
- All Phase 3 dependencies installed for remaining callback plans

---
*Phase: 03-callbacks-and-onnx-export*
*Completed: 2026-02-18*
