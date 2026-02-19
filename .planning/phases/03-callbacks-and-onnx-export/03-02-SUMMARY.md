---
phase: 03-callbacks-and-onnx-export
plan: 02
subsystem: training
tags: [callbacks, lightning, confusion-matrix, torchmetrics, matplotlib, rich, observability]

# Dependency graph
requires:
  - phase: 03-callbacks-and-onnx-export
    plan: 01
    provides: EMACallback, ONNXExportCallback, TrackingWeightedRandomSampler, matplotlib dependency
  - phase: 01-foundation-and-data-pipeline
    provides: ImageFolderDataModule with save_labels_mapping(), class_to_idx, _train_dataset.samples
provides:
  - ConfusionMatrixCallback with device-aware MulticlassConfusionMatrix and per-epoch PNG heatmaps
  - DatasetStatisticsCallback with rich table of class distribution at training start
  - ModelInfoCallback with parameter counts, model size, and labels_mapping.json sidecar
  - TrainingHistoryCallback with loss and accuracy curve PNGs
  - SamplerDistributionCallback reading TrackingWeightedRandomSampler._last_indices
  - SampleVisualizationCallback with predicted-vs-true grid PNG (color-coded correct/incorrect)
affects: [03-03, 04-training-pipeline]

# Tech tracking
tech-stack:
  added: [torchmetrics.classification.MulticlassConfusionMatrix]
  patterns: [device-aware metric init in on_fit_start, matplotlib Agg backend in plot methods, rich table for CLI output]

key-files:
  created:
    - src/classifier_training/callbacks/confusion_matrix.py
    - src/classifier_training/callbacks/statistics.py
    - src/classifier_training/callbacks/model_info.py
    - src/classifier_training/callbacks/plotting.py
    - src/classifier_training/callbacks/sampler.py
    - src/classifier_training/callbacks/visualization.py
    - tests/test_callbacks_observability.py
  modified:
    - src/classifier_training/callbacks/__init__.py

key-decisions:
  - "MulticlassConfusionMatrix initialized in on_fit_start (not __init__) for correct device placement"
  - "matplotlib.use('Agg') called inside plot methods, not at module level"
  - "SamplerDistributionCallback fires on_train_epoch_start (epoch 1+) since _last_indices empty at epoch 0"
  - "SampleVisualization denormalizes with hardcoded ImageNet constants to avoid circular import from datamodule"

patterns-established:
  - "Callback device pattern: initialize torchmetrics in on_fit_start, call .to(pl_module.device)"
  - "Plot save pattern: matplotlib Agg backend + plt.close(fig) to prevent memory leaks"
  - "Graceful degradation: all callbacks handle missing datamodule/dataset with warning and early return"

# Metrics
duration: 4min
completed: 2026-02-18
---

# Phase 3 Plan 2: Observability Callbacks Summary

**Six observability callbacks with per-epoch confusion matrix heatmaps, class distribution tables, model stats, loss/accuracy curves, sampler distribution tracking, and sample prediction grids -- all saving to disk without WandB**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T02:58:24Z
- **Completed:** 2026-02-19T03:02:30Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- ConfusionMatrixCallback produces per-epoch PNG heatmaps with device-aware MulticlassConfusionMatrix
- DatasetStatisticsCallback prints class distribution rich table at training start
- ModelInfoCallback reports parameter counts, model size in MB, writes labels_mapping.json
- TrainingHistoryCallback saves loss and accuracy curve PNGs (overwrites each epoch)
- SamplerDistributionCallback reads TrackingWeightedRandomSampler._last_indices and logs per-class counts
- SampleVisualizationCallback saves predicted-vs-true grid PNG with green/red color-coded borders
- 15 new tests, 87 total passing across full suite

## Task Commits

Each task was committed atomically:

1. **Task 1: ConfusionMatrix, DatasetStatistics, ModelInfo callbacks** - `0a229a9` (feat)
2. **Task 2: TrainingHistory, SamplerDistribution, SampleVisualization + tests** - `dbd76b1` (feat)

## Files Created/Modified
- `src/classifier_training/callbacks/confusion_matrix.py` - Per-epoch confusion matrix heatmap with device-aware torchmetrics
- `src/classifier_training/callbacks/statistics.py` - Rich table of class distribution from training dataset
- `src/classifier_training/callbacks/model_info.py` - Parameter count, model size, labels_mapping.json sidecar
- `src/classifier_training/callbacks/plotting.py` - Loss and accuracy history curve PNGs
- `src/classifier_training/callbacks/sampler.py` - Reads TrackingWeightedRandomSampler._last_indices per epoch
- `src/classifier_training/callbacks/visualization.py` - Predicted-vs-true image grid with color-coded borders
- `src/classifier_training/callbacks/__init__.py` - Updated exports: all 8 callbacks
- `tests/test_callbacks_observability.py` - 15 tests covering all six observability callbacks

## Decisions Made
- MulticlassConfusionMatrix initialized in on_fit_start for correct device placement
- matplotlib.use("Agg") called inside plot methods (not module level) to avoid backend conflicts
- SamplerDistributionCallback fires on_train_epoch_start; _last_indices empty at epoch 0, populated from epoch 1
- ImageNet normalization constants duplicated in visualization.py to avoid circular import from datamodule
- Confusion matrix uses Blues colormap with optional class name labels on axes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed lint issues: import sorting, nested if, line length, unused imports**
- **Found during:** Task 2 (lint check)
- **Issue:** ruff flagged RUF022 (__all__ sorting), SIM102 (nested if), E501 (line length), I001 (import sorting), F401 (unused imports)
- **Fix:** Applied ruff --fix for auto-fixable issues, manually simplified nested if and shortened docstring
- **Files modified:** __init__.py, plotting.py, sampler.py, visualization.py, test_callbacks_observability.py
- **Verification:** pixi run lint passes with 0 errors
- **Committed in:** dbd76b1

---

**Total deviations:** 1 auto-fixed (lint corrections)
**Impact on plan:** Standard lint cleanup. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 8 callbacks (2 from plan 01 + 6 from plan 02) exported from classifier_training.callbacks
- Ready for plan 03-03 (Hydra callback configuration)
- All callbacks save to disk only; WandB integration deferred to Phase 4

---
*Phase: 03-callbacks-and-onnx-export*
*Completed: 2026-02-18*
