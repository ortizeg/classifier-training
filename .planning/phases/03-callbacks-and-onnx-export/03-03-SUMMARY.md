---
phase: 03-callbacks-and-onnx-export
plan: 03
subsystem: training
tags: [hydra, callbacks, lightning, integration-test, onnxruntime, yaml]

# Dependency graph
requires:
  - phase: 03-callbacks-and-onnx-export
    plan: 01
    provides: EMACallback, ONNXExportCallback
  - phase: 03-callbacks-and-onnx-export
    plan: 02
    provides: ConfusionMatrixCallback, DatasetStatisticsCallback, ModelInfoCallback, TrainingHistoryCallback, SamplerDistributionCallback, SampleVisualizationCallback
provides:
  - Hydra conf/callbacks/default.yaml with all 12 callbacks configured via _target_ keys
  - Integration tests proving callback stack coexists and produces artifacts (checkpoint, ONNX, confusion matrix)
affects: [04-training-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [Hydra callback config group with flat keys, dict-batch integration testing with ClassificationBatch]

key-files:
  created:
    - src/classifier_training/conf/callbacks/default.yaml
    - tests/test_callbacks_integration.py
  modified: []

key-decisions:
  - "LightningDataModule subclass for mock datamodule in tests -- trainer.datamodule requires proper LDM interface"
  - "Dict-batch (_DictDataset) for integration tests -- ConfusionMatrixCallback expects ClassificationBatch dict format"

patterns-established:
  - "Callback config group: callbacks=default in Hydra config composes all 12 callbacks"
  - "Integration test pattern: _MinimalModule + _DictDataset + manual callback instantiation for smoke loop"

# Metrics
duration: 3min
completed: 2026-02-18
---

# Phase 3 Plan 3: Hydra Callback Config + Integration Tests Summary

**Hydra default.yaml wiring all 12 callbacks (EMA, ONNX, ConfusionMatrix, ModelInfo, DatasetStatistics, TrainingHistory, SamplerDistribution, SampleVisualization, ModelCheckpoint, EarlyStopping, LRMonitor, RichProgressBar) with integration tests proving the full stack works together**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T03:04:31Z
- **Completed:** 2026-02-19T03:07:36Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Hydra `conf/callbacks/default.yaml` configures all 12 callbacks with sensible defaults and _target_ resolution
- Integration test verifies all 12 _target_ paths are importable (catches typos)
- Smoke 2-epoch training loop with 6 callbacks produces checkpoint, model.onnx, confusion_matrix.png, labels_mapping.json
- EMA + ONNX end-to-end test validates ONNX output shape under onnxruntime CPUExecutionProvider
- 90 tests passing (3 new integration tests), lint and typecheck clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Hydra default.yaml for all callbacks** - `35f2847` (feat)
2. **Task 2: Integration tests for callback stack** - `0f0cdf8` (feat)

## Files Created/Modified
- `src/classifier_training/conf/callbacks/default.yaml` - Hydra config group with all 12 callbacks, flat keys, _target_ references
- `tests/test_callbacks_integration.py` - 3 integration tests: target importability, smoke training loop, EMA+ONNX end-to-end

## Decisions Made
- LightningDataModule subclass for mock datamodule -- trainer.datamodule requires proper LDM interface (not a plain class)
- Dict-batch dataset (_DictDataset) for integration tests -- ConfusionMatrixCallback accesses batch["images"] and batch["labels"]

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock datamodule must extend LightningDataModule**
- **Found during:** Task 2 (smoke training loop)
- **Issue:** Plain class assigned to trainer.datamodule caused ValueError ("Expected a parent") -- Lightning requires LDM interface
- **Fix:** Changed _MockDataModule to extend L.LightningDataModule
- **Files modified:** tests/test_callbacks_integration.py
- **Verification:** Test passes, trainer.fit completes
- **Committed in:** 0f0cdf8

**2. [Rule 1 - Bug] Dict-batch format required for ConfusionMatrixCallback**
- **Found during:** Task 2 (smoke training loop)
- **Issue:** TensorDataset returns tuples, but ConfusionMatrixCallback.on_validation_batch_end indexes batch["images"]/batch["labels"]
- **Fix:** Created _DictDataset returning ClassificationBatch-style dicts, updated _MinimalModule to accept dict batches
- **Files modified:** tests/test_callbacks_integration.py
- **Verification:** All 3 integration tests pass
- **Committed in:** 0f0cdf8

**3. [Rule 1 - Bug] Lint fixes: unused pytest import, line length**
- **Found during:** Task 2 (lint check)
- **Issue:** ruff flagged F401 (unused pytest import) and E501 (line too long)
- **Fix:** Removed unused import, wrapped long lines
- **Files modified:** tests/test_callbacks_integration.py
- **Verification:** pixi run lint passes
- **Committed in:** 0f0cdf8

---

**Total deviations:** 3 auto-fixed (3 bug fixes)
**Impact on plan:** All fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 complete: all 8 custom callbacks + 4 Lightning built-in callbacks implemented, tested, and configured
- `callbacks=default` Hydra config group ready for Phase 4 training pipeline integration
- 90 tests passing across full suite

---
*Phase: 03-callbacks-and-onnx-export*
*Completed: 2026-02-18*
