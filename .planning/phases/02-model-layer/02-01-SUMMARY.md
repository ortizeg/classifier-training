---
phase: 02-model-layer
plan: "01"
subsystem: models
tags: [lightning, torchmetrics, hydra, adamw, cosine-annealing, cross-entropy]

# Dependency graph
requires:
  - phase: 01-foundation-and-data-pipeline
    provides: ClassificationBatch TypedDict, pixi environment with pytorch/lightning
provides:
  - BaseClassificationModel LightningModule ABC with CrossEntropyLoss, Pattern A metrics, AdamW + SequentialLR
  - "@register decorator for Hydra ConfigStore registration"
  - hydra-core and torchmetrics dependencies in pixi environment
affects: [02-02, 03-training-pipeline, 04-export-pipeline]

# Tech tracking
tech-stack:
  added: [hydra-core, omegaconf, torchmetrics]
  patterns: [Pattern A torchmetrics logging, register_buffer for class weights, SequentialLR warmup+cosine]

key-files:
  created:
    - src/classifier_training/utils/hydra.py
    - src/classifier_training/models/__init__.py
    - src/classifier_training/models/base.py
  modified:
    - pixi.toml
    - pyproject.toml
    - src/classifier_training/utils/__init__.py

key-decisions:
  - "Explicit class_weights: torch.Tensor annotation on class to satisfy mypy strict mode with register_buffer"
  - "int() cast for warmup_epochs hparams access instead of type: ignore to avoid unused-ignore errors across mypy versions"
  - "Copied only @register decorator from sibling repo (not instantiate_* helpers) -- those will be added when needed"

patterns-established:
  - "Pattern A torchmetrics: update() in step, compute()+log()+reset() in on_*_epoch_end"
  - "register_buffer for tensors that must move with model device (class_weights)"
  - "save_hyperparameters() with dict-style access self.hparams['key']"

# Metrics
duration: 3min
completed: 2026-02-18
---

# Phase 2 Plan 01: Base Classification Model Summary

**BaseClassificationModel LightningModule with CrossEntropyLoss class-weight buffer, Pattern A torchmetrics (Top-1/Top-5/per-class), and AdamW + SequentialLR (LinearLR warmup + CosineAnnealingLR)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T01:43:51Z
- **Completed:** 2026-02-19T01:47:11Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added hydra-core, omegaconf, and torchmetrics to pixi.toml and pyproject.toml with mypy overrides
- Created @register decorator in utils/hydra.py for Hydra ConfigStore auto-registration
- Implemented BaseClassificationModel with CrossEntropyLoss (class_weights buffer + label_smoothing), Pattern A torchmetrics (train/val/test), and AdamW + SequentialLR optimizer
- All 34 existing tests still pass, lint and typecheck clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Add hydra-core deps and @register decorator** - `cdd0f2d` (chore)
2. **Task 2: Implement BaseClassificationModel** - `a85ab39` (feat)

## Files Created/Modified
- `pixi.toml` - Added hydra-core and torchmetrics dependencies
- `pyproject.toml` - Added hydra-core, omegaconf, torchmetrics deps + mypy overrides
- `src/classifier_training/utils/__init__.py` - Re-exports register decorator
- `src/classifier_training/utils/hydra.py` - @register decorator for Hydra ConfigStore
- `src/classifier_training/models/__init__.py` - Package stub exporting BaseClassificationModel
- `src/classifier_training/models/base.py` - BaseClassificationModel LightningModule ABC

## Decisions Made
- Explicit `class_weights: torch.Tensor` class-level annotation to satisfy mypy strict with `register_buffer` (Lightning types buffer as `Tensor | Module`)
- Used `int()` cast for `warmup_epochs` hparams access instead of `# type: ignore[assignment]` to avoid version-dependent unused-ignore errors
- Copied only the `@register` decorator from sibling repo, not the `instantiate_*` helper functions -- those are not yet needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mypy strict type errors in BaseClassificationModel**
- **Found during:** Task 2
- **Issue:** Multiple mypy errors: register_buffer types buffer as `Tensor | Module` (not `Tensor`), `type: ignore` comments from plan were unused with this mypy version, `dict` return type needed type params
- **Fix:** Added class-level `class_weights: torch.Tensor` annotation, removed unnecessary type: ignore comments, used `int()` cast for warmup, added `dict[str, Any]` return type, added explicit `loss: torch.Tensor` annotation
- **Files modified:** src/classifier_training/models/base.py
- **Committed in:** a85ab39

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential for typecheck pass. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BaseClassificationModel ready for subclassing in Plan 02-02 (ResNet18Model, ResNet50Model)
- @register decorator ready for Hydra ConfigStore integration
- All 34 existing tests still pass

---
*Phase: 02-model-layer*
*Completed: 2026-02-18*
