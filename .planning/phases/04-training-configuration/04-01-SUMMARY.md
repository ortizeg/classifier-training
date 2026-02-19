---
phase: 04-training-configuration
plan: 01
subsystem: training
tags: [hydra, wandb, lightning, training-pipeline, config-composition]

requires:
  - phase: 01-foundation-and-data-pipeline
    provides: ImageFolderDataModule with DataModuleConfig, class_to_idx, class weights
  - phase: 02-model-layer
    provides: ResNet18/50 models with @register decorator, BaseClassificationModel
  - phase: 03-callbacks-and-onnx-export
    provides: 12 callbacks configured in callbacks/default.yaml
provides:
  - train.py @hydra.main entrypoint with model/data/callback/logger instantiation
  - Root Hydra config composing 5 config groups (model, data, trainer, callbacks, logging)
  - Hydra-compatible ImageFolderDataModule accepting flat kwargs from instantiate
  - Fixed ModelCheckpoint dirpath for stable resume via ckpt_path="last"
  - wandb integration via WandbLogger config group
  - pixi run train task
affects: [04-02, 05-cloud-deployment]

tech-stack:
  added: [wandb]
  patterns: [hydra-config-composition, dual-init-pattern, fixed-checkpoint-dirpath]

key-files:
  created:
    - src/classifier_training/train.py
    - src/classifier_training/conf/train_basketball_resnet18.yaml
    - src/classifier_training/conf/trainer/default.yaml
    - src/classifier_training/conf/data/basketball_jersey_numbers.yaml
    - src/classifier_training/conf/logging/wandb.yaml
  modified:
    - src/classifier_training/data/datamodule.py
    - src/classifier_training/conf/callbacks/default.yaml
    - src/classifier_training/models/resnet.py
    - src/classifier_training/conf/model/resnet18.yaml
    - src/classifier_training/conf/model/resnet50.yaml
    - pixi.toml
    - tests/test_model.py

key-decisions:
  - "Dual-init DataModule: config=DataModuleConfig OR flat kwargs from Hydra, with **kwargs absorbing _target_"
  - "type: ignore[operator] for model.set_class_weights() -- mypy sees LightningModule, not BaseClassificationModel"
  - "ModelCheckpoint dirpath='checkpoints' (relative) + default_root_dir=HydraConfig.cwd for stable resume"
  - "Trainer built from flat dict, not hydra.utils.instantiate (no _target_ key)"

patterns-established:
  - "Dual-init pattern: accept config object OR flat kwargs for Hydra compatibility"
  - "Config group singular naming: conf/model/ not conf/models/ to match Hydra defaults list"
  - "Fixed checkpoint dir: dirpath='checkpoints' + default_root_dir=cwd for resume stability"

duration: 4min
completed: 2026-02-18
---

# Phase 4 Plan 1: Hydra Training Pipeline Wiring Summary

**Hydra config composition across 5 groups (model/data/trainer/callbacks/logging) with train.py entrypoint, wandb logging, and stable checkpoint resume via fixed dirpath**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T03:51:29Z
- **Completed:** 2026-02-19T03:55:50Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Created train.py @hydra.main entrypoint with full model/data/callback/logger instantiation pipeline
- Wired 5 Hydra config groups (model, data, trainer, callbacks, logging) composing from root config
- Adapted ImageFolderDataModule for dual instantiation: existing config= pattern and Hydra flat kwargs
- Fixed ModelCheckpoint dirpath to "checkpoints" (relative) for resume stability with ckpt_path="last"
- Added wandb>=0.19 and pixi run train task

## Task Commits

Each task was committed atomically:

1. **Task 1: Hydra config YAMLs + pixi.toml dependencies** - `220de49` (feat)
2. **Task 2: DataModule Hydra adapter + train.py entrypoint** - `c0cfeba` (feat)

## Files Created/Modified
- `src/classifier_training/train.py` - @hydra.main entrypoint: instantiates model, data, callbacks, loggers, trainer
- `src/classifier_training/conf/train_basketball_resnet18.yaml` - Root config with defaults list for all 5 groups
- `src/classifier_training/conf/trainer/default.yaml` - T4-tuned trainer: 16-mixed precision, clip_val=1.0
- `src/classifier_training/conf/data/basketball_jersey_numbers.yaml` - Dataset config: batch_size=64, num_workers=4
- `src/classifier_training/conf/logging/wandb.yaml` - WandbLogger config
- `src/classifier_training/conf/callbacks/default.yaml` - Fixed ModelCheckpoint dirpath to "checkpoints"
- `src/classifier_training/conf/model/` - Renamed from conf/models/ for Hydra group alignment
- `src/classifier_training/models/resnet.py` - Added group="model" to @register decorators
- `src/classifier_training/data/datamodule.py` - Dual-init: config=DataModuleConfig or flat kwargs
- `pixi.toml` - Added wandb dep and train task
- `tests/test_model.py` - Updated ConfigStore group from "models" to "model"

## Decisions Made
- **Dual-init DataModule pattern:** Accept both `config=DataModuleConfig(...)` (existing tests) and flat kwargs from `hydra.utils.instantiate` (Hydra). `**kwargs: Any` absorbs Hydra-injected keys like `_target_`. All 90 existing tests pass unchanged.
- **type: ignore[operator] for set_class_weights:** mypy sees `model` as `L.LightningModule` (no `set_class_weights`). The `attr-defined` ignore causes a cascade where mypy resolves the method as returning Tensor and flags the call as `operator`. Single `operator` ignore is sufficient.
- **Trainer from flat dict:** Trainer config has no `_target_` key -- built directly via `L.Trainer(**trainer_cfg)` with callbacks and loggers injected separately.
- **default_root_dir = HydraConfig.get().runtime.cwd:** Combined with `dirpath="checkpoints"`, this resolves to `{original_cwd}/checkpoints` -- stable across runs for ckpt_path="last" resume.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed config.xxx references after dual-init refactor**
- **Found during:** Task 2 (DataModule Hydra adapter)
- **Issue:** After making `config` parameter optional, lines 78-91 still referenced `config.num_workers`, `config.pin_memory`, etc. which would fail with `AttributeError: 'NoneType'`
- **Fix:** Changed all `config.xxx` references to `self._config.xxx` in the MPS guard and DataLoader parameter setup
- **Files modified:** src/classifier_training/data/datamodule.py
- **Verification:** All 90 tests pass
- **Committed in:** c0cfeba (Task 2 commit)

**2. [Rule 1 - Bug] Updated Hydra registration tests for singular group name**
- **Found during:** Task 2 (verification)
- **Issue:** Tests asserted ConfigStore group "models" (plural) but @register now uses group="model" (singular)
- **Fix:** Changed `cs.repo.get("models", {})` to `cs.repo.get("model", {})` in both test methods
- **Files modified:** tests/test_model.py
- **Verification:** 90/90 tests pass
- **Committed in:** c0cfeba (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness after the planned conf/models/ -> conf/model/ rename. No scope creep.

## Issues Encountered
- Line length violation on `@hydra.main(...)` decorator -- split across two lines to satisfy ruff E501 (88 char limit)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- train.py entrypoint ready: `pixi run train` will launch training with all config groups
- wandb installed but requires `wandb login` before first run (handled in Phase 4 Plan 2 or at runtime)
- All 90 tests passing, lint and typecheck clean
- Ready for 04-02 (hyperparameter tuning / smoke run)

---
*Phase: 04-training-configuration*
*Completed: 2026-02-18*
