---
phase: 01-foundation-and-data-pipeline
plan: "02"
subsystem: infra

tags: [pydantic, pytorch, pytest, types, config, fixtures]

# Dependency graph
requires:
  - phase: 01-foundation-and-data-pipeline
    plan: "01"
    provides: "pixi env, src-layout package, ruff/mypy/pytest tooling"
provides:
  - "ClassificationBatch TypedDict (images: Tensor, labels: Tensor) — DataLoader batch contract"
  - "DataModuleConfig frozen pydantic model with persistent_workers auto-correction validator"
  - "utils/__init__.py empty utils package"
  - "tests/conftest.py tmp_dataset_dir fixture (3-split/3-class JSONL-annotated synthetic dataset)"
  - "tests/test_types_config.py 7 unit tests for DataModuleConfig and ClassificationBatch"
affects:
  - "01-03: ImageFolderDataModule imports DataModuleConfig and ClassificationBatch; uses tmp_dataset_dir fixture"
  - "Phase 2: Model layer unpacks batch['images'], batch['labels'] per ClassificationBatch contract"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pydantic frozen=True model with object.__setattr__ in model_validator to auto-correct fields"
    - "TypedDict for inter-module batch contracts (not NamedTuple/dataclass — Lightning compat)"
    - "JSONL-annotated synthetic dataset fixture (mirrors real dataset structure, avoids class subdirs)"

key-files:
  created:
    - "src/classifier_training/types.py"
    - "src/classifier_training/config.py"
    - "src/classifier_training/utils/__init__.py"
    - "tests/conftest.py"
    - "tests/test_types_config.py"
  modified: []

key-decisions:
  - "TypedDict over NamedTuple/dataclass for ClassificationBatch — Lightning training_step receives dict batches"
  - "persistent_workers auto-correction in model_validator prevents silent no-op when num_workers=0"
  - "MPS num_workers guard deferred to DataModule (runtime torch check) — not in config (no runtime access)"
  - "tmp_dataset_dir uses flat image files + JSONL (not ImageFolder subdirs) — mirrors real dataset layout"
  - "Replaced /tmp/data strings with /data/test to satisfy ruff S108; used module-level _DUMMY_ROOT constant"

patterns-established:
  - "Frozen pydantic models use object.__setattr__ for validator self-correction on frozen instances"
  - "Test path strings avoid /tmp prefix (ruff S108); use /data/... for config-only string tests"
  - "Shared fixtures in tests/conftest.py — available to all test modules without import"

# Metrics
duration: 4min
completed: 2026-02-18
---

# Phase 1 Plan 02: Type Contracts and Config Summary

**DataModuleConfig frozen pydantic model with persistent_workers auto-correction, ClassificationBatch TypedDict, and JSONL-annotated synthetic dataset conftest fixture — 7 unit tests, all gates green**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-02-18T22:02:41Z
- **Completed:** 2026-02-18T22:07:00Z
- **Tasks:** 2
- **Files created:** 5

## Accomplishments

- `DataModuleConfig` frozen pydantic model: 6 fields, `model_validator` auto-corrects `persistent_workers=False` when `num_workers=0` to prevent silent PyTorch no-op
- `ClassificationBatch` TypedDict defines the `images`/`labels` batch contract used by DataModule (Plan 03) and Model (Phase 2)
- `tmp_dataset_dir` pytest fixture builds a 3-split/3-class JSONL-annotated synthetic dataset (18 JPEG images) in `tmp_path` — foundation for all Phase 1 integration tests
- All three tooling gates (`lint`, `typecheck`, `test`) pass clean; 8 total tests (7 new + 1 smoke)

## Task Commits

Each task was committed atomically:

1. **Task 1: types.py, config.py, and utils package** - `1cd8d45` (feat)
2. **Task 2: conftest.py fixture and test_types_config.py** - `7f6e8ff` (feat)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `src/classifier_training/types.py` - ClassificationBatch TypedDict with images/labels Tensor fields
- `src/classifier_training/config.py` - DataModuleConfig frozen pydantic model; persistent_workers validator
- `src/classifier_training/utils/__init__.py` - Empty utils package placeholder
- `tests/conftest.py` - tmp_dataset_dir fixture: 3 splits x 3 classes x 2 images, JSONL annotations
- `tests/test_types_config.py` - 7 tests: defaults, frozen enforcement, persistent_workers (3 cases), custom values, TypedDict keys/shapes

## Decisions Made

- TypedDict chosen over NamedTuple/dataclass for `ClassificationBatch`: Lightning's `training_step(self, batch, batch_idx)` works naturally with dict batches; Model layer unpacks as `batch["images"], batch["labels"]`
- `persistent_workers` auto-correction uses `object.__setattr__` (required because model is frozen — normal attribute assignment raises `ValidationError` even inside validators)
- MPS `num_workers` guard deferred to DataModule: requires runtime `torch.backends.mps.is_available()` check — config is a pure data model with no runtime access
- `tmp_dataset_dir` uses flat files + JSONL (not `ImageFolder`-style class subdirectories) — mirrors the real basketball-jersey-numbers-ocr dataset structure
- Test path strings use `/data/test` instead of `/tmp/data` to avoid ruff S108 (insecure temp path) false positives

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Shortened docstring in config.py to satisfy E501 line length**
- **Found during:** Task 1 (lint verification)
- **Issue:** Docstring `"""persistent_workers=True with num_workers=0 silently does nothing in PyTorch."""` was 90 chars, exceeding ruff's 88-char limit
- **Fix:** Shortened to `"""persistent_workers=True with num_workers=0 silently does nothing."""`
- **Files modified:** `src/classifier_training/config.py`
- **Verification:** `pixi run lint` exits 0
- **Committed in:** `1cd8d45` (Task 1 commit)

**2. [Rule 1 - Bug] Replaced /tmp/data with /data/test and wrapped long lines in test file**
- **Found during:** Task 2 (lint verification)
- **Issue:** ruff S108 flagged `/tmp/data` as "Probable insecure usage of temporary file or directory" on 6 occurrences; 3 lines exceeded E501 88-char limit
- **Fix:** Extracted `_DUMMY_ROOT = "/data/test"` module constant; wrapped multi-keyword `DataModuleConfig(...)` constructor calls across lines
- **Files modified:** `tests/test_types_config.py`
- **Verification:** `pixi run lint` exits 0; all 7 tests still pass
- **Committed in:** `7f6e8ff` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — line length / lint compliance)
**Impact on plan:** Pure lint fixes, no behavioral changes. Tests validate identical semantics.

## Issues Encountered

None beyond the lint deviations above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `DataModuleConfig` and `ClassificationBatch` ready for Plan 03 (ImageFolderDataModule)
- `tmp_dataset_dir` fixture available to all test modules — Plan 03 integration tests can use it directly
- All lint/typecheck/test gates green — Plan 03 can write src code and tests without clearing any backlog

---
*Phase: 01-foundation-and-data-pipeline*
*Completed: 2026-02-18*

## Self-Check: PASSED

- `src/classifier_training/types.py` — FOUND
- `src/classifier_training/config.py` — FOUND
- `src/classifier_training/utils/__init__.py` — FOUND
- `tests/conftest.py` — FOUND
- `tests/test_types_config.py` — FOUND
- `.planning/phases/01-foundation-and-data-pipeline/01-02-SUMMARY.md` — FOUND
- Commit `1cd8d45` — FOUND in git log
- Commit `7f6e8ff` — FOUND in git log
