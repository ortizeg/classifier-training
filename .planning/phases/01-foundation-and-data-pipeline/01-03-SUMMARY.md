---
phase: 01-foundation-and-data-pipeline
plan: "03"
subsystem: data

tags: [pytorch, lightning, torchvision, pydantic, jsonl, weighted-sampler, transforms, pytest]

# Dependency graph
requires:
  - phase: 01-foundation-and-data-pipeline
    plan: "01"
    provides: "pixi env, src-layout package, ruff/mypy/pytest tooling"
  - phase: 01-foundation-and-data-pipeline
    plan: "02"
    provides: "DataModuleConfig frozen pydantic model, ClassificationBatch TypedDict, tmp_dataset_dir conftest fixture"
provides:
  - "JerseyNumberDataset: flat JSONL-annotated dataset; len = annotation rows (not unique image count)"
  - "ImageFolderDataModule: LightningDataModule with WeightedRandomSampler, strict transform separation"
  - "class_to_idx built from train split only, alphabetically sorted ('' at index 0)"
  - "save_labels_mapping() writes class_to_idx + normalization stats to labels_mapping.json"
  - "IMAGENET_MEAN/IMAGENET_STD constants exported from datamodule"
  - "34-test suite: 6 dataset unit tests, 11 synthetic datamodule tests, 8 real dataset integration tests"
  - "Phase 1 complete: all 5 success criteria verified against real basketball-jersey-numbers-ocr dataset"
affects:
  - "Phase 2 (Model): imports DataModuleConfig, uses ImageFolderDataModule in training loop"
  - "Phase 3 (Training): Trainer.fit(dm) uses setup()/train_dataloader()/val_dataloader()"
  - "Phase 5 (ONNX export): save_labels_mapping() writes labels_mapping.json alongside model"
  - "basketball-2d-to-3d inference pipeline: reads labels_mapping.json for class mapping + normalization"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "JerseyNumberDataset reads annotation rows (not image files) — len = annotation count"
    - "class_to_idx: built from train only, sorted() for alphabetical order, '' sorts before numerics"
    - "Transform strict separation: augmentation assigned at Dataset construction, not in __getitem__"
    - "WeightedRandomSampler with shuffle=False — mutually exclusive with PyTorch DataLoader shuffle"
    - "RuntimeError (not assert) for pre-condition violations — S101 bandit compliant in src code"
    - "MPS guard in DataModule __init__: auto-sets num_workers=0 on Apple Silicon"
    - "labels_mapping.json sidecar pattern: normalization documented in JSON, not baked into ONNX"

key-files:
  created:
    - "src/classifier_training/data/__init__.py"
    - "src/classifier_training/data/dataset.py"
    - "src/classifier_training/data/datamodule.py"
    - "tests/test_dataset.py"
    - "tests/test_datamodule.py"
  modified: []

key-decisions:
  - "RuntimeError instead of assert for setup() pre-condition checks — ruff S101 flags assert in src; assert is acceptable only in tests"
  - "'' (empty-string) class legitimate at index 0 — alphabetical sort places it before '0', '1', etc."
  - "len(dataset) = len(annotation_rows) not len(image_files) — some images have multiple annotation rows in real dataset (2930 rows, 2891 images)"
  - "from collections.abc import Callable instead of from typing import Callable — UP035 pyupgrade rule"

patterns-established:
  - "Data pipeline uses RuntimeError (not assert) for all setup() guard conditions"
  - "Test files use underscore prefix (_img, _labels) for unused unpacked variables — RUF059 compliant"
  - "Test import assertions use hasattr() pattern to avoid N814 (CamelCase as constant) violations"
  - "Integration tests use pytest.mark.skipif(not path.exists()) to gate on real dataset presence"

# Metrics
duration: 4min
completed: 2026-02-18
---

# Phase 1 Plan 03: Data Pipeline Summary

**JerseyNumberDataset (JSONL-annotated) + ImageFolderDataModule (WeightedRandomSampler, strict transform separation, labels_mapping.json) with 34 tests — all Phase 1 criteria verified against real basketball-jersey-numbers-ocr dataset (43 classes, 2930 train rows, '' at idx 0)**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-02-18T23:56:12Z
- **Completed:** 2026-02-19T00:00:52Z
- **Tasks:** 2
- **Files created:** 5

## Accomplishments

- `JerseyNumberDataset` reads JSONL annotations directly; `len()` equals annotation rows (2930), not unique image count (2891) — critical for correct training sample counts
- `ImageFolderDataModule` with `WeightedRandomSampler` on train (inverse class frequency), deterministic val/test (no augmentation), MPS guard auto-zeroes `num_workers`
- `class_to_idx` built from train split only, alphabetical sort (`sorted()`), `''` (empty string) correctly at index 0
- `save_labels_mapping()` writes `labels_mapping.json` with `class_to_idx`, `idx_to_class`, and ImageNet normalization stats — sidecar for ONNX inference pipeline
- 34 tests pass: 6 dataset unit tests, 11 synthetic DataModule tests, 8 real dataset integration tests (real dataset present)
- All Phase 1 success criteria satisfied: importable, 43 classes, `''` at idx 0, 2930 train rows, val determinism confirmed

## Task Commits

Each task was committed atomically:

1. **Task 1: JerseyNumberDataset and ImageFolderDataModule** - `51d60c2` (feat)
2. **Task 2: Dataset and DataModule test suites** - `7c3cd13` (feat)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `src/classifier_training/data/__init__.py` - Public API: re-exports `ImageFolderDataModule`
- `src/classifier_training/data/dataset.py` - `JerseyNumberDataset`: JSONL reader, `samples` list from annotation rows
- `src/classifier_training/data/datamodule.py` - `ImageFolderDataModule`: full LightningDataModule with sampler, transforms, save_labels_mapping
- `tests/test_dataset.py` - 6 unit tests: len, getitem, transform, label range, splits, class_to_idx stored
- `tests/test_datamodule.py` - 11 synthetic + 8 real dataset integration tests; all Phase 1 criteria covered

## Decisions Made

- `RuntimeError` instead of `assert` for `setup()` pre-condition guards — ruff's S101 (bandit) rule flags `assert` in non-test source code. `RuntimeError` is semantically correct for API misuse.
- `from collections.abc import Callable` instead of `from typing import Callable` — UP035 pyupgrade rule enforced by ruff
- `''` (empty-string class) legitimately sorts first alphabetically — `sorted(['', '0', '1', ...])` places `''` at index 0. This is correct, not a bug: represents unreadable jersey numbers.
- `len(dataset)` = annotation rows, not images — the real dataset has 2930 annotation rows for 2891 unique images; some images appear in multiple annotations. Sampling by row preserves per-annotation label correctness.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced assert with RuntimeError in datamodule.py**
- **Found during:** Task 1 (lint verification)
- **Issue:** ruff S101 flags `assert` in src code (non-test files). Plan provided `assert` statements for `setup()` guard conditions.
- **Fix:** Replaced all 5 `assert` statements with `if ... raise RuntimeError(...)` in `datamodule.py`
- **Files modified:** `src/classifier_training/data/datamodule.py`
- **Verification:** `pixi run lint` exits 0
- **Committed in:** `51d60c2` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed UP035 in dataset.py: Callable import from collections.abc**
- **Found during:** Task 1 (lint verification)
- **Issue:** `from typing import Callable` triggers UP035 (pyupgrade) — `Callable` moved to `collections.abc` in Python 3.9+
- **Fix:** Changed to `from collections.abc import Callable`
- **Files modified:** `src/classifier_training/data/dataset.py`
- **Verification:** `pixi run lint` exits 0
- **Committed in:** `51d60c2` (Task 1 commit)

**3. [Rule 1 - Bug] Fixed E501 line length violations in datamodule.py**
- **Found during:** Task 1 (lint verification)
- **Issue:** 4 lines exceeded 88-char limit: MPS comment (89), docstring (90, 91), assert message (89)
- **Fix:** Wrapped long comment/docstring/message strings to fit 88-char limit
- **Files modified:** `src/classifier_training/data/datamodule.py`
- **Verification:** `pixi run lint` exits 0
- **Committed in:** `51d60c2` (Task 1 commit)

**4. [Rule 1 - Bug] Fixed 6 lint violations in test files**
- **Found during:** Task 2 (lint verification)
- **Issue:** N814 (CamelCase imported as constant `DM`), RUF100 (unused noqa), E501 x2 (long lines in test_datamodule.py/test_dataset.py), RUF059 x2 (unused unpacked variables `labels`, `img`)
- **Fix:** Replaced `import ... as DM` with `hasattr()` check; wrapped long lines; prefixed unused vars with `_`
- **Files modified:** `tests/test_datamodule.py`, `tests/test_dataset.py`
- **Verification:** `pixi run lint` exits 0; all 34 tests pass
- **Committed in:** `7c3cd13` (Task 2 commit)

---

**Total deviations:** 4 auto-fixed (all Rule 1 — lint compliance)
**Impact on plan:** Pure lint/style fixes matching established ruff rule set. No behavioral changes. `RuntimeError` is semantically equivalent to `assert` for API misuse detection, and produces clearer error messages.

## Issues Encountered

None beyond the lint deviations documented above. Real dataset was present at the expected path, so all 8 integration tests ran and passed on first attempt.

## User Setup Required

None — no external service configuration required. Real dataset is locally available at `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-jersey-numbers-ocr`.

## Next Phase Readiness

- Phase 1 complete: all 3 plans done, all 5 success criteria satisfied against real dataset
- `ImageFolderDataModule` contract stable — Phase 2 (Model) and Phase 3 (Training) can proceed
- `DataModuleConfig` + `ClassificationBatch` types form the data contract for model training loop
- Phase 2 (Object Detection) and Phase 5 (Court Mapping) can now start in parallel per ROADMAP
- `labels_mapping.json` output format established — basketball-2d-to-3d inference pipeline can be wired

---
*Phase: 01-foundation-and-data-pipeline*
*Completed: 2026-02-18*

## Self-Check: PASSED

- `src/classifier_training/data/__init__.py` — FOUND
- `src/classifier_training/data/dataset.py` — FOUND
- `src/classifier_training/data/datamodule.py` — FOUND
- `tests/test_dataset.py` — FOUND
- `tests/test_datamodule.py` — FOUND
- `.planning/phases/01-foundation-and-data-pipeline/01-03-SUMMARY.md` — FOUND
- Commit `51d60c2` — FOUND in git log
- Commit `7c3cd13` — FOUND in git log
