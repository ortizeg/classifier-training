# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.
**Current focus:** Phase 1 COMPLETE — ready for Phase 2 (Object Detection) + Phase 5 (Court Mapping) in parallel

## Current Position

Phase: 1 of 5 (Foundation and Data Pipeline) — COMPLETE
Plan: 3 of 3 in current phase — COMPLETE
Status: Phase 1 complete, ready for Phase 2
Last activity: 2026-02-18 — Plan 01-03 complete: JerseyNumberDataset, ImageFolderDataModule, 34 tests, all Phase 1 criteria

Progress: [███░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 3.7 min
- Total execution time: 0.18 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-data-pipeline | 3/3 | 11 min | 3.7 min |

**Recent Trend:**
- Last 5 plans: 01-01 (3 min), 01-02 (4 min), 01-03 (4 min)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Research]: Use `torchvision.models.resnet18/resnet50(weights=*.DEFAULT)` — not `timm` — for Phase 1-3 backbones; timm deferred until non-ResNet backbones are needed
- [Research]: Normalize in sidecar JSON (document in `labels_mapping.json`), not baked into ONNX graph — inference pipeline in basketball-2d-to-3d handles preprocessing
- [Research]: `class_to_idx` must be serialized to `labels_mapping.json` in Phase 1 (not deferred) — silent inference-breaking bug if omitted
- [01-01]: Python 3.11 pinned (matching object-detection-training sibling repo)
- [01-01]: flit_core build backend (matching sibling repo convention)
- [01-01]: `environments.default = ["dev"]` ensures dev tooling always available without separate env activation
- [01-01]: pytest 9 exits 5 on no tests collected — added smoke test (test_package.py) to satisfy `pixi run test` exits 0 success criterion
- [01-02]: TypedDict chosen for ClassificationBatch — Lightning training_step receives dict batches; Model unpacks batch["images"], batch["labels"]
- [01-02]: persistent_workers auto-correction uses object.__setattr__ inside model_validator (required for frozen pydantic models)
- [01-02]: MPS num_workers guard deferred to DataModule — config is a pure data model with no runtime torch access
- [01-02]: tmp_dataset_dir uses flat files + JSONL (not ImageFolder subdirs) — mirrors real basketball-jersey-numbers-ocr dataset structure
- [01-02]: Test path strings use /data/test instead of /tmp/data to avoid ruff S108 false positives
- [01-03]: RuntimeError instead of assert for setup() pre-condition guards — ruff S101 flags assert in src code; RuntimeError is semantically correct for API misuse
- [01-03]: from collections.abc import Callable instead of from typing import Callable — UP035 pyupgrade rule enforced by ruff
- [01-03]: len(dataset) = annotation rows not unique images — real dataset has 2930 rows for 2891 images; sampling by annotation row preserves per-label correctness
- [01-03]: '' (empty-string class) legitimately at index 0 — sorted(['', '0', ...]) places '' first; represents unreadable jersey numbers

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: EMA + ModelCheckpoint timing interaction (Lightning issue #11276) — plan-phase for Phase 3 should trace hook ordering before implementation
- [Research]: AMP stability on T4 with classification (`clip_val=1.0` vs `clip_val=5.0`) requires empirical smoke run in Phase 4 — plan for it explicitly
- [Resolved]: Actual basketball jersey numbers dataset class distribution characterized — 43 classes, '' at idx 0, 2930 train rows, 372 val rows, 365 test rows

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 01-03-PLAN.md — Phase 1 complete; JerseyNumberDataset, ImageFolderDataModule, 34 tests, all 5 criteria satisfied
Resume file: None
