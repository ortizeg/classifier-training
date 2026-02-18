# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.
**Current focus:** Phase 1 — Foundation and Data Pipeline

## Current Position

Phase: 1 of 5 (Foundation and Data Pipeline)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-02-18 — Plan 01-01 complete: project scaffold (pixi + flit_core + ruff + mypy + pre-commit + pytest)

Progress: [█░░░░░░░░░] 7%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 3 min
- Total execution time: 0.05 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-data-pipeline | 1/3 | 3 min | 3 min |

**Recent Trend:**
- Last 5 plans: 01-01 (3 min)
- Trend: -

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: EMA + ModelCheckpoint timing interaction (Lightning issue #11276) — plan-phase for Phase 3 should trace hook ordering before implementation
- [Research]: AMP stability on T4 with classification (`clip_val=1.0` vs `clip_val=5.0`) requires empirical smoke run in Phase 4 — plan for it explicitly
- [Research]: Actual basketball jersey numbers dataset class distribution not yet characterized — `DatasetStatisticsCallback` is a blocking deliverable in Phase 1

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 01-01-PLAN.md — project scaffold complete
Resume file: None
