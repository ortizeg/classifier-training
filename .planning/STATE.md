# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.
**Current focus:** Phase 3 in progress — Plans 01-02 complete (2/3). All 8 callbacks implemented. Remaining: Hydra callback configuration (plan 03).

## Current Position

Phase: 3 of 5 (Callbacks and ONNX Export)
Plan: 2 of 3 in current phase -- COMPLETE
Status: Plan 03-02 complete, continuing with 03-03
Last activity: 2026-02-18 — Plan 03-02 complete: 6 observability callbacks, 87 tests passing

Progress: [███████░░░] 70%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 4.0 min
- Total execution time: 0.47 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-data-pipeline | 3/3 | 11 min | 3.7 min |
| 02-model-layer | 2/2 | 7 min | 3.5 min |
| 03-callbacks-and-onnx-export | 2/3 | 10 min | 5.0 min |

**Recent Trend:**
- Last 5 plans: 01-03 (4 min), 02-01 (3 min), 02-02 (4 min), 03-01 (6 min), 03-02 (4 min)
- Trend: stable

*Updated after each plan completion*
| Phase 03 P02 | 4 | 2 tasks | 8 files |

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
- [Phase 02]: Explicit class_weights: torch.Tensor annotation on class to satisfy mypy strict with register_buffer
- [Phase 02]: Copied only @register decorator from sibling repo, not instantiate_* helpers
- [02-02]: **kwargs: Any (not object) for Hydra forwarded params -- mypy strict rejects **dict[str, object] for typed params
- [02-02]: ConfigStore repo is a plain dict in hydra-core 1.3.x -- use cs.repo.get() not cs.repo.list()
- [02-02]: pretrained=False parameter pattern for test usage -- avoids 44-98MB downloads in CI
- [03-01]: Warmup formula min(decay, (1+step)/(10+step)) -- smooth ramp matching sibling repo
- [03-01]: Legacy ONNX exporter via TORCH_ONNX_LEGACY_EXPORTER=1 + dynamo=False monkeypatch
- [03-01]: getattr(trainer, 'datamodule') for mypy compatibility -- trainer.datamodule not typed in Lightning stubs
- [03-01]: Real LightningModule in ONNX tests (not MagicMock) -- deepcopy compatibility with torch.onnx.export
- [Phase 03]: MulticlassConfusionMatrix initialized in on_fit_start for correct device placement
- [Phase 03]: matplotlib.use('Agg') inside plot methods, not module level -- avoids backend conflicts
- [Phase 03]: SamplerDistribution fires on_train_epoch_start; _last_indices empty at epoch 0, populated from epoch 1

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: EMA + ModelCheckpoint timing interaction (Lightning issue #11276) — plan-phase for Phase 3 should trace hook ordering before implementation
- [Research]: AMP stability on T4 with classification (`clip_val=1.0` vs `clip_val=5.0`) requires empirical smoke run in Phase 4 — plan for it explicitly
- [Resolved]: Actual basketball jersey numbers dataset class distribution characterized — 43 classes, '' at idx 0, 2930 train rows, 372 val rows, 365 test rows

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 03-02-PLAN.md -- 6 observability callbacks, 87 tests passing
Resume file: None
