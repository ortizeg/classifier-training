# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.
**Current focus:** Phase 4 COMPLETE -- Hydra training config wired, WandB logging, 101 tests passing. Ready for Phase 5 (Cloud Training).

## Current Position

Phase: 4 of 5 (Training Configuration) -- COMPLETE
Plan: 2 of 2 in current phase -- COMPLETE
Status: Phase 4 complete, ready for Phase 5
Last activity: 2026-02-18 — Plan 04-02 complete: WandB image logging and 11 config validation tests, 101 tests passing

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: 3.7 min
- Total execution time: 0.62 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-data-pipeline | 3/3 | 11 min | 3.7 min |
| 02-model-layer | 2/2 | 7 min | 3.5 min |
| 03-callbacks-and-onnx-export | 3/3 | 13 min | 4.3 min |
| 04-training-configuration | 2/2 | 6 min | 3.0 min |

**Recent Trend:**
- Last 5 plans: 03-01 (6 min), 03-02 (4 min), 03-03 (3 min), 04-01 (4 min), 04-02 (2 min)
- Trend: stable

*Updated after each plan completion*
| Phase 03 P03 | 3 | 2 tasks | 2 files |
| Phase 04 P02 | 2 | 2 tasks | 2 files |

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
- [03-03]: LightningDataModule subclass required for mock datamodule in tests -- trainer.datamodule requires proper LDM interface
- [03-03]: Dict-batch format for integration tests -- ConfusionMatrixCallback expects ClassificationBatch dict with images/labels keys
- [04-01]: Dual-init DataModule: config=DataModuleConfig OR flat kwargs from Hydra, with **kwargs absorbing _target_
- [04-01]: type: ignore[operator] for model.set_class_weights() -- mypy sees LightningModule, not BaseClassificationModel
- [04-01]: ModelCheckpoint dirpath='checkpoints' (relative) + default_root_dir=HydraConfig.cwd for stable resume
- [04-01]: Trainer built from flat dict, not hydra.utils.instantiate (no _target_ key)
- [04-01]: conf/model/ (singular) not conf/models/ -- Hydra requires exact match between directory name and defaults list key
- [Phase 04]: Factory fixture for Hydra override tests -- avoids GlobalHydra conflicts between compose calls
- [Phase 04]: pretrained=False override in model instantiation tests -- avoids weight downloads in CI

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: EMA + ModelCheckpoint timing interaction (Lightning issue #11276) — plan-phase for Phase 3 should trace hook ordering before implementation
- [Research]: AMP stability on T4 with classification (`clip_val=1.0` vs `clip_val=5.0`) requires empirical smoke run in Phase 4 — plan for it explicitly
- [Resolved]: Actual basketball jersey numbers dataset class distribution characterized — 43 classes, '' at idx 0, 2930 train rows, 372 val rows, 365 test rows

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 04-02-PLAN.md -- Phase 4 complete, 101 tests passing, ready for Phase 5
Resume file: None
