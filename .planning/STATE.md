# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.
**Current focus:** Phase 5 Plan 02 COMPLETE -- GitHub CI/CD workflows, templates, CODEOWNERS, develop branch. Pending: branch protection (human action), GitHub remote setup.

## Current Position

Phase: 5 of 5 (Infrastructure) -- IN PROGRESS
Plan: 2 of 2 in current phase -- COMPLETE (pending human action: branch protection)
Status: Phase 5 plan 02 complete; branch protection requires manual GitHub UI configuration
Last activity: 2026-02-19 — Plan 05-02 complete: 3 CI/CD workflows, 4 templates/CODEOWNERS, develop branch created

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 3.5 min
- Total execution time: 0.65 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-data-pipeline | 3/3 | 11 min | 3.7 min |
| 02-model-layer | 2/2 | 7 min | 3.5 min |
| 03-callbacks-and-onnx-export | 3/3 | 13 min | 4.3 min |
| 04-training-configuration | 2/2 | 6 min | 3.0 min |
| 05-infrastructure | 2/2 | 4 min | 2.0 min |

**Recent Trend:**
- Last 5 plans: 03-03 (3 min), 04-01 (4 min), 04-02 (2 min), 05-01 (2 min), 05-02 (2 min)
- Trend: stable/fast

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
- [05-02]: Workflows ported identically from object-detection-training sibling repo — same action versions, same CUDA override, same pixi integration
- [05-02]: CODEOWNERS references classifier_training (not object_detection_training) for conf/ path
- [05-02]: develop branch created locally only — no remote configured; push deferred until GitHub repo is set up
- [05-02]: Branch protection (Task 4) requires human action via GitHub Settings UI — not automatable

### Pending Todos

- Configure GitHub remote: `git remote add origin <url> && git push -u origin main && git push -u origin develop`
- Configure branch protection on main and develop (see 05-02-SUMMARY.md for detailed steps)

### Blockers/Concerns

- [Research]: EMA + ModelCheckpoint timing interaction (Lightning issue #11276) — plan-phase for Phase 3 should trace hook ordering before implementation
- [Research]: AMP stability on T4 with classification (`clip_val=1.0` vs `clip_val=5.0`) requires empirical smoke run in Phase 4 — plan for it explicitly
- [Resolved]: Actual basketball jersey numbers dataset class distribution characterized — 43 classes, '' at idx 0, 2930 train rows, 372 val rows, 365 test rows
- [Human Action Required]: Branch protection on main and develop — configure via GitHub Settings UI after remote is set up

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 05-02-PLAN.md -- Phase 5 complete (plan 2/2), 7 CI/CD files created, develop branch exists locally, branch protection requires human action
Resume file: None
