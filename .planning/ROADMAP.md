# Roadmap: Classifier Training

## Overview

A production-ready image classification training framework built to mirror the sibling `object-detection-training` repository. Five phases follow the strict import dependency order of the codebase: Foundation and Data Pipeline first (everything depends on it), then the Model Layer, then Callbacks and ONNX Export (which depend on stable model and data contracts), then Training Configuration (which wraps the completed codebase), and finally Infrastructure (CI, Docker, Cloud Build). Each phase delivers a coherent, independently verifiable capability. The framework is complete when a training run on the basketball jersey numbers dataset produces an ONNX model that the `basketball-2d-to-3d` inference pipeline can consume correctly.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation and Data Pipeline** - Project scaffold, data module, transforms, class imbalance handling ✓ (2026-02-18)
- [x] **Phase 2: Model Layer** - ResNet backbones, LightningModule, metrics, optimizer, scheduler ✓ (2026-02-18)
- [ ] **Phase 3: Callbacks and ONNX Export** - All callbacks including EMA, ONNX export, confusion matrix
- [ ] **Phase 4: Training Configuration** - Hydra configs, WandB, trainer defaults, checkpoint resume
- [ ] **Phase 5: Infrastructure** - Docker, GCP Cloud Build, GitHub Actions CI, semantic release

## Phase Details

### Phase 1: Foundation and Data Pipeline

**Goal**: The project scaffold is in place and the data pipeline can load the basketball jersey numbers dataset with strictly separated transforms, class imbalance handling, and `labels_mapping.json` serialization — eliminating all three data-layer critical pitfalls before any model code is written.

**Depends on**: Nothing (first phase)

**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, FOUND-07, FOUND-08, FOUND-09, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09

**Success Criteria** (what must be TRUE):
  1. `pixi run python -c "from classifier_training.data import ImageFolderDataModule"` succeeds without errors on both osx-arm64 and linux-64
  2. `pixi run pytest` passes and reports coverage — the test suite exists and runs
  3. `pixi run lint` and `pixi run typecheck` both pass — ruff and mypy are configured and enforced
  4. `DataModule.setup()` on the basketball jersey numbers dataset produces train/val/test `DataLoader` instances with no augmentation on val/test — verified by inspecting transform pipelines in tests
  5. `labels_mapping.json` is written to disk after `DataModule.setup()` and contains the correct alphabetically-ordered `class_to_idx` mapping for all 43 classes

**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffold: pixi.toml, pyproject.toml, src layout, pre-commit, ruff, mypy
- [x] 01-02-PLAN.md — Core types and configuration: DataModuleConfig pydantic model, ClassificationBatch TypedDict, test conftest fixtures
- [x] 01-03-PLAN.md — ImageFolderDataModule: JerseyNumberDataset, JSONL parsing, transforms, WeightedRandomSampler, labels_mapping.json

### Phase 2: Model Layer

**Goal**: `ResNet18ClassificationModel` and `ResNet50ClassificationModel` can complete a forward pass, log Top-1/Top-5/per-class accuracy using the correct Pattern A torchmetrics logging, and are fully configurable via Hydra YAML — with AdamW + cosine LR + linear warmup and class-weighted CrossEntropyLoss ready for training.

**Depends on**: Phase 1

**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, MODEL-07, MODEL-08, MODEL-09

**Success Criteria** (what must be TRUE):
  1. `pixi run pytest tests/test_model.py` passes — forward pass, loss computation, and metric logging tested for both ResNet18 and ResNet50
  2. `pixi run python -c "from classifier_training.models import ResNet18ClassificationModel, ResNet50ClassificationModel"` succeeds, confirming both models are importable and registered
  3. Top-1 accuracy, Top-5 accuracy, and per-class accuracy are logged per epoch with no NaN or 0.0 artifacts — confirmed by the Pattern A (update/compute/reset) unit test
  4. A Hydra config override `model=resnet18` or `model=resnet50` selects the correct backbone with the tuned default `lr=1e-4`

**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — Infrastructure (hydra-core in pixi.toml, @register decorator), BaseClassificationModel with CrossEntropyLoss buffer, Pattern A torchmetrics, AdamW + SequentialLR
- [x] 02-02-PLAN.md — ResNet18ClassificationModel and ResNet50ClassificationModel (torchvision pretrained, @register), Hydra YAML configs, full test suite (test_model.py)

### Phase 3: Callbacks and ONNX Export

**Goal**: All callbacks are implemented, wired into `conf/callbacks/default.yaml`, and the EMA + ModelCheckpoint + ONNX export integration is tested end-to-end — confirming that the ONNX export reflects EMA weights and that the confusion matrix callback does not crash on GPU.

**Depends on**: Phase 2

**Requirements**: CALL-01, CALL-02, CALL-03, CALL-04, CALL-05, CALL-06, CALL-07, CALL-08, CALL-09, CALL-10, CALL-11, CALL-12

**Success Criteria** (what must be TRUE):
  1. A smoke training run (3 epochs, small batch) completes without error, producing a checkpoint, a `model.onnx` file, a `labels_mapping.json` sidecar, and confusion matrix PNG files in the output directory
  2. The ONNX model's output name is `"logits"` and the model runs correctly under `onnxruntime` with `CPUExecutionProvider`
  3. `pixi run pytest tests/test_callbacks.py` passes — EMA state dict correctness, ONNX export from EMA weights, and confusion matrix device handling are all unit tested
  4. Class distribution table is printed at training start by `DatasetStatisticsCallback`
  5. `SamplerDistributionCallback` logs class sample counts to confirm `WeightedRandomSampler` is balancing the training batches

**Plans:** 3 plans

Plans:
- [ ] 03-01-PLAN.md — Dependencies + EMACallback + ONNXExportCallback + TrackingWeightedRandomSampler + tests
- [ ] 03-02-PLAN.md — Observability callbacks (ConfusionMatrix, DatasetStats, ModelInfo, TrainingHistory, SamplerDist, SampleViz) + tests
- [ ] 03-03-PLAN.md — Hydra conf/callbacks/default.yaml (all 12 callbacks) + integration test

### Phase 4: Training Configuration

**Goal**: A complete end-to-end training run on the basketball jersey numbers dataset can be launched with a single `pixi run train` command, checkpoints resume correctly, WandB receives all metrics and artifact logs, and the T4 GPU defaults are verified correct.

**Depends on**: Phase 3

**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07

**Success Criteria** (what must be TRUE):
  1. `pixi run train` with no overrides launches a training run using the basketball jersey numbers dataset config and the ResNet18 model config — the process starts, logs to WandB, and does not crash in the first epoch
  2. Interrupting training and re-running `pixi run train` resumes from the last checkpoint rather than starting over — verified by step counter in WandB
  3. WandB receives loss, Top-1 accuracy, Top-5 accuracy, learning rate, and confusion matrix images each epoch
  4. Mixed precision (`16-mixed`), gradient clipping (`clip_val=1.0`), and gradient accumulation are active — confirmed by Lightning Trainer summary at run start
  5. A Hydra config override (`model=resnet50`, `data.batch_size=32`) changes the run configuration correctly

**Plans**: TBD

Plans:
- [ ] 04-01: Hydra config composition — config groups for model, data, trainer, callbacks, transforms; `conf/train_basketball_resnet18.yaml` dataset override
- [ ] 04-02: Trainer defaults and WandB — T4-tuned trainer config, WandbLogger, ModelCheckpoint with fixed dirpath, end-to-end training run validation

### Phase 5: Infrastructure

**Goal**: The repository can be built as a Docker image, pushed via GCP Cloud Build, runs lint/test/typecheck in GitHub Actions CI on every push, and releases new versions automatically via semantic release — matching the sibling repo's CI/CD pipeline.

**Depends on**: Phase 4

**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08

**Success Criteria** (what must be TRUE):
  1. `docker build -f Dockerfile .` succeeds locally, producing an image with CUDA 12.1, pixi, and the `classifier_training` package installed
  2. `gcloud builds submit` using `cloudbuild.yaml` completes without error and pushes a versioned image to GCR
  3. A pull request to `develop` triggers GitHub Actions CI and all three workflows (lint, test, release) run — lint and test pass on the PR
  4. A conventional commit merged to `main` triggers the release workflow and creates a new GitHub release with a version bump
  5. The repository has issue templates, a PR template, and branch protection rules configured on `main` and `develop`

**Plans**: TBD

Plans:
- [ ] 05-01: Docker and GCP Cloud Build — Dockerfile (CUDA 12.1 + pixi), cloudbuild.yaml, cloud-build.sh script
- [ ] 05-02: GitHub Actions CI — lint workflow, test workflow, release workflow; semantic release config; Gitflow branch protection

## Coverage Validation

All 53 v1 requirements mapped. No orphans.

| Category | Requirements | Phase |
|----------|-------------|-------|
| Foundation | FOUND-01 through FOUND-09 (9 requirements) | Phase 1 |
| Data Pipeline | DATA-01 through DATA-09 (9 requirements) | Phase 1 |
| Model | MODEL-01 through MODEL-09 (9 requirements) | Phase 2 |
| Callbacks | CALL-01 through CALL-12 (12 requirements) | Phase 3 |
| Training Config | TRAIN-01 through TRAIN-07 (7 requirements) | Phase 4 |
| Infrastructure | INFRA-01 through INFRA-08 (8 requirements) | Phase 5 |

**Total: 53/53 v1 requirements mapped.**

## Dependency Graph

```
Phase 1: Foundation and Data Pipeline
    |
    v
Phase 2: Model Layer
    |
    v
Phase 3: Callbacks and ONNX Export
    |
    v
Phase 4: Training Configuration
    |
    v
Phase 5: Infrastructure
```

Each phase is a strict dependency of the next. The import graph enforces this order: types and utilities (Phase 1) have no internal imports; the data layer (Phase 1) depends on types; the model layer (Phase 2) depends on types and the data module for `num_classes`; callbacks (Phase 3) depend on stable model and data contracts; training configuration (Phase 4) wraps all code; infrastructure (Phase 5) wraps the entire codebase.

## Progress

**Execution Order:** 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation and Data Pipeline | 3/3 | ✓ Complete | 2026-02-18 |
| 2. Model Layer | 2/2 | ✓ Complete | 2026-02-18 |
| 3. Callbacks and ONNX Export | 0/3 | Not started | - |
| 4. Training Configuration | 0/2 | Not started | - |
| 5. Infrastructure | 0/2 | Not started | - |

---
*Roadmap created: 2026-02-18*
*Last updated: 2026-02-18 — Phase 2 complete (2/2 plans, 55 tests, verified 7/7)*
