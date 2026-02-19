# Requirements: Classifier Training

**Defined:** 2026-02-18
**Core Value:** Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Project Foundation

- [x] **FOUND-01**: Pixi environment with Python 3.11, conda-forge channel, osx-arm64 + linux-64 platforms
- [x] **FOUND-02**: src layout (`src/classifier_training/`) with flit_core build system
- [x] **FOUND-03**: Ruff linting and formatting with strict rules matching sibling repo
- [x] **FOUND-04**: MyPy strict type checking with third-party ignore overrides
- [x] **FOUND-05**: Pre-commit hooks (trailing whitespace, ruff, ruff-format, large files, secrets)
- [x] **FOUND-06**: Pytest test suite with coverage reporting
- [x] **FOUND-07**: Pydantic v2 frozen models for configuration validation
- [x] **FOUND-08**: Loguru structured logging throughout (replaces stdlib logging)
- [x] **FOUND-09**: Reproducibility seed via `lightning.seed_everything`

### Data Pipeline

- [x] **DATA-01**: ImageFolder-compatible DataModule with train/val/test splits and configurable transforms
- [x] **DATA-02**: JSONL annotation support for datasets using `annotations.jsonl` format (basketball-jersey-numbers-ocr)
- [x] **DATA-03**: Strictly separated transform pipelines — train (augmentation) vs val/test (resize + normalize only)
- [x] **DATA-04**: Standard augmentation pipeline: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize
- [x] **DATA-05**: Val/test transforms: Resize, CenterCrop, Normalize with ImageNet statistics
- [x] **DATA-06**: `class_to_idx` mapping persisted as `labels_mapping.json` alongside model exports
- [x] **DATA-07**: Class weight tensor computed from training split for weighted loss
- [x] **DATA-08**: WeightedRandomSampler for class imbalance handling
- [x] **DATA-09**: Configurable num_workers, pin_memory, persistent_workers for DataLoader

### Model

- [x] **MODEL-01**: ResNet18 model with pretrained ImageNet weights via torchvision `weights=ResNet18_Weights.DEFAULT`
- [x] **MODEL-02**: ResNet50 model with pretrained ImageNet weights via torchvision `weights=ResNet50_Weights.DEFAULT`
- [x] **MODEL-03**: Base classification LightningModule with torchmetrics (auto device placement)
- [x] **MODEL-04**: CrossEntropyLoss with configurable `class_weight` tensor and `label_smoothing`
- [x] **MODEL-05**: Top-1 accuracy, Top-5 accuracy, per-class accuracy metrics via torchmetrics
- [x] **MODEL-06**: Consistent TorchMetrics logging pattern (Pattern A: update/compute/reset)
- [x] **MODEL-07**: AdamW optimizer with configurable learning rate (default lr=1e-4 for fine-tuning)
- [x] **MODEL-08**: CosineAnnealingLR scheduler with linear warmup
- [x] **MODEL-09**: Hydra config YAMLs for ResNet18 and ResNet50 with tuned hyperparameters

### Callbacks

- [x] **CALL-01**: EMA callback with configurable decay and warmup (ported from sibling repo)
- [x] **CALL-02**: ONNX export callback — exports from EMA state dict, output `["logits"]`, dynamic batch axis
- [x] **CALL-03**: Confusion matrix callback — per-epoch validation heatmap PNG, logged to WandB
- [x] **CALL-04**: Model info callback — FLOPs, parameters, model size, labels_mapping.json
- [x] **CALL-05**: Dataset statistics callback — class distribution table at training start
- [x] **CALL-06**: Training history plots callback — loss/accuracy curves saved per epoch
- [x] **CALL-07**: Sample visualization callback — predicted vs true class overlaid on images
- [x] **CALL-08**: Sampler distribution callback — validates class balancing is working
- [x] **CALL-09**: Rich progress bar callback
- [x] **CALL-10**: Learning rate monitor callback
- [x] **CALL-11**: Early stopping callback with configurable patience
- [x] **CALL-12**: Model checkpoint callback with accuracy-based monitoring (best + last)

### Training Configuration

- [x] **TRAIN-01**: Hydra hierarchical config with config groups: model, data, trainer, callbacks, transforms
- [x] **TRAIN-02**: Mixed precision training (`16-mixed`) for GPU optimization
- [x] **TRAIN-03**: Gradient clipping (`clip_val=1.0`) and gradient accumulation
- [x] **TRAIN-04**: Default trainer config optimized for T4 GPU (batch_size=64, num_workers=4)
- [x] **TRAIN-05**: WandB logging via WandbLogger
- [x] **TRAIN-06**: Basketball jersey numbers OCR dataset config for validation training run
- [x] **TRAIN-07**: ModelCheckpoint with fixed dirpath for resume stability

### Infrastructure

- [ ] **INFRA-01**: Dockerfile with CUDA 12.1, pixi-based dependency installation
- [ ] **INFRA-02**: GCP Cloud Build integration (cloudbuild.yaml + cloud-build.sh script)
- [ ] **INFRA-03**: GitHub Actions CI — lint workflow (format-check, lint, typecheck)
- [ ] **INFRA-04**: GitHub Actions CI — test workflow (pytest with coverage)
- [ ] **INFRA-05**: GitHub Actions CI — release workflow (semantic release)
- [ ] **INFRA-06**: Semantic release with conventional commits
- [ ] **INFRA-07**: Gitflow workflow (main + develop branches)
- [ ] **INFRA-08**: GitHub repository with issue templates, PR templates, branch protection

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enhanced Metrics

- **METR-01**: Per-class F1 score (macro/micro/per-class) via torchmetrics
- **METR-02**: AUROC per-class for production confidence thresholds
- **METR-03**: Prediction visualization callback with confidence scores

### Advanced Augmentation

- **AUG-01**: RandAugment as Hydra-configurable option
- **AUG-02**: MixUp augmentation with soft-label loss variant
- **AUG-03**: CutMix augmentation with soft-label loss variant

### Additional Schedulers

- **SCHED-01**: Step LR scheduler as Hydra config alternative to cosine

### Advanced Features

- **ADV-01**: GradCAM visualization callback for model interpretability
- **ADV-02**: SAM (Sharpness Aware Minimization) optimizer

## Out of Scope

| Feature | Reason |
|---------|--------|
| Object detection | Handled by sibling `object-detection-training` repo |
| Vision Transformers (ViT) | Future model family, not v1 |
| Distributed multi-GPU training | Single GPU (T4) target for v1 |
| Model serving/inference API | Training only; inference in basketball-2d-to-3d |
| Custom backbone architectures | Use torchvision pretrained models |
| Multi-label classification | Requires sigmoid + BCELoss; separate task type |
| Knowledge distillation | Significant complexity; not needed for v1 |
| Neural architecture search | Enormous compute cost; use model config sweeps |
| TorchScript export | ONNX is the established inference format |
| Real-time inference server | Serving belongs in inference pipeline |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Complete |
| FOUND-05 | Phase 1 | Complete |
| FOUND-06 | Phase 1 | Complete |
| FOUND-07 | Phase 1 | Complete |
| FOUND-08 | Phase 1 | Complete |
| FOUND-09 | Phase 1 | Complete |
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| DATA-07 | Phase 1 | Complete |
| DATA-08 | Phase 1 | Complete |
| DATA-09 | Phase 1 | Complete |
| MODEL-01 | Phase 2 | Complete |
| MODEL-02 | Phase 2 | Complete |
| MODEL-03 | Phase 2 | Complete |
| MODEL-04 | Phase 2 | Complete |
| MODEL-05 | Phase 2 | Complete |
| MODEL-06 | Phase 2 | Complete |
| MODEL-07 | Phase 2 | Complete |
| MODEL-08 | Phase 2 | Complete |
| MODEL-09 | Phase 2 | Complete |
| CALL-01 | Phase 3 | Complete |
| CALL-02 | Phase 3 | Complete |
| CALL-03 | Phase 3 | Complete |
| CALL-04 | Phase 3 | Complete |
| CALL-05 | Phase 3 | Complete |
| CALL-06 | Phase 3 | Complete |
| CALL-07 | Phase 3 | Complete |
| CALL-08 | Phase 3 | Complete |
| CALL-09 | Phase 3 | Complete |
| CALL-10 | Phase 3 | Complete |
| CALL-11 | Phase 3 | Complete |
| CALL-12 | Phase 3 | Complete |
| TRAIN-01 | Phase 4 | Complete |
| TRAIN-02 | Phase 4 | Complete |
| TRAIN-03 | Phase 4 | Complete |
| TRAIN-04 | Phase 4 | Complete |
| TRAIN-05 | Phase 4 | Complete |
| TRAIN-06 | Phase 4 | Complete |
| TRAIN-07 | Phase 4 | Complete |
| INFRA-01 | Phase 5 | Pending |
| INFRA-02 | Phase 5 | Pending |
| INFRA-03 | Phase 5 | Pending |
| INFRA-04 | Phase 5 | Pending |
| INFRA-05 | Phase 5 | Pending |
| INFRA-06 | Phase 5 | Pending |
| INFRA-07 | Phase 5 | Pending |
| INFRA-08 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 53 total
- Mapped to phases: 53
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-18*
*Last updated: 2026-02-18 after initial definition*
