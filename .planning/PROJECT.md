# Classifier Training

## What This Is

A production-ready image classification training framework built with PyTorch Lightning and Hydra, following the same architecture and conventions as the sibling `object-detection-training` repository. Supports ResNet family models (ResNet18, ResNet50) with configurable training on cloud GPUs (T4), mixed precision, EMA, and comprehensive callbacks for dataset statistics, confusion matrices, training history plots, and model info. Designed for basketball jersey number OCR classification and extensible to other classification tasks.

## Core Value

Train image classifiers with the same production-quality infrastructure, reproducibility, and cloud deployment workflow established in the object-detection-training repository — configurable via Hydra YAML, with full training observability through callbacks.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] ResNet model family support (ResNet18, ResNet50) with ImageNet pretrained weights
- [ ] PyTorch Lightning module for classification (training/validation/test steps, metrics)
- [ ] Hydra-based hierarchical config (model, data, trainer, callbacks, transforms config groups)
- [ ] ImageFolder dataset support with train/val/test splits and configurable transforms
- [ ] LightningDataModule for classification datasets with configurable augmentation pipeline
- [ ] EMA callback (Exponential Moving Average) with configurable decay and warmup
- [ ] Mixed precision training (16-mixed) for T4 GPU optimization
- [ ] Model checkpoint callback with accuracy-based monitoring
- [ ] Early stopping callback with configurable patience
- [ ] Learning rate monitor callback
- [ ] Dataset statistics callback (class distribution, sample counts, image sizes)
- [ ] Confusion matrix callback (per-epoch validation confusion matrix visualization)
- [ ] Training history plotter callback (loss/accuracy curves)
- [ ] Model info callback (parameter count, FLOPs, inference speed)
- [ ] Rich progress bar callback
- [ ] ONNX export callback for best/final models
- [ ] Pixi environment management with conda + PyPI dependencies
- [ ] Ruff linting and formatting with strict configuration
- [ ] MyPy strict type checking
- [ ] Pre-commit hooks (trailing whitespace, ruff, ruff-format, large files, secrets)
- [ ] Pytest test suite with coverage
- [ ] GitHub Actions CI (lint, test, release workflows)
- [ ] Dockerfile with CUDA 12.1, pixi-based dependency installation
- [ ] GCP Cloud Build integration (cloudbuild.yaml + cloud-build.sh script)
- [ ] Loguru structured logging throughout
- [ ] Pydantic v2 frozen models for configuration validation
- [ ] Default trainer config optimized for T4 GPU (gradient clipping, accumulation, precision)
- [ ] Specific Hydra configs for ResNet18 and ResNet50 with tuned hyperparameters
- [ ] Basketball jersey numbers OCR dataset config for validation training run
- [ ] Gitflow workflow (main + develop branches)
- [ ] GitHub repository with issue templates, PR templates, branch protection
- [ ] Semantic release with conventional commits
- [ ] Visualization callback (sample predictions with confidence scores)

### Out of Scope

- Object detection — handled by sibling `object-detection-training` repo
- Vision Transformers (ViT) — future model family, not v1
- Distributed multi-GPU training — single GPU (T4) target for v1
- W&B integration — defer to v2 (keep optional dependency structure)
- Model serving/inference API — training only for v1
- Custom backbone architectures — use torchvision pretrained models

## Context

- Sibling project: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/` provides the architectural template
- The object-detection-training repo uses: pixi, Hydra, PyTorch Lightning, loguru, pydantic v2, ruff, mypy, pre-commit, GitHub Actions, Docker + GCP Cloud Build, semantic release
- Target dataset: `basketball-jersey-numbers-ocr` — an image classification dataset with folder-per-class structure for jersey number recognition
- Same author and conventions as the detection repo — maintain consistency across the "Next Play" project ecosystem
- Python 3.11 pinned (matching object-detection-training)
- CUDA 12.1 for Linux/cloud, MPS for macOS development

## Constraints

- **Stack parity**: Must use identical tooling to object-detection-training (pixi, hydra, lightning, ruff, mypy, etc.)
- **Python version**: 3.11 (matching sibling repo)
- **Cloud target**: Google Cloud T4 GPU — all defaults tuned for this hardware
- **Build system**: flit_core (matching sibling repo)
- **Package layout**: src layout (`src/classifier_training/`)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mirror object-detection-training architecture | Consistency across Next Play ecosystem, proven patterns | — Pending |
| ResNet family first | Well-understood, fast to train, good baseline for classification | — Pending |
| ImageFolder dataset format | Standard PyTorch format, works with Roboflow exports | — Pending |
| flit_core build system | Matching sibling repo convention | — Pending |
| Gitflow (main + develop) | User requested, matches team workflow | — Pending |

---
*Last updated: 2026-02-18 after initialization*
