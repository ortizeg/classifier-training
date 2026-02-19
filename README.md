[![Test Suite](https://github.com/ortizeg/classifier-training/actions/workflows/test.yml/badge.svg)](https://github.com/ortizeg/classifier-training/actions/workflows/test.yml)
[![Lint & Format](https://github.com/ortizeg/classifier-training/actions/workflows/lint.yml/badge.svg)](https://github.com/ortizeg/classifier-training/actions/workflows/lint.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

# Classifier Training

End-to-end image classification framework for training, exporting, and deploying CNN classifiers with modern training recipes.

**Train** classifiers (ResNet family) with PyTorch Lightning and Hydra config management. **Export** to ONNX for production inference. **Track** experiments with Weights & Biases.

## Features

- **ResNet model family** &mdash; ResNet18 and ResNet50 with pretrained ImageNet initialization and configurable head replacement
- **Hydra configuration** &mdash; hierarchical YAML configs with full CLI override support for models, datasets, callbacks, and trainers
- **ONNX export** &mdash; automatic checkpoint-to-ONNX conversion at end of training with EMA weight support
- **Class imbalance handling** &mdash; inverse-frequency weighted CrossEntropyLoss and WeightedRandomSampler for long-tail distributions
- **Experiment tracking** &mdash; Weights & Biases integration with confusion matrix logging
- **Training callbacks** &mdash; EMA, confusion matrices, dataset statistics, training history plots, sample visualizations, sampler distribution monitoring
- **Production-ready** &mdash; Docker builds for GCP Cloud Build, GitHub Actions CI, strict MyPy typing, 100+ unit tests

## Quick Start

### Prerequisites

- [Pixi](https://pixi.sh) for environment and dependency management
- CUDA 12.1 (Linux) or MPS (macOS) for GPU acceleration

### Install & Run

```bash
git clone https://github.com/ortizeg/classifier-training.git
cd classifier-training
pixi install

# Train locally (macOS / CPU)
pixi run train-local

# Train with default config
pixi run train

# Override any config from CLI
pixi run train -- model=resnet50 data.batch_size=64 trainer.max_epochs=50
```

### Code Quality

```bash
pixi run format       # Ruff formatter
pixi run lint         # Ruff linter
pixi run typecheck    # MyPy strict mode
pixi run test         # Pytest suite
```

## Architecture

```
src/classifier_training/
  models/          ResNet18/50 with BaseClassificationModel Lightning module
  data/            JSONL-annotated ImageFolder datamodule with weighted sampling
  callbacks/       EMA, ONNX export, confusion matrix, plotting, visualization
  conf/            Hydra YAML configs (models, data, callbacks, trainer, logging)
  utils/           Hydra @register decorator and config helpers
  train.py         Training entrypoint (@hydra.main)
  config.py        Pydantic frozen DataModuleConfig
  types.py         ClassificationBatch TypedDict
```

## Configuration

All configs live in `src/classifier_training/conf/`:

```
conf/
├── train_jersey_ocr_resnet18.yaml       # GCP training config
├── train_jersey_ocr_resnet18_local.yaml  # Local training config
├── model/              # Model configs (resnet18, resnet50)
├── data/               # Dataset configs
├── callbacks/          # Callback configs (default, local)
├── trainer/            # Trainer configs (default, local)
└── logging/            # Logger configs (wandb, none)
```

Switch between configurations using Hydra overrides:

```bash
# Use ResNet50 instead of ResNet18
pixi run train -- model=resnet50

# Change batch size and epochs
pixi run train -- data.batch_size=64 trainer.max_epochs=100
```

## Docker

```bash
pixi run build         # Cloud Build on GCP (fast, cached)
pixi run build-local   # Local Docker build
```

The image targets GCP Cloud Build. Secrets (`WANDB_API_KEY`, etc.) are injected at runtime via environment variables.

## License

Apache License 2.0
