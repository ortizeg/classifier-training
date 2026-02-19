# Classifier Training Project

## Project Overview

This is an image classification training framework built with PyTorch Lightning, Hydra, and specialized for training ResNet-family models. The project focuses on training classifiers with modern recipes including EMA, label smoothing, and class imbalance handling.

**Author**: Enrique G. Ortiz (ortizeg@gmail.com)
**Version**: 0.1.0
**Python Version**: 3.11

## Key Technologies

- **PyTorch Lightning**: Training framework
- **Hydra**: Configuration management
- **Pixi**: Package and environment management (replaces conda/pip)
- **ResNet**: CNN classification models (ResNet18, ResNet50)
- **Weights & Biases (wandb)**: Experiment tracking
- **ONNX**: Model export and deployment

## Project Structure

```
.
├── src/classifier_training/        # Main source code
│   ├── models/                     # Model implementations
│   │   ├── base.py                # BaseClassificationModel (Lightning)
│   │   └── resnet.py              # ResNet18/50 with @register
│   ├── callbacks/                  # Lightning callbacks
│   │   ├── ema.py                 # Exponential Moving Average
│   │   ├── onnx_export.py        # ONNX export with EMA support
│   │   ├── confusion_matrix.py   # Per-epoch confusion matrix heatmaps
│   │   ├── plotting.py           # Training history plots
│   │   ├── visualization.py      # Sample prediction grids
│   │   ├── model_info.py         # Model architecture info
│   │   ├── statistics.py         # Dataset statistics
│   │   └── sampler.py            # Sampler distribution monitoring
│   ├── data/                       # Data pipeline
│   │   ├── datamodule.py          # ImageFolderDataModule (Lightning)
│   │   ├── dataset.py             # JerseyNumberDataset (JSONL annotations)
│   │   └── sampler.py             # TrackingWeightedRandomSampler
│   ├── utils/                      # Utility functions
│   │   └── hydra.py               # @register decorator for ConfigStore
│   ├── conf/                       # Hydra configuration files
│   │   ├── train_jersey_ocr_resnet18.yaml      # GCP root config
│   │   ├── train_jersey_ocr_resnet18_local.yaml # Local root config
│   │   ├── model/                 # Model configs
│   │   ├── data/                  # Dataset configs
│   │   ├── callbacks/             # Callback configs
│   │   ├── trainer/               # Trainer configs
│   │   └── logging/               # Logger configs
│   ├── train.py                    # Training entrypoint
│   ├── config.py                   # Pydantic DataModuleConfig
│   └── types.py                    # ClassificationBatch TypedDict
├── tests/                          # Unit tests (100+)
├── scripts/                        # Build scripts
├── pixi.toml                       # Pixi project configuration
├── pyproject.toml                  # Python project metadata
├── Dockerfile                      # Docker configuration
└── .pre-commit-config.yaml         # Pre-commit hooks
```

## Development Workflow

### Environment Setup

This project uses **Pixi** (not conda/pip) for all dependency management:

```bash
pixi install
```

### Running Tasks

All tasks are managed through Pixi:

```bash
# Training
pixi run train                        # Default GCP config
pixi run train-local                  # Local macOS config

# Testing
pixi run test                         # Run pytest

# Code Quality
pixi run format                       # Format with ruff
pixi run lint                         # Lint with ruff
pixi run typecheck                    # MyPy strict mode
```

### Configuration with Hydra

The project uses Hydra for hierarchical configuration management. All configs are in `src/classifier_training/conf/`:

- Override from command line: `pixi run train -- model=resnet50 data.batch_size=64`
- Use config groups for different models, datasets, etc.

## Key Components

### Models

- **BaseClassificationModel** (`models/base.py`): Abstract Lightning module with Pattern A metrics, AdamW + SequentialLR (warmup + cosine), weighted CrossEntropyLoss
- **ResNet18/50** (`models/resnet.py`): Pretrained torchvision ResNets with replaced FC head, registered via `@register`

### Data Pipeline

- **JerseyNumberDataset**: JSONL-annotated dataset (one row per annotation, not per image)
- **ImageFolderDataModule**: Train/val/test splits, WeightedRandomSampler for class imbalance, strict transform separation
- **class_to_idx**: Built from train split only, alphabetically sorted, empty string at index 0

### Callbacks

- **EMA**: Exponential moving average of model weights
- **ONNX Export**: Auto-export at training end with EMA weights
- **Confusion Matrix**: Per-epoch heatmap with actual class labels
- **Training History**: Loss and accuracy plots
- **Sample Visualization**: Prediction grids with ground truth

## Code Style & Quality

- **Formatter**: Ruff (Black-compatible)
- **Linter**: Ruff (multi-rule linting)
- **Type Checking**: MyPy (strict mode)
- **Line Length**: 88 characters
- **Python Version**: 3.11

## Important Notes for AI Assistants

### When Making Changes

1. **Always use Pixi**: Don't suggest `pip install` or `conda`. Use `pixi add <package>`
2. **Respect Hydra configs**: Changes to training behavior should go in YAML configs, not hardcoded
3. **Type annotations**: This project uses strict typing (MyPy). Always add type hints
4. **Format code**: Run `pixi run format` before committing
5. **Tests**: Add tests in `tests/` for new functionality

### Dependencies

Key constraints:
- `numpy<2.0.0`: Compatibility with existing models
- `pydantic>=2.0`: Modern Pydantic API
- `onnxscript>=0.5.6,<0.6`: Specific ONNX version requirements

### CUDA Support

- Linux: CUDA 12.1
- macOS: MPS (Metal Performance Shaders), num_workers=0

## Common Patterns

### Adding a New Model

1. Create model class extending `BaseClassificationModel` in `src/classifier_training/models/`
2. Use `@register(group="model", name="model_name")` decorator
3. Hydra config is auto-generated via ConfigStore

### Adding a Callback

1. Implement callback in `src/classifier_training/callbacks/`
2. Add entry in `src/classifier_training/conf/callbacks/default.yaml`
3. For local-only: also add to `callbacks/local.yaml`

### The @register Decorator

Models are registered with Hydra's ConfigStore via the `@register` decorator:

```python
from classifier_training.utils.hydra import register

@register(group="model", name="resnet18")
class ResNet18ClassificationModel(BaseClassificationModel):
    ...
```

## Output Structure

Training outputs go to `outputs/` (Hydra-managed):
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/              # Hydra config snapshots
        ├── checkpoints/         # Model checkpoints
        ├── model.onnx           # Exported ONNX model
        ├── labels_mapping.json  # Class mapping + normalization
        ├── epoch_XXX/           # Per-epoch artifacts
        │   ├── confusion_matrix.png
        │   └── sample_predictions.png
        └── training_history/    # Loss/accuracy plots
```
