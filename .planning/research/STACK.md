# Stack Research

**Domain:** Image classification training framework (Python, PyTorch)
**Researched:** 2026-02-18
**Confidence:** HIGH (core framework verified via official docs/PyPI; version pins verified Feb 2026)

---

## Context

This repo mirrors the object-detection-training sibling. Every tooling choice below
must stay in lockstep with that repo unless a classification-specific reason overrides it.
Where the sibling already has a working pattern, this doc confirms or refines it.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | `3.11.*` | Runtime | Mirrors sibling; Lightning 2.6.1 supports 3.10–3.13, but 3.11 is the proven stable target. Do NOT move to 3.12 yet — onnxruntime wheel issues documented in sibling memory. |
| PyTorch (`pytorch`) | `>=2.5,<2.11` (conda-forge pin) | Deep learning runtime | Current conda-forge stable is 2.10.0 (Jan 2026). Pixi resolves from conda-forge; let solver pick latest compatible. |
| torchvision | `*` (solver-pinned to match pytorch) | ResNet model definitions, pretrained weights, transforms v2, ImageFolder | Official: torchvision 0.25 pairs with PyTorch 2.10. Contains `ResNet18_Weights.DEFAULT` and `ResNet50_Weights.DEFAULT` — the canonical way to load ImageNet weights. |
| PyTorch Lightning (`lightning`) | `>=2.5,<3` | Training loop orchestration | Current stable: 2.6.1 (Jan 30 2026). Provides LightningModule, LightningDataModule, Trainer. Mirrored from sibling; classification step structure is a subset of the detection pattern. |
| hydra-core | `1.3.2` | Hierarchical YAML config | Last release Feb 2023, API stable. Mirrored from sibling exactly; same Hydra `@hydra.main` + config-group pattern. Pin `==1.3.2` to avoid surprises. |
| pydantic | `>=2.0` | Schema validation for configs and IO contracts | Current: 2.12.5. Mirrored from sibling. Use frozen Pydantic models for inter-component contracts (e.g., `ClassificationBatch`). |
| loguru | `>=0.7,<1` | Structured logging | Current: 0.7.3. Mirrored from sibling. Replace stdlib logging entirely; `logger.info(...)` with lazy formatting. |
| torchmetrics | `>=1.0` | Classification metrics (Accuracy, F1, AUROC, ConfusionMatrix) | Current: 1.8.2. Use `MulticlassAccuracy`, `MulticlassF1Score`, `MulticlassConfusionMatrix`, `MulticlassAUROC`. NOT raw sklearn — torchmetrics is GPU-native, DDP-safe, and integrates with `self.log()`. |

### Classification-Specific Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchvision (transforms.v2) | same as torchvision above | Data augmentation | Always. Use `torchvision.transforms.v2` (not v1). v2 is the current API; v1 frozen, no new features. Provides `RandomResizedCrop`, `RandomHorizontalFlip`, `AutoAugment`, `TrivialAugmentWide`, `CutMix`, `MixUp`. |
| torchvision (datasets.ImageFolder) | same | Dataset loading from directory structure | When training data is organized as `root/class_name/image.jpg`. This is the standard for custom classification datasets. Pair with `LightningDataModule`. |
| wandb | `>=0.25,<1` (PyPI) | Experiment tracking | Current: 0.25.0 (Feb 13 2026). Log metrics, confusion matrices as wandb Tables, per-class accuracy plots, sample predictions. Mirrored from sibling. |
| matplotlib | `*` | Plotting confusion matrices, loss curves | Always. Use for offline plots saved as artifacts. |
| seaborn | `*` | Styled confusion matrix heatmaps | When rendering confusion matrices for reports. `sns.heatmap` on a numpy confusion matrix is the standard pattern. |
| pillow | `*` | Image IO (underlying ImageFolder dependency) | Required by torchvision; do not import directly — use torchvision transforms. |
| rich | `*` | Console progress + per-epoch summary tables | Always. Use `rich.table.Table` to print class-level accuracy summaries at end of validation epoch. |
| pandas | `*` | Per-class metric DataFrames, CSV logging | When saving per-epoch per-class metrics to disk for post-hoc analysis. |
| onnxruntime | `>=1.23.2,<2` (PyPI) | ONNX inference validation after export | Post-training export validation. Pin `>=1.23.2` — 1.24.1 has macOS platform tag issues with pixi solver (documented in sibling memory). |
| onnxscript | `>=0.5.6,<0.6` | ONNX graph construction and export utilities | Used by torch.onnx dynamo exporter under the hood; pin range matches sibling. |
| numpy | `<2.0.0` | Numerical ops | Pin `<2.0.0` — same constraint as sibling; torchvision and some metric libraries have numpy 2.x incompatibilities. |
| opencv (`opencv` via conda-forge) | `*` | Image reading/pre-processing in scripts | For any preprocessing scripts that need cv2. Use conda-forge `opencv`, NOT `opencv-python-headless` via PyPI (documented sibling issue). |

### Development Tools

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| ruff | `>=0.15,<1` (dev dep) | Linting + formatting | Current: 0.15.1 (Feb 12 2026). Replaces flake8, black, isort. Same rule set as sibling: `E, W, F, I, N, UP, B, SIM, S, A, C4, RUF`. `target-version = "py311"`. |
| mypy | `*` (dev dep) | Static type checking | Use `strict = true` with per-module overrides for torch/torchvision/hydra (none of these stub fully). Same `[[tool.mypy.overrides]]` pattern as sibling. |
| pytest | `*` (dev dep) | Test runner | `pixi run pytest`. Mirror sibling's coverage targets. |
| pytest-cov | `*` (dev dep) | Coverage reporting | `--cov=src --cov-report=term --cov-report=xml` |
| pre-commit | `*` (dev dep) | Git hook runner | Mirror sibling's `.pre-commit-config.yaml` hooks: ruff, mypy, trailing whitespace. |
| tensorboard | `>=2.20,<3` (dev dep) | Local loss visualization | Fallback for offline training. Lightning logs to tensorboard by default when wandb is not configured. |
| mkdocs + mkdocs-material + mkdocstrings | `*` (dev dep) | API documentation | Mirror sibling if docs are needed. Not critical for MVP. |
| pixi | `>=0.59` | Environment + task management | Current: 0.59.0. The only supported env manager per project rules. Use `pixi run` for all tasks. |
| flit_core | `>=3.11,<4` | Build backend | Mirror sibling's `[build-system]`. Lightweight, no build dependencies beyond flit itself. |
| semantic-release | configured in `pyproject.toml` | Automated versioning | Mirror sibling's `[tool.semantic_release]` config. Triggered via GitHub Actions on merge to `main`. |

### Infrastructure

| Technology | Purpose | Notes |
|------------|---------|-------|
| Docker (`nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`) | Training container | Mirror sibling Dockerfile exactly. pixi installs all deps inside container. `CONDA_OVERRIDE_CUDA=12.1` required for pixi solver to accept CUDA-dependent packages in CI. |
| GCP Cloud Build | CI/CD image build | Mirror sibling `cloudbuild.yaml`. `E2_HIGHCPU_8` machine, layer caching from Artifact Registry. 1800s timeout. |
| GCP Vertex AI (T4) | Training runtime | T4 is Turing (sm_75); fully supported by CUDA 12.1 and PyTorch 2.5+. |
| GitHub Actions | CI: lint, test, type-check | Run `pixi run test-cov`, `pixi run lint`, `pixi run typecheck` on push/PR. |
| Weights & Biases | Experiment tracking | `WANDB_API_KEY` injected at runtime via environment variable, never baked into image. |

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `torchvision.models.ResNet` + `weights=DEFAULT` | `timm` (pytorch-image-models) | Use timm when: (a) you need SOTA accuracy beyond ResNet (EfficientNet, ViT, ConvNeXt), (b) training 600 epochs with A3 recipe, or (c) non-standard input channel counts. For this project targeting ResNet18/ResNet50 with ImageNet weights in a controlled training setup, torchvision is sufficient and eliminates an extra dependency. |
| `torchvision.transforms.v2` | `albumentations` | Use albumentations when: preprocessing is CPU-bound and you need augmentations not in v2 (elastic deformation, CLAHE, optical distortion). For ImageNet-standard classification (RandomResizedCrop + HFlip + AutoAugment), v2 is sufficient and keeps the stack purely PyTorch. |
| `torchvision.transforms.v2` | `kornia` | Use kornia when doing augmentation on GPU tensors in a custom training loop. Not needed with Lightning's standard DataModule pattern. |
| `torchmetrics.MulticlassAccuracy` | `sklearn.metrics.accuracy_score` | NEVER use sklearn metrics inside `training_step`/`validation_step` — they are not DDP-safe and require `.cpu().numpy()` calls that break GPU training. sklearn is fine for post-hoc analysis in notebooks. |
| `wandb` | `MLflow` | Use MLflow only if the organization mandates an on-premise tracking server. wandb is simpler to set up and better for image/table logging. |
| `flit_core` | `hatchling`, `setuptools` | Either works; flit_core is already proven in the sibling. No reason to switch. |
| `loguru` | `structlog` or stdlib `logging` | structlog is better for JSON-structured logs in production services; loguru is sufficient for ML training scripts. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `opencv-python-headless` (PyPI) | Causes solver conflicts with pixi + conda-forge opencv; documented sibling issue | `opencv` via conda-forge |
| `onnxruntime==1.24.1` | macOS platform tag issue with pixi solver (documented sibling memory) | `onnxruntime>=1.23.2,<2` — solver will select compatible version |
| `numpy>=2.0` | Breaking changes in the C API affect torchvision and some metric dependencies | `numpy<2.0.0` |
| `torchvision.transforms` (v1) | Frozen, no new features, deprecated in favor of v2 | `torchvision.transforms.v2` |
| `timm` (as primary model source) | Adds a large dependency for a use case torchvision handles natively; harder to pin versions | `torchvision.models.resnet18/resnet50` with `weights=*.DEFAULT` |
| `torchvision.models.resnet50(pretrained=True)` | Deprecated API, raises deprecation warning in torchvision 0.13+ | `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` |
| Python 3.12+ | onnxruntime lacks stable wheel coverage for 3.12+ in pixi; see sibling memory | Python 3.11.* |
| uv / venv / conda directly | Project policy: pixi only. uv is fast but doesn't handle conda-forge CUDA packages. | `pixi` with `[system-requirements] cuda = "12.1"` |

---

## Stack Patterns by Variant

**If training on CPU only (local dev/testing):**
- Remove `[system-requirements] cuda = "12.1"` from `pixi.toml`
- Force `CPUExecutionProvider` in any onnxruntime calls (same pattern as basketball-2d-to-3d sibling)
- Lightning Trainer: `accelerator="cpu"`, `devices=1`

**If adding MixUp / CutMix augmentation:**
- Use `torchvision.transforms.v2.MixUp` and `CutMix` — these are natively supported in v2
- Apply at the batch level in `LightningDataModule.on_after_batch_transfer` or in `training_step`
- These require soft labels; update loss to `CrossEntropyLoss` with `label_smoothing`

**If adding a new backbone (EfficientNet, ViT):**
- Add `timm` as a PyPI dependency
- Use `timm.create_model(model_name, pretrained=True, num_classes=N)`
- Keep the same LightningModule interface; swap the backbone in `__init__`

**If exporting to ONNX:**
- Use `torch.onnx.export(..., dynamo=False)` (legacy exporter) for ResNet — simpler models don't require Dynamo
- Set `TORCH_ONNX_LEGACY_EXPORTER=1` as a safety measure (same pattern as sibling)
- Validate with `onnxruntime.InferenceSession` with `CPUExecutionProvider` in tests

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `pytorch==2.10.*` | `torchvision==0.25.*` | conda-forge resolves this automatically |
| `pytorch==2.5.*–2.10.*` | `lightning>=2.5,<3` | Lightning 2.6.1 is stable; both ranges safe |
| `pytorch>=2.5` | `torchmetrics>=1.0` | No known incompatibilities |
| `hydra-core==1.3.2` | `omegaconf==2.3.*` | Hydra 1.3 ships with omegaconf 2.3 as a dependency; do not pin omegaconf separately |
| `pydantic>=2.0` | Any Python 3.11 | v2 API is stable; no compat issues with Lightning or Hydra |
| `numpy<2.0.0` | `torchvision 0.25`, `opencv`, `torchmetrics 1.8` | Upper bound is a hard requirement |
| `onnxruntime>=1.23.2,<2` | `pytorch>=2.5` | Do NOT use 1.24.1 on macOS with pixi |
| `onnxscript>=0.5.6,<0.6` | `pytorch>=2.5` | Matches sibling; stable range for the torch.onnx dynamo path |

---

## pixi.toml Template

```toml
[workspace]
name = "classifier-training"
version = "0.1.0"
description = "ResNet image classification training framework"
authors = ["Enrique G. Ortiz <ortizeg@gmail.com>"]
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64"]

[system-requirements]
cuda = "12.1"

[tasks]
train       = { cmd = "train", env = { KMP_DUPLICATE_LIB_OK = "TRUE" } }
test        = "pytest"
test-cov    = "pytest --cov=src --cov-report=term --cov-report=xml -v"
precommit   = "pre-commit run --all-files"
format      = "ruff format ."
format-check = "ruff format --check ."
lint        = "ruff check ."
typecheck   = "mypy src/"
build       = { cmd = "bash scripts/cloud-build.sh" }
build-local = { cmd = "bash scripts/build-docker.sh --local" }

[dependencies]
python      = "3.11.*"
hydra-core  = "==1.3.2"
loguru      = ">=0.7,<1"
numpy       = "<2.0.0"
lightning   = ">=2.5,<3"
pydantic    = ">=2.0"
wandb       = ">=0.25,<1"  # PyPI below
pytorch     = "*"
torchvision = "*"
pandas      = "*"
matplotlib  = "*"
seaborn     = "*"
pillow      = "*"
opencv      = "*"
rich        = "*"
psutil      = "*"

[feature.dev.dependencies]
pytest       = "*"
pytest-cov   = "*"
tensorboard  = ">=2.20,<3"
ruff         = ">=0.15,<1"
mypy         = "*"
pre-commit   = "*"
mkdocs       = "*"
mkdocs-material = "*"

[environments]
default = ["dev"]
prod    = []

[pypi-dependencies]
torchmetrics         = { version = ">=1.0" }
onnxruntime          = { version = ">=1.23.2,<2" }
onnxscript           = { version = ">=0.5.6,<0.6" }
requests             = { version = "*" }
tqdm                 = { version = "*" }
wandb                = { version = ">=0.25,<1" }
mkdocstrings         = { version = "*", extras = ["python"] }
classifier-training  = { path = ".", editable = true }
```

---

## pyproject.toml Template

```toml
[build-system]
requires = ["flit_core >=3.11,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "classifier_training"
version = "0.1.0"
description = "ResNet image classification training framework"
authors = [{name = "Enrique G. Ortiz", email = "ortizeg@gmail.com"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.11"
dependencies = [
    "hydra-core==1.3.2",
    "loguru>=0.7,<1",
    "numpy<2.0.0",
    "lightning>=2.5,<3",
    "pydantic>=2.0",
    "torchmetrics>=1.0",
    "wandb>=0.25,<1",
    "torch",
    "torchvision",
    "pandas",
    "matplotlib",
    "seaborn",
    "Pillow",
    "opencv-python",
    "rich",
    "requests",
    "tqdm",
    "onnxruntime>=1.23.2,<2",
    "onnxscript>=0.5.6,<0.6",
]

[project.scripts]
train = "classifier_training.task_manager:main"

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "S", "A", "C4", "RUF"]
ignore = ["N812"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]
"scripts/**" = ["ALL"]

[tool.ruff.lint.isort]
known-first-party = ["classifier_training"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = [
    "cv2.*", "torchvision.*", "hydra.*", "omegaconf.*",
    "torch.*", "lightning.*", "tensorboard.*", "onnx.*",
    "onnxscript.*", "numpy.*", "wandb.*", "tqdm.*",
    "torchmetrics.*", "pandas.*", "matplotlib.*",
    "seaborn.*", "PIL.*", "rich.*",
]
ignore_missing_imports = true

[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "pip install build && python -m build"

[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

---

## Dockerfile Template

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/ortizeg/classifier-training"
LABEL org.opencontainers.image.license="Apache-2.0"

WORKDIR /app

RUN apt-get update && apt-get install -y curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"
ENV CONDA_OVERRIDE_CUDA=12.1

COPY pixi.toml pixi.lock* pyproject.toml ./

RUN mkdir -p src/classifier_training && touch src/classifier_training/__init__.py

RUN pixi install --environment prod

COPY . .

ENTRYPOINT ["pixi", "run", "train"]
```

---

## Sources

- [PyTorch Lightning 2.6.1 — PyPI](https://pypi.org/project/pytorch-lightning/) — Version confirmed HIGH confidence
- [TorchVision 0.25 Models Docs](https://docs.pytorch.org/vision/stable/models.html) — ResNet weight API verified HIGH confidence
- [TorchMetrics 1.8.2 — PyPI](https://pypi.org/project/torchmetrics/) — Version confirmed HIGH confidence
- [hydra-core 1.3.2 — PyPI](https://pypi.org/project/hydra-core/) — Version confirmed HIGH confidence
- [Pydantic 2.12.5 — PyPI](https://pypi.org/project/pydantic/) — Version confirmed HIGH confidence
- [loguru 0.7.3 — PyPI](https://pypi.org/project/loguru/) — Version confirmed HIGH confidence
- [ruff 0.15.1 — PyPI](https://pypi.org/project/ruff/) — Version confirmed HIGH confidence
- [wandb 0.25.0 — PyPI](https://pypi.org/project/wandb/) — Version confirmed HIGH confidence
- [PyTorch 2.10.0 (conda-forge)](https://anaconda.org/conda-forge/pytorch) — Current conda-forge version HIGH confidence
- [PyTorch get-started](https://pytorch.org/get-started/locally/) — Current stable 2.10.0, CUDA 12.6/12.8/13.0 HIGH confidence
- [torchvision.transforms v2](https://docs.pytorch.org/vision/stable/transforms.html) — v2 is current API HIGH confidence
- [nvidia/cuda:12.1.0-cudnn8 Docker Hub](https://hub.docker.com/r/nvidia/cuda) — Base image exists MEDIUM confidence (sibling proven)
- [sibling object-detection-training pixi.toml](../../../object-detection-training/pixi.toml) — Canonical reference for all shared tooling HIGH confidence (direct file read)
- timm vs torchvision discussion — WebSearch MEDIUM confidence (multiple sources agree)

---

## Notes on CUDA Pinning

The sibling uses `cuda = "12.1"` in `[system-requirements]`. PyTorch 2.10 on conda-forge
officially targets CUDA 12.6/12.8 for new installs, but GCP T4 VMs run whatever CUDA driver
is installed on the host (not the container), and the CUDA 12.1 runtime inside the container
is forward-compatible with driver >= 525. The `CONDA_OVERRIDE_CUDA=12.1` env var in Docker
tells pixi to accept packages built for CUDA 12.1 even if the build host has no GPU attached
(critical for Cloud Build). This pattern is proven working in the sibling.

If a future phase moves to newer hardware (A100, H100), bump the system-requirements to
`cuda = "12.6"` and repin `onnxruntime` accordingly.

---
*Stack research for: image classification training framework (ResNet18/ResNet50, PyTorch Lightning, pixi)*
*Researched: 2026-02-18*
