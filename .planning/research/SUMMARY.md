# Project Research Summary

**Project:** classifier-training
**Domain:** PyTorch Lightning image classification training framework (ResNet, ONNX export)
**Researched:** 2026-02-18
**Confidence:** HIGH

---

## Executive Summary

This project is a production-grade image classification training framework built as a deliberate sibling to the existing `object-detection-training` repo. The two repos share a common design language: Hydra YAML config composition, PyTorch Lightning for the training loop, Pydantic frozen models for inter-component contracts, loguru for logging, and pixi for environment management. The fundamental architecture decision is already made — mirror the detection sibling's structure but swap out its detection-specific components (COCO DataModule, mAP metrics, box-aware transforms, multi-loss function) with classification-equivalent ones (ImageFolder DataModule, accuracy/confusion matrix metrics, standard image transforms, CrossEntropyLoss). This dramatically reduces design risk: roughly 60% of the codebase can be copied or lightly adapted from the proven sibling.

The recommended approach is ResNet18/ResNet50 via torchvision as the backbone (pretrained ImageNet weights via the modern `weights=ResNet*Weights.DEFAULT` API), `torchvision.transforms.v2` for augmentations, `torchmetrics` for GPU-safe classification metrics, and a callback-driven extension architecture that keeps visualization and export logic out of the LightningModule. ONNX export is a first-class output — not an afterthought — because the trained model feeds directly into the `basketball-2d-to-3d` inference pipeline. Class imbalance (jersey number digit frequency skew) is the highest-probability domain risk and must be addressed in Phase 1 through class-weighted loss and WeightedRandomSampler, not left as a Phase 2 fix.

The top risks are: (1) `class_to_idx` alphabetical ordering not being persisted alongside the ONNX export — a silent inference-breaking bug; (2) augmentation leaking into validation splits via improper use of `random_split`; and (3) the EMA callback saving non-EMA weights into the "best model" checkpoint. All three are preventable by following patterns already established in the sibling repo, and all three must be addressed before Phase 1 is considered done.

---

## Key Findings

### Recommended Stack

The stack is almost entirely locked in by the sibling repo's proven choices. Python 3.11 (not 3.12+) is pinned to avoid onnxruntime wheel gaps. PyTorch resolves from conda-forge with `pytorch>=2.5,<2.11` (currently 2.10). torchvision pairs automatically via the conda-forge solver. PyTorch Lightning 2.6.1 provides `LightningModule`, `LightningDataModule`, and `Trainer`. Hydra 1.3.2 (pinned exactly) handles YAML config composition using the same config-group structure as the detection sibling. `torchmetrics>=1.0` provides all classification metrics as GPU-native stateful objects. The full pixi.toml and pyproject.toml templates are ready to use as-is from STACK.md.

Two classification-specific stack notes: `timm` is listed in FEATURES.md as a backbone source but STACK.md recommends `torchvision.models.resnet18/resnet50` with `weights=*.DEFAULT` to avoid adding a large optional dependency for a use case torchvision handles natively. If EfficientNet, ViT, or ConvNeXt are needed in a future phase, add `timm` then. MixUp/CutMix augmentation (a v1.x feature) requires `timm.data.Mixup` or the v2 native implementation — but this also requires switching from standard `CrossEntropyLoss` to a soft-label loss variant, making it an isolated opt-in feature rather than a default.

**Core technologies:**
- **Python 3.11.\***: Runtime — same as sibling; avoids onnxruntime 3.12+ wheel gaps
- **PyTorch >= 2.5, < 2.11** (conda-forge): Deep learning runtime — conda-forge solver manages torchvision pairing
- **torchvision \*** (solver-pinned): ResNet model definitions, pretrained weights, transforms.v2, ImageFolder — all classification needs in one package
- **lightning >= 2.5, < 3**: Training loop orchestration (LightningModule, LightningDataModule, Trainer, built-in callbacks)
- **hydra-core == 1.3.2**: Hierarchical YAML config — pinned exactly; identical config-group pattern as sibling
- **pydantic >= 2.0**: Frozen models for inter-component contracts (ClassificationBatch, ModelStats, EMAState)
- **torchmetrics >= 1.0**: GPU-native, DDP-safe classification metrics (MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassAUROC)
- **loguru >= 0.7, < 1**: Structured logging — replaces stdlib logging entirely
- **wandb >= 0.25, < 1**: Experiment tracking with image/table logging
- **onnxruntime >= 1.23.2, < 2** (PyPI): ONNX inference validation — do not use 1.24.1 (macOS pixi solver bug)
- **numpy < 2.0.0**: Hard upper bound — torchvision and torchmetrics have numpy 2.x incompatibilities
- **opencv** (conda-forge, not PyPI opencv-python-headless): Preprocessing scripts — pixi solver conflict with PyPI variant
- **pixi >= 0.59**: Only supported environment manager; all tasks via `pixi run`

**Do not use:** `timm` as primary backbone source, `torchvision.transforms` v1, `torch.onnx` with `pretrained=True` (deprecated), Python 3.12+, uv/venv/conda directly.

### Expected Features

The feature bar is set by the detection sibling. The classification-specific MVP adds a confusion matrix callback (new — no detection equivalent) and drops the detection-specific sampler distribution callback in favor of a classification-aware variant. All other sibling callbacks (EMA, ONNX export, model info, dataset statistics, training history plots, sample visualization) port directly with minor adaptations.

**Must have (P1 — table stakes for sibling parity):**
- ImageFolder DataModule with train/val/test splits — core data pipeline; everything depends on it
- ResNet18/50 via torchvision (pretrained ImageNet weights) — the model; configurable via Hydra
- CrossEntropyLoss with `class_weight` tensor + `label_smoothing=0.1` — handles imbalance and calibration in one loss
- Top-1 accuracy, Top-5 accuracy, per-class accuracy — primary metrics logged per step and epoch
- Confusion matrix callback (new) — classification-specific must-have; heatmap PNG saved per epoch
- Standard augmentation pipeline: train (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize) and val (Resize, CenterCrop, Normalize) — strict separation
- EMA callback — port from sibling; improves val accuracy; model-agnostic
- ONNX export callback — port from sibling; output consumed by basketball-2d-to-3d inference pipeline
- Model info callback — port from sibling; FLOPs/params/size + `labels_mapping.json`
- Dataset statistics callback — port from sibling; class distribution table at training start
- Training history plots callback — port from sibling; accuracy/loss curves replacing mAP plots
- Sampler distribution callback — port from sibling; validates class balancing is working
- WeightedRandomSampler — enables sampler distribution callback; critical for jersey number digit imbalance
- Cosine annealing LR with linear warmup — standard ResNet fine-tuning schedule
- Mixed precision (`16-mixed`), gradient clipping (`clip_val=1.0`), gradient accumulation — Lightning Trainer flags
- Early stopping + ModelCheckpoint + RichProgressBar + LRMonitor — Lightning built-ins
- WandB logging via WandbLogger
- Hydra YAML config with config groups: model, data, trainer, callbacks, transforms
- Reproducibility seed via `lightning.seed_everything`
- `labels_mapping.json` serialized alongside ONNX export — maps class indices to class names

**Should have (P2 — classification-specific value):**
- Per-class F1 score — better than accuracy for imbalanced classes
- AUROC — threshold-independent quality metric for production confidence calibration
- Prediction visualization callback — predicted vs true class overlaid on sample images
- RandAugment — strong augmentation for small datasets
- MixUp / CutMix — with soft-label loss; add when strong regularization is needed
- Step LR scheduler — as Hydra config alternative to cosine

**Defer (P3 / v2+):**
- GradCAM visualization callback — high value, high complexity; defer until model debugging is the bottleneck
- SAM (Sharpness Aware Minimization) optimizer — defer until accuracy plateau
- Multi-label classification support — requires separate model type (sigmoid + BCELoss pipeline)

### Architecture Approach

The architecture is a direct classification adaptation of the detection sibling. Five build layers exist in strict dependency order: (1) Foundation — `types.py`, `utils/hydra.py`, `tasks/base_task.py`, all copy-as-is from sibling; (2) Data Layer — `ImageFolderDataModule` replaces `COCODataModule`; folder-per-class structure eliminates JSON parsing, COCO ID remapping, and custom collate functions entirely; (3) Model Layer — `BaseClassificationModel(L.LightningModule)` centralizes metrics as class attributes so Lightning auto-moves them to the correct device; concrete `ResNetClassificationModel` registered via `@register` for Hydra config-driven backbone selection; (4) Callbacks Layer — EMA and model_info copy as-is; ONNX export adapts output names to `["logits"]`; statistics and visualization adapt from detection protocols; `ConfusionMatrixCallback` is the only net-new component; (5) Tasks and Entry Point — `task_manager.py` and `TrainTask` copy verbatim from sibling.

**Major components:**
1. **`task_manager.py`** — `@hydra.main` CLI entry point; wires model + datamodule + trainer + callbacks; copy from sibling
2. **`data/image_folder_data_module.py`** — `LightningDataModule` over `torchvision.datasets.ImageFolder`; exposes `num_classes` and `class_names` from sorted subdirectory list; no custom collate needed
3. **`models/base.py` + `models/resnet_lightning.py`** — `BaseClassificationModel` centralizes torchmetrics metric objects (automatically device-placed); ResNet18/50 variants registered with `@register` for Hydra
4. **`callbacks/confusion_matrix.py`** — only net-new callback; reads `pl_module._last_val_confusion` after `on_validation_epoch_end`; renders seaborn heatmap PNG and logs to WandB
5. **`callbacks/ema.py`** — copy from sibling verbatim; operates on `state_dict()` and is model-agnostic
6. **`callbacks/onnx_export.py`** — adapt from sibling; change output names from `["dets","labels"]` to `["logits"]`; must export from EMA state dict, not raw checkpoint
7. **`conf/`** — same Hydra config group structure as sibling: `task/`, `models/`, `data/`, `trainer/`, `callbacks/`, `transforms/`

**Key data flow:** `ImageFolder(disk) → train_transforms(v2.Compose) → DataLoader(default collate) → tuple[Tensor[B,3,H,W], Tensor[B]] → ResNetClassificationModel.training_step() → logits[B,num_classes] → F.cross_entropy() → optimizer.step() → EMACallback.on_train_batch_end()`

**Validation flow terminates at:** `on_validation_epoch_end()` stores `_last_val_confusion` on the module → `ConfusionMatrixCallback` reads it → renders PNG → WandB log.

### Critical Pitfalls

1. **`class_to_idx` not persisted to ONNX sidecar** — `ImageFolder` assigns indices alphabetically at load time; if `labels_mapping.json` is not written alongside the ONNX export, inference consumers reconstruct the wrong mapping silently. Especially dangerous for jersey number digits where string sort ("10" before "2") diverges from numeric order. Prevent by serializing `dataset.class_to_idx` in `ModelInfoCallback` or `ONNXExportCallback` from day one of Phase 1.

2. **Augmentation leaking into validation via `random_split`** — splitting an augmented `ImageFolder` means all subsets inherit train transforms; validation accuracy appears inflated and varies epoch to epoch. Prevent by always constructing separate `ImageFolder` instances for train, val, and test with distinct `v2.Compose` pipelines.

3. **EMA checkpoint saves non-EMA weights** — `ModelCheckpoint` fires from `state_dict()` during `on_validation_epoch_end`; if the EMA swap/restore cycle and checkpoint timing are not explicitly coordinated, the saved "best" checkpoint holds live weights while the metric that triggered the save was computed with EMA weights. Prevent by exporting ONNX directly from `EMACallback.ema_state_dict` in `on_train_end`, never from a loaded `.ckpt` file.

4. **TorchMetrics mixed step/epoch logging produces 0.0 or NaN** — calling `self.log("val/acc", self.val_acc)` in `validation_step` AND `self.log("val/acc", self.val_acc.compute())` in `on_validation_epoch_end` causes Lightning to reset the metric between calls. Pick one pattern (Pattern A: `.update()` in step, `.compute()` + `.reset()` in epoch end) and enforce it consistently across all metrics.

5. **Confusion matrix device mismatch crashes GPU training** — metrics declared in callbacks (not inside `LightningModule.__init__`) are not auto-moved by Lightning. Declaring `MulticlassConfusionMatrix` inside the `LightningModule` and storing the computed result as `self._last_val_confusion` (a plain numpy array) before the callback reads it is the correct pattern. This avoids any device mismatch entirely.

**Moderate pitfalls to flag:**
- `model.eval()` must be called before `torch.onnx.export()` or BatchNorm uses batch statistics at inference
- Jersey number class imbalance destroys rare-digit recognition — class-weighted loss + WeightedRandomSampler are mandatory from Phase 1, not optional
- Fine-tuning ResNet with `lr=1e-3` destroys pretrained features — default should be `lr=1e-4` in model YAML configs
- FP16 GradScaler collapse on T4 with small batches — monitor scale factor; prefer `bf16-mixed` if instability occurs
- Hydra timestamped output dirs cause checkpoint resume to start a fresh directory — set `ModelCheckpoint.dirpath` to a fixed path independent of Hydra output dir

---

## Implications for Roadmap

Based on the architectural dependency order established in ARCHITECTURE.md and the pitfall-to-phase mapping in PITFALLS.md, five phases are recommended. The component build order is a hard constraint: types and utilities have no internal imports; the data layer depends on types; the model layer depends on types and torchmetrics; callbacks depend on model and data; the entry point depends on everything. This ordering cannot be reversed without creating circular imports.

### Phase 1: Foundation and Data Pipeline

**Rationale:** Everything else has an import dependency on this layer. Class imbalance handling (class-weighted loss weights, WeightedRandomSampler) must be computed from the data — making it a data-pipeline decision, not a training-loop decision. Two of the five critical pitfalls (augmentation leaking into val, `class_to_idx` not persisted) are data module bugs that corrupt all downstream work if introduced here and discovered late.

**Delivers:** Working `ImageFolderDataModule` with train/val/test splits, strictly separated transform pipelines (train with augmentation vs. val with resize+normalize only), `class_to_idx` serialized to `labels_mapping.json`, class weight tensor computed from training split, `WeightedRandomSampler` wired in, `DatasetStatisticsCallback` printing class distribution at epoch 0, pixi project scaffold (pixi.toml, pyproject.toml, src layout, CI workflow stubs).

**Addresses:** ImageFolder DataModule, standard augmentation pipeline (train + val), WeightedRandomSampler, Dataset statistics callback, `labels_mapping.json` serialization, reproducibility seed.

**Avoids:** Pitfall 1 (class_to_idx), Pitfall 2 (augmentation in val splits), Pitfall 7 (class imbalance ignored until too late).

**Research flag:** Standard pattern — no additional research needed. `LightningDataModule` over `ImageFolder` is thoroughly documented.

### Phase 2: Model Layer

**Rationale:** Model depends on the data module for `num_classes` injection. The base model's metric logging pattern (Pattern A: `.update()` / `.compute()` / `.reset()`) must be established here and enforced from the start to avoid the TorchMetrics mixed logging pitfall. ONNX normalization strategy (bake into model graph vs. document in sidecar) must be decided before training proceeds.

**Delivers:** `BaseClassificationModel` with all torchmetrics objects declared in `__init__` (auto-device placement), consistent Pattern A logging across `training_step`/`validation_step`/`on_*_epoch_end`, `ResNet18ClassificationModel` and `ResNet50ClassificationModel` registered via `@register`, Hydra config YAMLs for both variants with `lr=1e-4` and `warmup_epochs=2` as tuned defaults, AdamW + CosineAnnealingLR + LinearLR warmup configured.

**Addresses:** ResNet backbone (pretrained), CrossEntropyLoss with class_weight + label_smoothing, Top-1/Top-5/per-class accuracy metrics, cosine LR + warmup, mixed precision + gradient clipping + gradient accumulation flags, Hydra model configs with tuned LR defaults.

**Avoids:** Pitfall 5 (TorchMetrics mixed logging), Pitfall 8 (transfer learning LR too high), Pitfall 3 (ImageNet normalization mismatch — normalization strategy finalized here).

**Research flag:** Standard pattern — LightningModule + torchmetrics integration is official-docs-verified.

### Phase 3: Callbacks and ONNX Export

**Rationale:** Callbacks depend on both the model and data layers being stable. The EMA + ModelCheckpoint + ONNXExport interaction (Pitfall 4) must be solved and tested as a unit — splitting them across phases risks discovering the integration bug only after a long training run. The confusion matrix callback (the only net-new component) has a device mismatch risk (Pitfall 11) that must be tested against a GPU device.

**Delivers:** EMA callback (copied from sibling), ONNX export callback (adapted to export from `ema_state_dict`, output name `["logits"]`, `model.eval()` guard, dynamic batch axis), `ConfusionMatrixCallback` (new), model info callback (copied), training history plots callback (adapted for accuracy keys), sample visualization callback (adapted — no boxes), sampler distribution callback (adapted), label mapping callback (copied). All callbacks wired into `conf/callbacks/default.yaml`. Integration test: ONNX model accuracy == best val accuracy from training log.

**Addresses:** EMA callback, ONNX export callback, ConfusionMatrixCallback (new), model info callback, training history plots, visualization, sampler distribution.

**Avoids:** Pitfall 4 (EMA weight mismatch in ONNX), Pitfall 6 (BatchNorm eval mode in ONNX), Pitfall 11 (confusion matrix device mismatch).

**Research flag:** The EMA + ModelCheckpoint timing issue (Lightning #11276) may need specific attention during plan-phase. Otherwise callbacks follow established sibling patterns.

### Phase 4: Training Configuration and Infrastructure

**Rationale:** Trainer config, Docker image, CI/CD, and Hydra run configuration are independent of the model architecture but depend on all code being in place. Checkpoint resume behavior (Pitfall 10) can only be tested once checkpointing is wired up end-to-end. AMP stability on T4 (Pitfall 9) is verified here with a real training run.

**Delivers:** `conf/trainer/default.yaml` with T4-tuned defaults (batch_size=64, num_workers=4, persistent_workers=True, pin_memory=True, precision=16-mixed, gradient_clip_val=1.0), `conf/train_basketball_resnet18.yaml` dataset-specific override, Dockerfile (mirror sibling's nvidia/cuda:12.1 base with pixi inside), `cloudbuild.yaml` for GCP Cloud Build, GitHub Actions CI (lint + test + typecheck on push/PR), WandB logger integration, ModelCheckpoint with fixed `dirpath` for resume stability.

**Addresses:** Mixed precision + gradient clipping (Trainer flags), WandB logging, checkpoint resume, Docker + GCP Cloud Build, CI/CD.

**Avoids:** Pitfall 9 (FP16 GradScaler collapse — verified on T4), Pitfall 10 (Hydra output dir collision on resume).

**Research flag:** Docker + GCP Cloud Build follows the sibling exactly — no research needed. AMP stability on T4 with classification may need empirical validation during execution.

### Phase 5: Validation, Testing, and End-to-End Verification

**Rationale:** The framework is only done when a complete training run produces an ONNX model that the `basketball-2d-to-3d` inference pipeline can consume correctly. P2 features (per-class F1, AUROC, prediction visualization) can be added incrementally once the core loop is validated. All "looks done but isn't" checks from PITFALLS.md are gating criteria here.

**Delivers:** Full test suite (unit tests per component, integration test for ONNX export vs Lightning accuracy, CI green across all pixi tasks), completed "looks done but isn't" checklist from PITFALLS.md, P2 features added (per-class F1, AUROC, prediction visualization callback, RandAugment as Hydra-configurable option), end-to-end training run on basketball jersey number dataset, ONNX consumed by basketball-2d-to-3d verified.

**Addresses:** Per-class F1, AUROC, prediction visualization callback, RandAugment, Step LR scheduler (as Hydra alternative).

**Avoids:** All remaining pitfalls via the "looks done but isn't" checklist.

**Research flag:** P2 augmentation features (MixUp/CutMix with soft-label loss) may need a targeted research pass — the soft-label loss swap has interaction effects with the existing CrossEntropyLoss configuration.

### Phase Ordering Rationale

- Foundation → Data → Model → Callbacks → Config/Infra → Validation follows the import dependency graph exactly, preventing circular import issues
- Class imbalance tooling (WeightedRandomSampler, class-weighted loss weights) lands in Phase 1 not Phase 2, because the weights are computed from the data pipeline, not the model
- Callbacks are deferred until Phase 3 because the EMA + ModelCheckpoint + ONNX export integration requires stable model and data contracts to test meaningfully
- Infrastructure (Docker, CI, Cloud Build) is Phase 4 because it wraps the completed codebase — building it earlier creates maintenance churn as interfaces evolve
- P2 features land in Phase 5 after core loop validation — this prevents scope creep from delaying the critical path to a working ONNX export

### Research Flags

Phases needing deeper research during plan-phase:
- **Phase 3:** EMA + ModelCheckpoint timing interaction (Lightning issue #11276). The sibling's pattern is documented but the exact hook ordering should be traced before planning the EMA/checkpoint/ONNX integration test.
- **Phase 5 (MixUp/CutMix):** Soft-label loss swap has interaction effects — the plan should explicitly cover how to switch loss classes without breaking the standard CrossEntropyLoss path.

Phases with well-documented standard patterns (skip research-phase, go straight to plan-phase):
- **Phase 1:** ImageFolder + LightningDataModule is thoroughly documented in official Lightning docs
- **Phase 2:** LightningModule + torchmetrics metric pattern is official-docs-verified
- **Phase 4:** Docker + GCP Cloud Build mirrors sibling exactly — no unknowns

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified against PyPI, conda-forge, and sibling repo pixi.toml (direct file read). Complete pixi.toml and pyproject.toml templates ready. |
| Features | HIGH | Sibling repo callbacks directly inspected. torchmetrics official docs verified for all classification metrics. Feature dependency graph is complete. |
| Architecture | HIGH | Sibling repo codebase directly inspected. Component-to-component boundaries, data flow, and anti-patterns derived from working production code. |
| Pitfalls | HIGH | Top pitfalls sourced from official Lightning GitHub issues, PyTorch forums, and torchmetrics issues tracker. Jersey number imbalance validated against published sports CV research. |

**Overall confidence:** HIGH

### Gaps to Address

- **CUDA/AMP stability on T4 with classification tasks:** The sibling uses `precision="16-mixed"` with `gradient_clip_val=5.0`. For classification (smaller loss scale), `clip_val=1.0` and potentially `bf16-mixed` may be preferable. Empirical validation needed in Phase 4 — plan for a quick smoke training run on T4 before declaring Phase 4 complete.

- **Jersey number dataset class distribution:** Pitfall 7 references research showing 24% of jersey numbers have <100 samples in typical sports datasets. The actual dataset for this project has not been characterized. Phase 1 must include the `DatasetStatisticsCallback` as a blocking deliverable so the true distribution is known before hyperparameter decisions are made.

- **Normalization baked into ONNX graph vs. sidecar:** Pitfall 3 identifies two valid approaches (prepend `nn.Module` normalize layer vs. document in `labels_mapping.json`). The plan for Phase 2 should make this an explicit decision with a clear recommendation before implementation starts. Recommend documenting in sidecar JSON for simplicity; the inference pipeline in basketball-2d-to-3d already handles preprocessing.

- **timm vs. torchvision as backbone source:** FEATURES.md lists timm as the backbone source while STACK.md recommends torchvision. The plan should resolve this explicitly. Recommendation: start with `torchvision.models.resnet18/resnet50(weights=*.DEFAULT)` for Phase 1-3; add `timm` as an opt-in Hydra config alternative in Phase 5 if non-ResNet backbones are needed.

---

## Sources

### Primary (HIGH confidence)

- Sibling repo direct code inspection: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/` — all callback, model, and data patterns
- [PyTorch Lightning 2.6.1 — PyPI](https://pypi.org/project/pytorch-lightning/) — version and API verified
- [TorchVision 0.25 Models Docs](https://docs.pytorch.org/vision/stable/models.html) — ResNet weight API (`ResNet*Weights.DEFAULT`)
- [TorchMetrics 1.8.2 — PyPI + official docs](https://lightning.ai/docs/torchmetrics/stable/) — all classification metrics (`MulticlassAccuracy`, `MulticlassF1Score`, `MulticlassConfusionMatrix`, `MulticlassAUROC`)
- [TorchMetrics in Lightning (official)](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html) — metric device placement and logging patterns
- [hydra-core 1.3.2 — PyPI](https://pypi.org/project/hydra-core/) — API stable, version confirmed
- [PyTorch get-started](https://pytorch.org/get-started/locally/) — PyTorch 2.10.0 current stable
- [torchvision.transforms v2 docs](https://docs.pytorch.org/vision/stable/transforms.html) — v2 is current API
- [PyTorch Lightning LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) — DataModule pattern
- [PyTorch Lightning EarlyStopping + ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.html) — built-in callbacks
- [timm training scripts](https://timm.fast.ai/training_scripts) — MixUp/CutMix, augmentation baselines
- [ResNet Strikes Back paper](https://arxiv.org/pdf/2110.00476) — training procedure (cosine LR, label smoothing, EMA)

### Secondary (MEDIUM confidence)

- [PyTorch Lightning EMA discussion #11276](https://github.com/Lightning-AI/pytorch-lightning/discussions/11276) — checkpoint timing issue
- [TorchMetrics mixed logging bug #597](https://github.com/Lightning-AI/torchmetrics/issues/597) — step/epoch logging conflict
- [ImageFolder augmentation on val splits — PyTorch Forums](https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580) — documented community pitfall
- [PyTorch ONNX export BatchNorm issue #75252](https://github.com/pytorch/pytorch/issues/75252) — training mode during export
- [Jersey number recognition — PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9583843/) — class imbalance statistics
- [MIC-DKFZ image_classification framework](https://github.com/MIC-DKFZ/image_classification) — competitor feature analysis
- [nvidia/cuda:12.1.0-cudnn8 Docker Hub](https://hub.docker.com/r/nvidia/cuda) — base image (sibling-proven)

### Tertiary (LOW confidence)

- [Label Smoothing guide 2025 — shadecoder](https://www.shadecoder.com/topics/label-smoothing-a-comprehensive-guide-for-2025) — single source; standard parameter `0.1` well-established elsewhere
- [Jersey number recognition CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf) — soccer domain, may not directly apply to basketball

---

*Research completed: 2026-02-18*
*Ready for roadmap: yes*
