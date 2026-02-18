# Feature Research

**Domain:** PyTorch Lightning Image Classification Training Framework
**Researched:** 2026-02-18
**Confidence:** HIGH (torchmetrics/Lightning official docs verified; sibling repo directly inspected)

---

## Context: Sibling Repo Baseline

The object-detection-training sibling repo defines the feature bar for this project.
These features were confirmed by direct code inspection and must be matched or exceeded:

| Sibling Feature | Implementation | Notes |
|----------------|----------------|-------|
| EMA callback | `callbacks/ema.py` — decay + warmup_steps, swap on val/test | Port directly, classification-generic |
| ONNX export callback | `callbacks/onnx_export.py` — best/final/all, opset, simplify | Simpler for classification (no dynamic axes needed) |
| Model info callback | `callbacks/model_info.py` — FLOPs, params, size, FPS, labels_mapping.json | Port with classification head dimensions |
| Dataset statistics callback | `callbacks/statistics.py` — per-split counts, class distribution rich table | Port; classification is simpler (no bbox stats) |
| Training history plots | `callbacks/plotting.py` — loss/metric curves saved to disk per epoch | Port; replace mAP with accuracy/F1 |
| Sample visualization | `callbacks/visualization.py` — GT + predictions, WandB logging | Replace detection boxes with classification overlays |
| Sampler distribution | `callbacks/sampler_distribution.py` — baseline vs effective class dist per epoch | Port directly; critical for imbalanced datasets |
| Mixed precision | Trainer `precision="16-mixed"` flag | Built into Lightning Trainer |
| Gradient clipping | Trainer `gradient_clip_val` flag | Built into Lightning Trainer |
| Gradient accumulation | Trainer `accumulate_grad_batches` flag | Built into Lightning Trainer |
| Rich progress bar | `RichProgressBar` callback | Built into Lightning |
| LR monitor | `LearningRateMonitor` callback | Built into Lightning |
| Early stopping | `EarlyStopping` callback | Built into Lightning |
| Model checkpoint | `ModelCheckpoint` callback | Built into Lightning |
| Hydra config | YAML composition via Hydra | Port config structure |
| WandB logging | `WandbLogger` | Same integration |
| Label mapping JSON | Exported at training start | Required for inference |

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must exist or the framework feels broken compared to the sibling repo or community standard.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **ImageFolder-style DataModule** | Standard PyTorch convention: `train/class_a/`, `val/class_a/`, `test/class_a/` | LOW | Use `torchvision.datasets.ImageFolder`; Lightning `LightningDataModule` wrapper |
| **ResNet backbone via timm** | timm has 270+ pretrained weights; `timm.create_model('resnet50', pretrained=True)` is the community default | LOW | `timm` with `num_classes` override replaces final FC; avoids writing backbone code |
| **Cross-entropy loss** | Default classification loss; every training tutorial uses it | LOW | `torch.nn.CrossEntropyLoss`; supports `weight=` for class imbalance |
| **Top-1 accuracy** | Primary classification metric; expected in every training output | LOW | `torchmetrics.MulticlassAccuracy` |
| **Top-5 accuracy** | Expected for datasets with >10 classes; standard ImageNet convention | LOW | `torchmetrics.MulticlassAccuracy(top_k=5)` |
| **Per-class accuracy** | Required to detect per-class failure modes; `average=None` in torchmetrics | LOW | `MulticlassAccuracy(average=None)` logged per class |
| **Confusion matrix** | Classification-specific must-have; missing = users cannot diagnose errors | MEDIUM | `torchmetrics.MulticlassConfusionMatrix`; plot as heatmap via seaborn/matplotlib; save to disk |
| **Standard augmentations** | RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize are expected by every user | LOW | `torchvision.transforms.v2` (v2 API preferred for consistency); ImageNet mean/std default |
| **Val/test augmentations** | Resize + CenterCrop + Normalize only — deterministic; users expect this separation | LOW | Separate transform pipelines for train vs val/test |
| **Model checkpoint (best + last)** | Resume training, export best model; table stakes | LOW | `lightning.pytorch.callbacks.ModelCheckpoint` |
| **Early stopping** | Prevents wasted compute on overfit runs | LOW | `lightning.pytorch.callbacks.EarlyStopping` |
| **Rich progress bar** | Parity with sibling repo; CLI usability | LOW | `lightning.pytorch.callbacks.RichProgressBar` |
| **LR monitor** | See learning rate changes during training | LOW | `lightning.pytorch.callbacks.LearningRateMonitor` |
| **Mixed precision training** | 2x+ speedup on modern GPUs; expected in any serious training framework | LOW | `Trainer(precision="16-mixed")` — Lightning flag |
| **Gradient clipping** | Prevents gradient explosion especially with label smoothing + augmentations | LOW | `Trainer(gradient_clip_val=1.0)` — Lightning flag |
| **WandB logging** | Parity with sibling repo; remote experiment tracking | LOW | `lightning.pytorch.loggers.WandbLogger` |
| **Label mapping JSON** | Maps integer class IDs to class names; required for ONNX inference integration | LOW | Exported at `on_fit_start`; follows sibling repo pattern |
| **ONNX export** | Sibling repo has it; inference pipeline uses ONNX models | MEDIUM | Simpler than detection: no dynamic spatial axes needed; `(batch, 3, H, W) -> (batch, num_classes)` |
| **Training history plots** | Sibling repo has it; visual confirmation training is progressing | LOW | Replace mAP plots with accuracy/F1 plots; save loss_history.png, accuracy_history.png |
| **EMA callback** | Sibling repo has it; improves validation accuracy by 0.5-1% on classification | MEDIUM | Port from sibling repo; classification-generic (no detection-specific types) |
| **Cosine annealing LR scheduler** | Standard for ResNet training; "ResNet Strikes Back" paper uses it | LOW | `torch.optim.lr_scheduler.CosineAnnealingLR` with optional warmup |
| **Hydra configuration** | Parity with sibling repo; YAML composition for experiments | MEDIUM | Config groups: model, data, training, callbacks |
| **Reproducibility seed** | Expected in any research-grade training framework | LOW | `lightning.seed_everything(seed)` |

### Differentiators (Competitive Advantage)

Features beyond the sibling repo baseline that add real value for classification-specific workflows.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Label smoothing** | Improves calibration; reduces overconfident predictions; standard for ResNet fine-tuning | LOW | `CrossEntropyLoss(label_smoothing=0.1)` — single parameter; HIGH value, LOW cost |
| **Class-weighted loss** | Essential for imbalanced datasets (common in sports domain classification) | LOW | `CrossEntropyLoss(weight=class_weights_tensor)`; weights computed from class frequencies in training set |
| **WeightedRandomSampler** | Upsamples rare classes at the batch level; complementary to class-weighted loss | MEDIUM | Port `TrackingWeightedRandomSampler` from sibling repo; enables sampler distribution callback |
| **Sampler distribution callback** | Validates that oversampling is working correctly; unique to this codebase | MEDIUM | Port from sibling repo with classification-specific stats |
| **Confusion matrix visualization** | Heatmap saved to disk and logged to WandB; classification-specific, absent in detection sibling | MEDIUM | `seaborn.heatmap` with class names; log at validation end; trigger via callback |
| **Per-class F1 score** | Better than accuracy for imbalanced classes; macro/micro/per-class variants | LOW | `torchmetrics.MulticlassF1Score`; log all averaging modes |
| **AUROC per-class** | Threshold-independent quality metric; useful for production confidence thresholds | MEDIUM | `torchmetrics.MulticlassAUROC`; expensive for many classes but valuable |
| **Dataset statistics callback** | Count images/classes per split; detect imbalance before training starts | LOW | Port from sibling repo; simpler without bbox statistics; output rich table |
| **Model info callback** | FLOPs, params, size, FPS; essential for deployment decisions | MEDIUM | Port from sibling repo; use `fvcore` for FLOPs with `(1, 3, H, W)` input |
| **MixUp augmentation** | Improves generalization; used in "ResNet Strikes Back" training procedure | MEDIUM | `timm.data.Mixup`; requires soft target loss; switch to BCE per-class or soft-label CE |
| **CutMix augmentation** | Complementary to MixUp; timm applies one per batch randomly | MEDIUM | `timm.data.Mixup(mixup_alpha=0, cutmix_alpha=1.0)`; same soft-label requirement as MixUp |
| **RandAugment** | Strong augmentation policy; significant accuracy gains on small datasets | LOW | `torchvision.transforms.v2.RandAugment`; configurable num_ops, magnitude |
| **Step LR scheduler** | Alternative to cosine; preferred for fine-tuning with fixed schedule | LOW | `torch.optim.lr_scheduler.MultiStepLR`; Hydra-configurable scheduler choice |
| **GradCAM visualization callback** | Explains which image regions drove predictions; differentiating for debugging | HIGH | `pytorch-grad-cam` library; fire on test epoch end; optional/off by default |
| **Gradient accumulation** | Enables larger effective batch sizes on single GPU; parity with sibling | LOW | `Trainer(accumulate_grad_batches=N)` — Lightning flag |
| **Prediction visualization callback** | Show sample images with predicted class overlaid; analog to detection visualization | MEDIUM | Classification-specific: overlay predicted label + confidence + ground truth label on image |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create unjustified complexity for this project.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Multi-label classification support** | Some datasets have multiple labels per image | Requires sigmoid + BCELoss instead of softmax + CE; breaks metric pipeline; separate task | Build single-label first; add multi-label as separate model type if needed |
| **Knowledge distillation** | Compress large model into smaller one | Significant architecture complexity; teacher model management; out of scope for training framework | Deploy smaller timm model directly with pretrained weights |
| **Neural architecture search (NAS)** | Automatically find best architecture | Enormous compute cost; framework complexity; not needed when timm has 270+ ready architectures | Use timm model sweeps instead |
| **Custom backbone from scratch** | Sometimes users want novel architectures | Goes against "use timm" principle; maintenance burden; pretrained weights are the value | Accept timm model names as config parameter; timm supports custom registrations |
| **Segmentation or detection heads** | Unified vision framework seems appealing | Completely different data pipeline and loss functions; would undermine the focused classification design | Those tasks have the sibling detection repo |
| **Real-time inference server** | Serve predictions via HTTP | Out of scope for training repo; training output is ONNX; serving belongs in inference pipeline (basketball-2d-to-3d) | Export ONNX; use inference pipeline from sibling project |
| **Automatic hyperparameter tuning** | Optuna/Ray integration | Framework complexity explosion; hyperparameter search is a separate concern | Run Hydra multirun sweeps manually |
| **TorchScript export** | Alternative to ONNX | ONNX is the established format for the existing inference pipeline; TorchScript adds maintenance with no benefit | ONNX only; inference pipeline already uses ONNX Runtime |
| **Federated learning** | Privacy-preserving training | Massive complexity; not a project requirement | Standard centralized training |

---

## Feature Dependencies

```
[ImageFolder DataModule]
    └──requires──> [Standard augmentation transforms]
                       └──requires──> [Normalize with ImageNet stats]

[ResNet via timm]
    └──requires──> [ImageFolder DataModule]
    └──enhances──> [EMA callback] (EMA works on any LightningModule)
    └──enables──> [ONNX export] (timm models export cleanly)
    └──enables──> [GradCAM] (requires conv layer reference)

[Cross-entropy loss]
    └──enhances──> [Label smoothing] (single parameter on CE)
    └──conflicts──> [MixUp/CutMix augmentations] (need soft-label loss variant)

[MixUp/CutMix augmentations]
    └──requires──> [Soft-label loss] (BCE per-class or soft CE from timm)
    └──conflicts──> [Standard CrossEntropyLoss] (hard labels incompatible with mixed targets)

[Class-weighted loss]
    └──requires──> [Dataset statistics] (to compute per-class frequencies)
    └──enhances──> [WeightedRandomSampler] (both address imbalance; complementary)

[WeightedRandomSampler]
    └──enables──> [Sampler distribution callback] (callback reads sampler._last_indices)

[Confusion matrix visualization]
    └──requires──> [Per-class metrics] (accumulated per-step, computed per-epoch)
    └──enhances──> [WandB logging] (log heatmap as image)

[Per-class F1 score]
    └──requires──> [torchmetrics MulticlassF1Score]
    └──enhances──> [Confusion matrix] (together give full error diagnosis picture)

[AUROC]
    └──requires──> [Softmax probabilities] (needs probability outputs, not argmax)
    └──conflicts──> [Large num_classes] (expensive to compute; disable if >50 classes)

[Model info callback]
    └──requires──> [fvcore] (FLOPs computation)
    └──requires──> [Label mapping JSON generation]

[EMA callback]
    └──enhances──> [Model checkpoint] (save EMA weights as separate checkpoint)
    └──requires──> [Training loop start] (initialized on_fit_start)

[ONNX export callback]
    └──requires──> [Model checkpoint] (loads best checkpoint for export)
    └──requires──> [export_onnx() method on LightningModule]

[GradCAM visualization]
    └──requires──> [Target layer name] (e.g., `layer4` on ResNet)
    └──requires──> [pytorch-grad-cam library]
    └──enhances──> [Prediction visualization callback]

[Hydra config]
    └──enables──> [All callback configuration] (on/off per experiment)
    └──enables──> [LR scheduler choice] (cosine vs step via config group)
```

### Dependency Notes

- **MixUp/CutMix conflicts with standard CrossEntropyLoss:** When MixUp is active, targets become soft distributions. Must use timm's `SoftTargetCrossEntropy` or `LabelSmoothingCrossEntropy`. These cannot be naively combined with standard CE. Implement MixUp as opt-in with separate loss class.
- **AUROC requires probabilities:** Must call `torch.softmax(logits, dim=1)` before passing to AUROC metric. If model outputs logits directly (standard), this must be done in the metric step.
- **WeightedRandomSampler enables SamplerDistributionCallback:** The sibling repo's `TrackingWeightedRandomSampler` stores `_last_indices` which the callback reads. Both must be present together or neither works.
- **GradCAM requires conv layer reference:** ResNet's `model.layer4` must be passed to GradCAM. This makes GradCAM architecture-aware, adding coupling. Keep it optional via config flag.

---

## MVP Definition

### Launch With (v1)

Minimum viable product — feature parity with sibling repo for the classification domain.

- [ ] **ImageFolder DataModule with train/val/test splits** — core data pipeline; everything else depends on it
- [ ] **ResNet backbone via timm (pretrained)** — the model; ResNet50 default, configurable via Hydra
- [ ] **CrossEntropyLoss with class_weight and label_smoothing** — covers both imbalance and calibration in one loss
- [ ] **Top-1 accuracy + Top-5 accuracy** — primary metrics; logged per step and epoch
- [ ] **Per-class accuracy** — required to detect per-class failure; logged at epoch end
- [ ] **Confusion matrix callback** — classification-specific must-have; saved as heatmap PNG
- [ ] **Standard augmentation pipeline (train + val)** — RandomResizedCrop/CenterCrop + flips + ColorJitter + Normalize
- [ ] **EMA callback** — port from sibling repo; improves val accuracy
- [ ] **ONNX export callback** — port from sibling repo; output is what the inference pipeline consumes
- [ ] **Model info callback** — port from sibling repo; FLOPs/params/size/labels_mapping.json
- [ ] **Dataset statistics callback** — port from sibling repo; class count table per split
- [ ] **Training history plots callback** — loss + accuracy curves saved per epoch
- [ ] **Sampler distribution callback** — port from sibling repo; validates class balancing
- [ ] **WeightedRandomSampler** — required for sampler distribution callback; handles imbalanced datasets
- [ ] **Cosine annealing LR with warmup** — standard ResNet training schedule
- [ ] **Mixed precision, gradient clipping, gradient accumulation** — Lightning Trainer flags
- [ ] **Early stopping + ModelCheckpoint + RichProgressBar + LRMonitor** — Lightning built-ins
- [ ] **WandB logging** — parity with sibling repo
- [ ] **Hydra YAML config** — parity with sibling repo; config groups: model, data, training, callbacks
- [ ] **Reproducibility seed** — `lightning.seed_everything`

### Add After Validation (v1.x)

Features to add once core training loop is validated working.

- [ ] **Per-class F1 score** — add when per-class accuracy proves insufficient for diagnosis
- [ ] **AUROC** — add when production confidence thresholds are needed
- [ ] **Prediction visualization callback** — add when debugging model errors qualitatively
- [ ] **RandAugment** — add when baseline augmentations plateau
- [ ] **MixUp / CutMix** — add with soft-label loss; add when strong regularization needed
- [ ] **Step LR scheduler** — add as Hydra config alternative to cosine

### Future Consideration (v2+)

Features to defer until project-market fit is established.

- [ ] **GradCAM visualization callback** — high complexity, high value; add when model debugging becomes the bottleneck
- [ ] **Sharpness Aware Minimization (SAM) optimizer** — DKFZ uses it; adds complexity; defer until accuracy plateau

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| ImageFolder DataModule | HIGH | LOW | P1 |
| ResNet via timm | HIGH | LOW | P1 |
| CrossEntropyLoss (weight + smoothing) | HIGH | LOW | P1 |
| Top-1 / Top-5 / Per-class accuracy | HIGH | LOW | P1 |
| Confusion matrix callback | HIGH | MEDIUM | P1 |
| EMA callback (port) | HIGH | LOW | P1 |
| ONNX export callback (port) | HIGH | MEDIUM | P1 |
| Model info callback (port) | MEDIUM | LOW | P1 |
| Dataset statistics callback (port) | MEDIUM | LOW | P1 |
| Training history plots (port) | MEDIUM | LOW | P1 |
| Sampler distribution callback (port) | MEDIUM | MEDIUM | P1 |
| WeightedRandomSampler (port) | HIGH | MEDIUM | P1 |
| Standard augmentation pipeline | HIGH | LOW | P1 |
| Cosine LR + warmup | HIGH | LOW | P1 |
| Mixed precision / gradient clipping / accumulation | HIGH | LOW | P1 |
| Hydra config | HIGH | MEDIUM | P1 |
| WandB logging | MEDIUM | LOW | P1 |
| Per-class F1 | MEDIUM | LOW | P2 |
| AUROC | MEDIUM | LOW | P2 |
| Prediction visualization callback | MEDIUM | MEDIUM | P2 |
| RandAugment | MEDIUM | LOW | P2 |
| MixUp / CutMix | MEDIUM | MEDIUM | P2 |
| Step LR scheduler | LOW | LOW | P2 |
| GradCAM callback | HIGH | HIGH | P3 |
| SAM optimizer | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch — needed for feature parity with sibling repo
- P2: Should have — adds classification-specific value beyond sibling parity
- P3: Nice to have — defer until P1+P2 are stable

---

## Competitor Feature Analysis

Reference frameworks analyzed for classification-specific feature norms.

| Feature | MIC-DKFZ image_classification | timm training scripts | karasawatakumi pytorch-image-classification | Our Approach |
|---------|-------------------------------|----------------------|----------------------------------------------|--------------|
| Backbone source | Custom + timm | timm native | timm via Lightning | timm via Lightning (same) |
| Augmentation | RandAugment, AutoAugment, Cutout, MixUp | RandAugment, MixUp, CutMix, Random Erasing | Basic torchvision | Standard torchvision.v2; RandAugment as P2 |
| Metrics | Accuracy, confusion matrix, parity plots | Top-1/5 accuracy | Top-1 accuracy | Accuracy + per-class + F1 + confusion matrix |
| LR scheduling | Cosine, MultiStep, Step + warmup | Cosine, step | Step, multistep, reduce-on-plateau | Cosine + warmup (P1); step (P2) |
| Class imbalance | Not featured | Not featured | Not featured | WeightedRandomSampler + class-weighted loss (P1, domain requirement) |
| EMA | No | Yes (native timm) | No | Yes, port from sibling repo |
| ONNX export | No | No | No | Yes, port from sibling repo (required for inference pipeline) |
| Config system | Hydra | argparse | argparse | Hydra (parity with sibling) |
| Visualization | Confusion matrix (WandB) | None | TensorBoard | Confusion matrix heatmap + prediction visualization |
| Mixed precision | Yes (default) | Yes (AMP) | No | Yes (Lightning Trainer flag) |
| GradCAM | No | No | No | P3 (differentiator) |

---

## Sources

- Sibling repo direct code inspection: `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/` (HIGH confidence)
- [TorchMetrics Classification Metrics — official docs](https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html) (HIGH confidence)
- [TorchMetrics MulticlassAUROC — official docs](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html) (HIGH confidence)
- [TorchMetrics MulticlassF1Score — official docs](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html) (HIGH confidence)
- [PyTorch Lightning EarlyStopping — official docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html) (HIGH confidence)
- [PyTorch Lightning ModelCheckpoint — official docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html) (HIGH confidence)
- [timm training scripts and augmentation docs](https://timm.fast.ai/training_scripts) (HIGH confidence)
- [timm MixUp/CutMix docs](https://timm.fast.ai/mixup_cutmix) (HIGH confidence)
- [MIC-DKFZ image_classification framework](https://github.com/MIC-DKFZ/image_classification) (MEDIUM confidence — README inspection)
- [ResNet Strikes Back paper — improved timm training procedure](https://arxiv.org/pdf/2110.00476) (HIGH confidence — published paper)
- [pytorch-grad-cam — GradCAM library](https://github.com/jacobgil/pytorch-grad-cam) (MEDIUM confidence — WebSearch verified)
- [Label Smoothing — shadecoder guide 2025](https://www.shadecoder.com/topics/label-smoothing-a-comprehensive-guide-for-2025) (LOW confidence — single source)
- [PyTorch ONNX export docs](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) (HIGH confidence — official docs)

---
*Feature research for: PyTorch Lightning image classification training framework*
*Researched: 2026-02-18*
