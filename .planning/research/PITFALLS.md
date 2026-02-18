# Pitfalls Research

**Domain:** Image classification training framework (PyTorch Lightning + Hydra + ResNet + ONNX)
**Researched:** 2026-02-18
**Confidence:** HIGH (most pitfalls verified against official docs, GitHub issues, and sibling repo patterns)

---

## Critical Pitfalls

### Pitfall 1: class_to_idx Alphabetical Ordering Not Persisted to ONNX

**What goes wrong:**
`ImageFolder` assigns class indices by sorting folder names alphabetically at load time. If the class-to-index mapping is not explicitly serialized and embedded in the ONNX model (or stored alongside it), inference will decode predictions with the wrong labels. On filesystems where sort order differs (case sensitivity, locale), two runs can produce different mappings from the same dataset.

**Why it happens:**
Developers train a model, export it to ONNX, and assume the index-to-class mapping is self-evident. The ONNX file stores only tensors — no class names. At inference time, the consumer must reconstruct the mapping, and if they use a different folder enumeration order, predictions appear correct (loss is low, accuracy is high) but class labels are wrong.

This is especially dangerous for jersey number OCR: class "0" (digit zero) gets index 0 alphabetically, but "10" sorts before "2" as a string, making multi-digit classes a landmine.

**How to avoid:**
- Always serialize `dataset.class_to_idx` to a `labels_mapping.json` file during training, co-located with the ONNX export.
- Embed class names as ONNX metadata using `model.metadata_props` or write a sidecar JSON that the inference consumer loads.
- In the `LightningDataModule`, expose `class_to_idx` as a property and log it explicitly to the Hydra output directory.
- Test: load the exported ONNX model in a fresh Python session with no dataset reference and verify predictions against known images.

The sibling `object-detection-training` repo already writes a `labels_mapping.json` — replicate this pattern from day one.

**Warning signs:**
- Model accuracy in training looks great, but real-world inference returns wrong digit classes.
- `class_to_idx` is only accessible on the live dataset object, never written to disk.
- Confusion matrix shows off-diagonal errors on adjacent numeric classes (e.g., "1" confused with "10").

**Phase to address:** Phase 1 (Data Module) — `class_to_idx` serialization must be part of `DataModule.setup()` or the training task entry point, not deferred to a later phase.

---

### Pitfall 2: Augmentation Applied to Validation / Test Splits

**What goes wrong:**
Random augmentations (flips, color jitter, random crop) leak into validation and test splits, producing unreliable metrics. With `ImageFolder` + `random_split`, if augmentation transforms are attached to the parent dataset before splitting, all splits inherit train augmentations. Validation accuracy appears higher than it should be (augmentation regularization effect) or noisily varies epoch to epoch.

**Why it happens:**
PyTorch's `ImageFolder` accepts a single `transform` argument. Developers split with `random_split` after attaching augmentation transforms, not realizing all resulting `Subset` objects share the same transform. This is a documented community pitfall in PyTorch forums.

**How to avoid:**
- Never use `random_split` on an augmented `ImageFolder`. Use separate directory structures (`train/`, `val/`, `test/` subdirs) and construct three separate `ImageFolder` instances with distinct transforms.
- In `LightningDataModule`, define `train_transforms` (with augmentation) and `val_transforms` (resize + normalize only) as separate properties, and instantiate datasets separately.
- Hydra config group `transforms/` should have `train.yaml` and `val.yaml` explicitly separate.
- The `DataModule.val_dataloader()` must use `val_transforms`, never `train_transforms`.

**Warning signs:**
- Validation loss/accuracy varies significantly between epochs with no learning rate change.
- Removing augmentation from training has no visible effect on validation metrics.
- `Subset.__getitem__` shows the same transform as the training set.

**Phase to address:** Phase 1 (Data Module) — bake separate transform pipelines into the `LightningDataModule` from the start.

---

### Pitfall 3: ImageNet Normalization Mismatch at Inference

**What goes wrong:**
ResNet18/50 pretrained weights expect inputs normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` in `[0, 1]` float range. If the preprocessing pipeline at inference (in the `basketball-2d-to-3d` consumer) applies a different normalization (or none), predictions degrade silently — the model runs without errors but outputs are garbage.

**Why it happens:**
ONNX export strips the normalization transform — it's a Python-side pre-processing step, not part of the model graph by default. The training pipeline normalizes inputs, but the exported ONNX model receives raw tensors. The inference consumer must apply identical normalization. This disconnect is never validated automatically.

**How to avoid:**
- Either bake normalization INTO the ONNX graph by prepending a normalization layer (preferred for deployment), or document the exact normalization parameters in the `labels_mapping.json` / model metadata.
- Add a `Normalize` layer as the first `nn.Module` layer in the exported model (not just in the transform pipeline), so the ONNX model accepts raw `[0, 1]` floats or raw uint8 pixels and self-normalizes.
- Write an integration test that runs the ONNX model against a known image with the expected preprocessing and asserts a specific output class — test this from the first export.

**Warning signs:**
- Accuracy of PyTorch model in training is high; ONNX inference accuracy is low on the same images.
- Exported ONNX model returns nearly uniform softmax probabilities.
- Inference pipeline does `img / 255.0` but omits mean/std subtraction.

**Phase to address:** Phase 2 (Model Module) — define the normalization strategy for ONNX export during model architecture design, not after training.

---

### Pitfall 4: EMA Checkpoint Saves Non-EMA Weights When Best Model is Checkpointed

**What goes wrong:**
`ModelCheckpoint` saves the model state when a new best metric is observed — but during training, the model holds the *live weights*, not the EMA weights. EMA weights are only swapped in for validation via `on_validation_start` / `on_validation_end`. If checkpointing triggers inside the validation loop at a timing boundary, the saved checkpoint may contain the live (non-EMA) weights, while the metric that triggered the save was computed using EMA weights. The "best" checkpoint is then inconsistent.

**Why it happens:**
This is a known timing issue documented in the PyTorch Lightning GitHub discussions (#11276). The EMA callback and ModelCheckpoint callback execute in a specific hook ordering. If the ordering is not explicitly managed, the checkpoint saved at "best val_acc" actually contains pre-swap live weights.

The sibling repo's `EMACallback` does correctly swap in `on_validation_start` and restore in `on_validation_end`. But `ModelCheckpoint` saves from `state_dict()` which is called during `on_validation_epoch_end` — which can execute between the swap/restore cycle depending on hook ordering.

**How to avoid:**
- After training, always export ONNX from the EMA weights directly (via `EMACallback.ema_state_dict`), not from the best checkpoint loaded back.
- Write an explicit `save_ema_checkpoint()` call in `on_train_end` that saves EMA weights to a dedicated `.ckpt` file separate from ModelCheckpoint-managed files.
- Validate that the final ONNX model produces the same accuracy as the EMA-weighted validation run (not a random lower number).
- The ONNX export callback should load from the EMA state dict, not from the raw best `.ckpt` file, to avoid this ambiguity.

**Warning signs:**
- `model_best.onnx` inference accuracy is lower than the best validation accuracy logged during training.
- Reloading the best checkpoint and running validation produces different accuracy than the training log shows.
- No explicit test that verifies ONNX output matches Lightning model output on the same input.

**Phase to address:** Phase 3 (Callbacks) — EMA + ModelCheckpoint + ONNXExport callback integration must be tested end-to-end, with an explicit assertion test.

---

### Pitfall 5: TorchMetrics Mixed Step/Epoch Logging Produces Incorrect Accuracy

**What goes wrong:**
Calling `self.log("val/acc", self.val_acc)` (passing the metric object) in `validation_step` AND separately calling `self.log("val/acc", self.val_acc.compute())` in `on_validation_epoch_end` causes Lightning to reset the metric between the two calls. The epoch-level log then computes on an empty state, logging `0.0` or NaN for `val/acc`. This is a documented bug pattern in torchmetrics (#597).

**Why it happens:**
When a metric object is passed to `self.log()`, Lightning owns the reset cycle. When you additionally call `.compute()` manually, you're computing on the accumulated state, but Lightning has already reset it for the next epoch. The two logging patterns are mutually exclusive.

**How to avoid:**
- Pick ONE pattern per metric and stick to it:
  - Pattern A (recommended): Call `self.val_acc.update(preds, targets)` in `validation_step`, call `self.log("val/acc", self.val_acc.compute())` and `self.val_acc.reset()` in `on_validation_epoch_end`.
  - Pattern B: Pass the metric object directly to `self.log("val/acc", self.val_acc)` in `validation_step` and let Lightning manage compute/reset. Do NOT touch the metric in `on_validation_epoch_end`.
- Never mix the two patterns for the same metric.
- For the confusion matrix callback (which returns a matrix, not a scalar), always use Pattern A and handle logging manually via `logger.experiment`.

**Warning signs:**
- `val/acc` shows 0.0 or NaN for one of every two epochs.
- Confusion matrix is all-zeros in logged images.
- `val/acc` in logs is wildly inconsistent with training behavior.

**Phase to address:** Phase 2 (Model Module) — enforce consistent metric logging pattern in the `ClassifierModule` from the start.

---

## Moderate Pitfalls

### Pitfall 6: BatchNorm in Training Mode During ONNX Export

**What goes wrong:**
If `model.eval()` is not called before `torch.onnx.export()`, BatchNorm layers remain in training mode. Training-mode BatchNorm uses batch statistics (mean/var of the current batch) rather than running statistics — this produces numerically different outputs at inference time. The exported ONNX model gives correct results on large batches but degrades significantly on batch size 1 (the typical inference case).

ResNet18/50 are entirely BatchNorm-heavy; this pitfall is especially impactful.

**How to avoid:**
- Always call `pl_module.eval()` before `torch.onnx.export()` in the ONNX export callback.
- After export, call `pl_module.train()` to restore training mode.
- Verify by running the ONNX model with `onnxruntime` on batch size 1 and asserting output matches PyTorch model output (with `torch.no_grad()` and `model.eval()`).

**Warning signs:**
- A `UserWarning` during export mentioning BatchNorm or training mode.
- ONNX model output differs from PyTorch output by more than floating-point epsilon.
- ONNX accuracy is lower on single-image inference than multi-image batches.

**Phase to address:** Phase 3 (Callbacks / ONNX Export).

---

### Pitfall 7: Jersey Number Class Imbalance Destroys Rare-Class Recognition

**What goes wrong:**
Basketball jersey numbers are inherently imbalanced — single-digit numbers (0-9) appear far more frequently than two-digit numbers (10-99). Even within single digits, some (like 1, 2) dominate. A model trained with default `CrossEntropyLoss` will achieve high accuracy by ignoring rare classes and will confuse visually similar digits (6/9, 1/7, 3/8).

Research on jersey number OCR datasets confirms: "24% of the numbers have less than 100 samples and only 5% reach the 400-sample mark" in typical sports datasets.

**How to avoid:**
- Compute class weights from the training split and pass `weight` tensor to `nn.CrossEntropyLoss`.
- Log per-class accuracy in the confusion matrix callback to detect which classes are failing.
- Use the `DatasetStatisticsCallback` to log class distribution at training start and flag severe imbalance.
- Consider oversampling minority classes via a weighted sampler (`WeightedRandomSampler`) as an alternative to loss weighting.
- Set early stopping `min_delta` conservatively — aggregate accuracy will plateau while rare class accuracy may still improve.

**Warning signs:**
- Validation accuracy is high (>90%) but the confusion matrix shows entire rows/columns of misclassifications for certain digits.
- The model correctly classifies only the 5-6 most common digits.
- Loss plateaus early but per-class recall for rare digits remains near 0.

**Phase to address:** Phase 1 (Data Module) — class weight computation and weighted sampler are data pipeline decisions.

---

### Pitfall 8: Transfer Learning Learning Rate Too High Destroys Pretrained Features

**What goes wrong:**
Fine-tuning ResNet with a high learning rate (e.g., `1e-3`) on a small domain-specific dataset (jersey number crops) destroys the ImageNet pretrained features in early layers. The classifier head is randomly initialized and receives large gradients, which propagate backward and corrupt earlier convolutional layers. The model then takes many more epochs to converge, or never recovers its pretrained baseline.

**How to avoid:**
- Use a low learning rate for fine-tuning: `1e-4` or lower for the full network.
- Optionally, use discriminative learning rates: lower LR for backbone layers (e.g., `1e-5`) and higher for the classification head (e.g., `1e-3`). This is achievable with parameter group configuration in the optimizer.
- Add a warmup schedule for the first few hundred steps to prevent the randomly initialized head from generating destructive early gradients.
- The Hydra `models/resnet18.yaml` and `models/resnet50.yaml` configs should encode these tuned defaults, not leave LR as an afterthought.

**Warning signs:**
- Validation accuracy after epoch 1 is LOWER than the pretrained model baseline (which should be ~70-80% on any classification task with ImageNet features).
- Training loss is NaN or diverges in the first epoch.
- Accuracy improvement is very slow compared to the detection counterpart.

**Phase to address:** Phase 2 (Model Module) — bake tuned LR defaults into model configs.

---

### Pitfall 9: FP16 Mixed Precision Loss Scaling NaN with Small Batch Sizes

**What goes wrong:**
With `precision: 16-mixed` on a T4 GPU and small batch sizes, the dynamic loss scaler may produce `inf`/`NaN` gradients when classification losses become very small (after model converges). The GradScaler skips optimizer steps when overflow is detected, but if the loss scale factor collapses too aggressively, training effectively stalls with no visible error — just no parameter updates.

**Why it happens:**
FP16 has a smaller dynamic range than FP32. When training converges and loss values become very small, the gradient magnitude after loss scaling can underflow to zero in FP16. The GradScaler tries to compensate but may oscillate.

**How to avoid:**
- Monitor `GradScaler` scale factor via the `LearningRateMonitor` or a custom callback; if it drops below `2^8`, training has an AMP instability issue.
- Use `precision: bf16-mixed` if the T4 supports it (T4 supports BF16 as of PyTorch 2.0+), as BF16 keeps the FP32 exponent range and is more stable.
- Add `gradient_clip_val: 1.0` in trainer config (the sibling repo uses `5.0` — classification benefits from tighter clipping).
- If NaN occurs, fall back to `precision: 32-true` as a debugging step.

**Warning signs:**
- Training loss hits a floor and stops decreasing after epoch 5-10.
- `GradScaler` scale factor decreasing over time (visible in Lightning logs with `log_every_n_steps`).
- `val/acc` does not improve despite low training loss.

**Phase to address:** Phase 2 (Model Module) / Phase 4 (Training Config) — verify AMP stability on T4 early in training.

---

### Pitfall 10: Hydra Output Directory Collision When Resuming Training

**What goes wrong:**
Hydra creates timestamped output directories for each run. If a training job is interrupted and restarted, a new output directory is created and Lightning's `ModelCheckpoint` begins saving to a fresh location — losing reference to the previous best checkpoint. The resumed run effectively starts from scratch in terms of checkpoint management, even if Lightning's `resume_from_checkpoint` points to the old checkpoint for weights.

**Why it happens:**
Hydra's `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` scheme always creates new directories. Lightning's `ModelCheckpoint.dirpath` is set to `${hydra:runtime.output_dir}/checkpoints`, which changes every run.

**How to avoid:**
- Set `ModelCheckpoint.dirpath` to a fixed path (`checkpoints/experiment_name/`) that is independent of the Hydra output directory.
- Use `ModelCheckpoint(save_last=True)` and configure Lightning trainer's `ckpt_path="last"` for automatic resume.
- Document in the Hydra config that `dirpath` must be manually set for long-running experiments.

**Warning signs:**
- `outputs/` contains many timestamped directories each with only a few checkpoints.
- `last.ckpt` exists in a different directory than where the current run is saving.
- Resumed training logs show `epoch=0` instead of the expected epoch number.

**Phase to address:** Phase 4 (Training Config / Infrastructure).

---

### Pitfall 11: Confusion Matrix Callback Device Mismatch Crashes Training

**What goes wrong:**
`torchmetrics.ConfusionMatrix` accumulates state tensors on whichever device the first batch arrives on. If the callback initializes the metric on CPU but tensors arrive on GPU, a `RuntimeError: Expected all tensors to be on the same device` error crashes training — typically after the first validation epoch.

**Why it happens:**
Metric objects declared in callbacks are not automatically moved to the training device by Lightning (unlike metrics declared inside `LightningModule.__init__`). Only metrics registered as `nn.Module` attributes on `LightningModule` are moved automatically.

**How to avoid:**
- Declare confusion matrix metrics inside `LightningModule.__init__` (not in callbacks), or explicitly call `.to(device)` in the callback's `on_validation_start` hook.
- Use `ConfusionMatrix(...).to(self.device)` where `self.device` is resolved from the Lightning module.
- Alternatively, accumulate raw predictions/targets as Python lists in the callback and compute the confusion matrix with `sklearn` on CPU at `on_validation_epoch_end` — avoids the device issue entirely.

**Warning signs:**
- `RuntimeError: Expected all tensors to be on the same device` during the first validation epoch.
- Error appears only when `accelerator: gpu` is set, not during CPU testing.
- The confusion matrix callback works on MPS (Mac) but crashes on T4 (CUDA).

**Phase to address:** Phase 3 (Callbacks).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single `transforms` for train+val | Faster initial setup | Augmentation leaks into val; inflated metrics; misleading early stopping | Never — always separate from day one |
| Skip `class_to_idx` serialization | Less boilerplate | Inference pipeline uses wrong label mapping; silent accuracy regression in production | Never — write it in Phase 1 |
| Hardcode normalization constants in inference consumer | Works now | Model swap requires updating two repos; mismatch goes undetected | Acceptable only for throwaway prototypes; bake into ONNX by Phase 3 |
| Use `model.ckpt` directly as ONNX source without verifying EMA | Saves one step | ONNX model weights may not match validation accuracy; silent degradation | Never in production exports |
| Skip per-class accuracy logging | Faster iteration | Class imbalance failure invisible until deployed | Never — add in Phase 1 DatasetStatistics callback |
| Use aggregate accuracy as early stopping monitor | Simple | Stops training while rare classes still improving | Acceptable for MVP; revisit in Phase 2 |
| Fix batch size in ONNX export without dynamic axes | Simpler export | ONNX model only works at training batch size; cannot serve single images | Never — always set dynamic axes for batch dim |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ImageFolder + ONNX | Not saving `class_to_idx` alongside ONNX | Write `labels_mapping.json` in ONNXExportCallback with class names and indices |
| EMA + ModelCheckpoint | Checkpoint saves live weights, not EMA weights | Export ONNX from `EMACallback.ema_state_dict` directly in `on_train_end`, not from loaded checkpoint |
| Hydra + Lightning `log_dir` | Hydra output dir and Lightning log dir diverge | Set `ModelCheckpoint.dirpath: ${hydra:runtime.output_dir}/checkpoints` explicitly |
| TorchMetrics + `self.log()` | Mixing `.compute()` and metric-object logging for same metric | Pick one pattern: either pass metric object OR call `.compute()` manually, never both |
| ONNX + BatchNorm | Forgetting `model.eval()` before export | Wrap export in `model.eval()` / `model.train()` guard, add post-export numerical test |
| pixi + CUDA Docker | `pixi install` inside Dockerfile downloads CUDA-only packages, bloating image | Pin `cuda` variant explicitly in `pixi.toml` to avoid OS mismatch packages being downloaded |
| T4 + FP16 | T4 supports FP16 but small batches cause GradScaler collapse | Monitor scale factor; prefer BF16 or use gradient accumulation to simulate larger effective batch |
| `ImageFolder` + `random_split` | Augmentation attached before split propagates to all subsets | Use separate `ImageFolder` instances per split, not `random_split` on a single dataset |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `num_workers=0` in DataLoader | GPU utilization < 50%; training is I/O bound, not compute bound | Set `num_workers=4` minimum on T4; use `persistent_workers=True` | Any serious training run — detectable in first 10 steps |
| Not `pin_memory=True` on GPU | CPU→GPU transfer bottleneck; GPU idle between batches | `pin_memory=True` when `accelerator=gpu` | Always on GPU — T4 benefits significantly |
| Validation confusion matrix on every step (not epoch) | Confusion matrix accumulates wrong state; high memory usage | Compute confusion matrix only in `on_validation_epoch_end`, not per-step | With datasets > 1000 validation samples |
| `copy.deepcopy` of EMA state dict on every batch | Slow training if model is large (ResNet50 = 25M params) | EMA update in-place on pre-allocated shadow dict; only deepcopy on validation swap | ResNet50 or larger; detectable as training step time > 1s |
| High-resolution input images without resize | T4 OOM with batch_size > 8 if images are not resized to 224x224 | Enforce image size in transform config; add OOM guard comment in Hydra config | First training run with new dataset |
| `accumulate_grad_batches` without adjusting LR schedule | Effective batch size doubles/quadruples but LR stays the same | Scale LR proportionally with `accumulate_grad_batches` (linear scaling rule) | Noticeable when `accumulate_grad_batches >= 4` |

---

## "Looks Done But Isn't" Checklist

- [ ] **ONNX export:** `labels_mapping.json` written alongside `.onnx` file — verify it exists and contains correct class names matching the training dataset.
- [ ] **ONNX export:** Export uses `model.eval()` and correct EMA weights — verify ONNX model accuracy matches Lightning validation accuracy on a held-out test set.
- [ ] **ONNX export:** Dynamic axes set for batch dimension — verify the ONNX model accepts both batch size 1 and batch size N without shape errors.
- [ ] **Validation augmentation:** Val dataloader transform is `resize + normalize` only (no random flips/color jitter) — inspect `LightningDataModule.val_dataloader()` transforms.
- [ ] **EMA callback:** Validation metrics are computed with EMA weights and original weights restored afterward — log `ema_applied=True` in `on_validation_start` to confirm.
- [ ] **Per-class accuracy:** Confusion matrix callback logs per-class recall — verify it shows per-digit breakdown, not just aggregate accuracy.
- [ ] **Class imbalance:** `DatasetStatisticsCallback` logs class distribution at epoch 0 — confirm all target classes have at least N samples before declaring training complete.
- [ ] **Mixed precision:** Training completes without GradScaler scale factor collapse (stays above `2^16`) — add a scale factor log in the trainer.
- [ ] **Checkpoint resume:** Training can be interrupted and resumed from `last.ckpt` with correct epoch number and optimizer state.
- [ ] **CI green:** All pixi tasks pass (`lint`, `test`, `format-check`, `typecheck`) — do not merge a phase PR without CI green.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong `class_to_idx` in deployed ONNX | HIGH | Retrain or reconstruct mapping from dataset; regenerate `labels_mapping.json`; redeploy ONNX with sidecar; revalidate inference pipeline |
| Augmentation in validation | LOW | Separate transform pipelines; retrain from same checkpoint with corrected DataModule; validation metrics will likely drop to the true (lower) value |
| EMA/checkpoint weight mismatch in ONNX | MEDIUM | Re-export from `EMACallback.ema_state_dict` (not from `.ckpt`); no retraining needed if EMA state is saved |
| TorchMetrics mixed logging | LOW | Fix logging pattern; restart training; issue is code-only, not weights |
| BatchNorm eval mode in ONNX | LOW | Re-export with `model.eval()` guard; no retraining needed |
| NaN loss from FP16 | MEDIUM | Fall back to `bf16-mixed` or `32-true`; restart from last checkpoint; investigate gradient clip value |
| Class imbalance ignored | HIGH | Add class weights or weighted sampler; retrain from scratch; rare-class accuracy will not recover from fine-tuning alone |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| `class_to_idx` not persisted | Phase 1: Data Module | Unit test: `DataModule.setup()` writes `labels_mapping.json` with correct mapping |
| Augmentation in val/test | Phase 1: Data Module | Unit test: `val_dataloader()` transforms contain no random ops |
| ImageNet normalization mismatch at inference | Phase 2: Model Module | Integration test: PyTorch model output vs ONNX model output on same preprocessed input |
| EMA checkpoint weight mismatch | Phase 3: Callbacks | Integration test: ONNX accuracy == best val accuracy logged during training |
| TorchMetrics mixed logging | Phase 2: Model Module | Unit test: `val/acc` is never 0.0 or NaN across 3 epochs of mock training |
| BatchNorm eval mode in ONNX | Phase 3: Callbacks | Unit test: ONNX export callback calls `pl_module.eval()` before export |
| Jersey number class imbalance | Phase 1: Data Module | Integration test: `DatasetStatisticsCallback` logs distribution; per-class recall logged in confusion matrix |
| Transfer learning LR too high | Phase 2: Model Module | Smoke test: after epoch 1, val/acc > 50% (ImageNet features should immediately help) |
| FP16 GradScaler collapse | Phase 4: Training Config | Monitor: GradScaler scale factor stays > `2^10` throughout training |
| Hydra output dir collision on resume | Phase 4: Training Config | Manual test: interrupt training, resume with `ckpt_path=last`, verify epoch number continues |
| Confusion matrix device mismatch | Phase 3: Callbacks | CI test: callback runs without error on GPU device (mock GPU test or explicit `.to(device)` assertion) |

---

## Sources

- PyTorch Lightning EMA discussion (checkpoint timing issue): https://github.com/Lightning-AI/pytorch-lightning/discussions/11276
- PyTorch Lightning ModelCheckpoint step timing issue: https://github.com/Lightning-AI/pytorch-lightning/issues/20919
- TorchMetrics mixed logging bug: https://github.com/Lightning-AI/torchmetrics/issues/597
- TorchMetrics in PyTorch Lightning (official): https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
- ImageFolder augmentation on validation splits (PyTorch Forums): https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580
- ImageFolder class_to_idx ordering issue: https://discuss.pytorch.org/t/how-to-represent-class-to-idx-map-for-custom-dataset-in-pytorch/37510
- Jersey number recognition imbalance research: https://pmc.ncbi.nlm.nih.gov/articles/PMC9583843/
- Jersey number recognition CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf
- PyTorch ONNX export BatchNorm training mode: https://github.com/pytorch/pytorch/issues/75252
- PyTorch ONNX dynamic axes (ResNet): https://github.com/pytorch/pytorch/issues/157621
- PyTorch AMP mixed precision NaN: https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369
- PyTorch official AMP documentation: https://docs.pytorch.org/docs/stable/amp.html
- ResNet fine-tuning LR pitfalls: https://mikulskibartosz.name/the-optimal-learning-rate-during-fine-tuning-of-an-artificial-neural-network
- Hydra ML reproducibility pitfalls: https://marktechpost.com/2025/11/04/how-can-we-build-scalable-and-reproducible-machine-learning-experiment-pipelines-using-meta-research-hydra/
- Sibling repo EMACallback implementation (verified patterns): `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/ema.py`
- Sibling repo ONNXExportCallback (verified patterns): `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/src/object_detection_training/callbacks/onnx_export.py`

---
*Pitfalls research for: image classification training framework (basketball jersey number OCR)*
*Researched: 2026-02-18*
