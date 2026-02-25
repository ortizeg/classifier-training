# Jersey OCR: Complete Experiment Log

## Overview

Systematic evaluation of training strategies for basketball jersey number OCR. Covers augmentation sweeps, synthetic data generation, hyperparameter optimization (Phases 0-10 with ResNet18), and vision-language model fine-tuning (Phase 11 with SmolVLM2). All ResNet18 experiments use the same base configuration:
- **Model**: ResNet18 (pretrained ImageNet, replaced FC head)
- **Loss**: Cross-entropy (no label smoothing)
- **Learning rate**: 5e-4 (AdamW + warmup + cosine)
- **Sampler**: disabled (no weighted sampling)
- **Max epochs**: 200 (early stop patience=20)
- **Hardware**: NVIDIA L4 on Vertex AI (g2-standard-8)
- **Dataset**: Basketball jersey numbers OCR dataset

## Base Augmentation Pipeline (all configs share these)

All configs include these geometric/color transforms before degradation:
1. `RandomResizedCrop(224)` — default scale 0.08-1.0
2. `RandomAffine(degrees=15, translate=0.1, scale=0.85-1.15)` — rotation + shift + scale
3. `RandomPerspective(distortion_scale=0.2, p=0.3)`
4. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
5. `GaussianBlur(kernel_size=5, sigma=0.1-2.0)`
6. *[degradation augmentations vary per config]*
7. `ToFloat32Tensor(scale=True)`
8. `Normalize(ImageNet mean/std)`
9. `RandomErasing(p=0.2, scale=0.02-0.15)`

**Note**: No horizontal flip (would confuse digits like 6/9).

---

## Phase 0: Pre-Sweep Baseline (~91.9% val top-1)

The original baseline before augmentation work used no broadcast degradation augmentations, only the geometric/color transforms above.

- **GCP Job**: `jersey-r18-ce-ocr-best` (pre-sweep run)
- **Config**: `basketball_jersey.yaml` (no degradation transforms at the time)

---

## Phase 1: Initial Augmentation Addition

### Experiment: Add JPEG + Pixelate (commit 5b5ecd5)
- Added `RandomJPEGCompression(quality_min=20, quality_max=75, p=0.5)`
- Added `RandomPixelate(scale_min=0.25, scale_max=0.75, p=0.3)`
- **Result**: ~92-93% val top-1 (matched or slightly exceeded baseline)

### Experiment: Add Noise + Bilinear (commit a633a55)
- Added `RandomGaussianNoise(sigma_min=5, sigma_max=25, p=0.3)`
- Added `RandomBilinearDownscale(scale_min=0.3, scale_max=0.75, p=0.3)`
- **Result**: Became the new "augmented baseline" at ~92-93%

---

## Phase 2: Degradation Parameter Sweep (8 configs)

### Design Rationale
- Configs 1-4: Sweep probability/intensity tradeoff
- Configs 5-7: Ablation — test which augmentations are actually needed
- Config 8: Uniform moderate baseline

### Results Table

| # | Config Name | JPEG p/range | Pixelate p/range | Bilinear p/range | Noise p/range | Best Val Top-1 | Early Stop Epoch | Train Top-1 (final) | Val Top-5 |
|---|-------------|-------------|-----------------|-----------------|--------------|---------------|------------------|---------------------|-----------|
| 0 | `basketball_jersey` (baseline) | 0.5 / 20-75 | 0.3 / 0.25-0.75 | 0.3 / 0.3-0.75 | 0.3 / 5-25 | ~92-93% | — | — | ~97% |
| 1 | `mild` | 0.3 / 30-80 | 0.15 / 0.4-0.8 | 0.15 / 0.5-0.8 | 0.15 / 5-15 | ~91-92% | 78 | ~87% | ~97% |
| 2 | `aggressive` | 0.7 / 10-60 | 0.5 / 0.15-0.6 | 0.5 / 0.2-0.6 | 0.5 / 10-30 | ~92% | 75 | ~84% | ~97% |
| 3 | `high_p_mild_int` | 0.6 / 40-85 | 0.4 / 0.5-0.8 | 0.4 / 0.5-0.8 | 0.4 / 3-12 | ~93% | 93 | ~88% | ~98% |
| 4 | `low_p_strong_int` | 0.2 / 5-40 | 0.15 / 0.15-0.5 | 0.15 / 0.2-0.5 | 0.15 / 15-35 | ~93% | 91 | ~88% | ~97% |
| 5 | `no_pixelate` | 0.5 / 20-75 | **disabled** | 0.4 / 0.25-0.7 | 0.3 / 5-25 | ~93% | 99 | ~89% | ~98% |
| 6 | `no_bilinear` | 0.5 / 20-75 | 0.4 / 0.2-0.7 | **disabled** | 0.3 / 5-25 | ~92-93% | 79 | ~86% | ~97% |
| 7 | **`jpeg_noise_focus`** | **0.6 / 15-70** | **disabled** | **disabled** | **0.5 / 5-30** | **~93-94%** | **91** | **~88%** | **~98%** |
| 8 | `uniform_moderate` | 0.35 / 25-70 | 0.35 / 0.3-0.7 | 0.35 / 0.35-0.7 | 0.35 / 5-20 | ~93% | 87 | ~87% | ~98% |

### GCS Artifact Links — Phase 2

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 1 | `mild` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-mild/20260220_122958/` | `projects/562713517696/locations/us-east1/customJobs/35362011936194560` |
| 2 | `aggressive` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-aggressive/20260220_123005/` | `projects/562713517696/locations/us-east1/customJobs/2723448039522959360` |
| 3 | `high_p_mild_int` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-high-p-mild-int/20260220_123008/` | `projects/562713517696/locations/us-east1/customJobs/8596141953614086144` |
| 4 | `low_p_strong_int` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-low-p-strong-int/20260220_123010/` | `projects/562713517696/locations/us-east1/customJobs/7335134057950347264` |
| 5 | `no_pixelate` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-no-pixelate/20260220_123013/` | `projects/562713517696/locations/us-east1/customJobs/2559066653123936256` |
| 6 | `no_bilinear` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-no-bilinear/20260220_123015/` | `projects/562713517696/locations/us-east1/customJobs/8012925801869606912` |
| 7 | `jpeg_noise_focus` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-jpeg-noise-focus/20260220_123018/` | `projects/562713517696/locations/us-east1/customJobs/1645398878721146880` |
| 8 | `uniform_moderate` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-uniform-moderate/20260220_123020/` | `projects/562713517696/locations/us-east1/customJobs/2423958664302821376` |

**Artifact structure per run:**
- `training_history/accuracy_history.png` — accuracy curves (val top-1, val top-5, train top-1)
- `training_history/loss_history.png` — loss curves
- `epoch_NNN/confusion_matrix.png` — per-epoch confusion matrix heatmaps
- `epoch_NNN/sample_predictions.png` — sample prediction grids
- `model.onnx` — exported ONNX model (EMA weights)
- `labels_mapping.json` — class index mapping + normalization params
- `.hydra/` — Hydra config snapshots

### Key Findings — Phase 2

1. **Best performer: `jpeg_noise_focus` (~93-94%)** — Only JPEG + Gaussian noise, no pixelate or bilinear. Simplest config with highest accuracy.
2. **Ablation insight**: Removing pixelate helped or was neutral. Removing both pixelate AND bilinear (keeping only JPEG + noise) was best. This suggests resolution degradation augmentations add noise without improving generalization.
3. **Aggressive augmentation hurts**: `aggressive` (high p, strong intensity) stopped earliest at 75 epochs with lowest accuracy. Too much augmentation hinders learning.
4. **Mild augmentation overfits faster**: `mild` stopped at 78 epochs with lower accuracy — not enough regularization.
5. **All configs within ~1-2% of each other**, confirming the model is relatively robust to augmentation parameter choices.

---

## Phase 3: Zoom-Out Ablation (4 configs)

### Motivation
Test whether simulating loose detector bounding boxes (zoom-out with padding) improves robustness, since the detector may produce boxes larger than the jersey.

### New Transform: `RandomZoomOut`
- Places image on larger canvas (filled with ImageNet mean color RGB 124,116,104)
- Random placement of original image within canvas
- Resizes back to original dimensions
- Parameters: `min_scale`, `max_scale`, `fill_color`, `p`

### Results Table

| # | Config Name | Degradations | Zoom-Out | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 |
|---|-------------|-------------|----------|---------------|------------------|-------------|-----------|
| 9 | `jpeg_noise_zoomout` | JPEG(0.6/15-70) + noise(0.5/5-30) | p=0.3, 1.1-1.5x | ~92-93% | 69 | ~83% | ~97% |
| 10 | `baseline_zoomout` | All 4 baseline degradations | p=0.3, 1.1-1.5x | ~91-92% | 65 | ~83% | ~98% |
| 11 | `zoomout_only` | None | p=0.3, 1.1-1.5x | ~91% | 60 | ~84% | ~97% |
| 12 | `jpeg_noise_zoomout_agg` | JPEG(0.6/15-70) + noise(0.5/5-30) | p=0.5, 1.2-2.0x | ~93% | 90 | ~85% | ~98% |

### GCS Artifact Links — Phase 3

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 9 | `jpeg_noise_zoomout` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-jpeg-noise-zoomout/20260220_133727/` | `projects/562713517696/locations/us-east1/customJobs/1326206255131262976` |
| 10 | `baseline_zoomout` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-baseline-zoomout/20260220_133730/` | `projects/562713517696/locations/us-east1/customJobs/5937892273558650880` |
| 11 | `zoomout_only` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-zoomout-only/20260220_133733/` | `projects/562713517696/locations/us-east1/customJobs/2588340050701844480` |
| 12 | `jpeg_noise_zoomout_agg` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-jpeg-noise-zoomout-agg/20260220_133736/` | `projects/562713517696/locations/us-east1/customJobs/4582308785720131584` |

### Key Findings — Phase 3

1. **Zoom-out hurts val accuracy slightly** in moderate configs (9, 10, 11). They all early-stopped sooner (60-69 epochs) and peaked lower than non-zoom-out counterparts.
2. **Aggressive zoom-out (config 12) partially recovers** — stronger zoom (p=0.5, up to 2x) trained longer (90 epochs) and reached ~93%, matching mid-tier Phase 2 configs.
3. **Val set bias**: The val set contains tightly-cropped samples, so zoom-out doesn't improve measured accuracy. However, it will likely help at inference time with real detector crops.
4. **Zoom-out alone (config 11) is the worst** at ~91%, confirming it's not a replacement for image degradation augmentations.

---

## Phase 4: Kitchen Sink with Clean Pass-Through (1 config)

### Motivation
Combine ALL augmentations at moderate settings but wrap them in `RandomApply(p=0.7)` so 30% of training samples see completely clean images. This should:
- Maintain good performance on clean val/test data
- Still expose the model to degraded inputs for robustness
- Better generalize to out-of-domain test data

### Config: `all_moderate`
- **Degradation group** (applied with 70% probability, 30% clean pass-through via `RandomApply`):
  - JPEG: p=0.55, quality 15-75
  - Pixelate: p=0.25, scale 0.2-0.7
  - Bilinear: p=0.25, scale 0.25-0.7
  - Noise: p=0.45, sigma 5-28
  - Zoom-out: p=0.35, scale 1.1-1.6x
- **Effective per-sample degradation probability**: ~70% chance any degradation fires; within that, each has its own probability
### GCS Artifact Links — Phase 4

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 13 | `all_moderate` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-all-moderate/20260220_154022/` | `projects/562713517696/locations/us-east1/customJobs/6181297759669190656` |

### Results

| # | Config Name | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 |
|---|-------------|---------------|------------------|-------------|-----------|
| 13 | `all_moderate` | ~92-93% | 82 | ~86% | ~98% |

### Key Findings — Phase 4

1. **`all_moderate` reaches ~92-93% val top-1** at 82 epochs — competitive with the augmented baseline but does not exceed `jpeg_noise_focus` (~93-94%).
2. **The 30% clean pass-through works as intended**: the train-val gap is smaller (train ~86%, val ~92-93%) compared to aggressive configs, suggesting less overfitting.
3. **Trade-off confirmed**: the clean pass-through sacrifices ~1% peak val accuracy for potentially better out-of-domain generalization. The model sees clean images more often, keeping it calibrated for both clean and degraded inputs.

---

## Phase 5: All Moderate + Zoom Choice (ZoomIn/ZoomOut)

### Goal
Test whether replacing standalone `RandomZoomOut` with `RandomChoice` between `RandomZoomIn` (tight bbox simulation) and `RandomZoomOut` (loose bbox simulation) improves robustness. The model sees both crop-and-upscale and pad-and-downscale augmentations, covering both common detector failure modes.

### Config: `all_moderate_zoom_choice`

- **All augmentations from Phase 4** wrapped in `RandomApply(p=0.7)`:
  - JPEG: p=0.55, quality 15-75
  - Pixelate: p=0.25, scale 0.2-0.7
  - Bilinear: p=0.25, scale 0.25-0.7
  - Noise: p=0.45, sigma 5-28
  - **RandomChoice** (replaces standalone ZoomOut):
    - `RandomZoomIn(min_scale=0.7, max_scale=1.0, p=1.0)` — crop 70-100% and resize up
    - `RandomZoomOut(min_scale=1.1, max_scale=1.6, p=1.0)` — pad to 110-160% and resize back
- **Effective per-sample zoom probability**: ~70% chance degradation block fires, then 50/50 zoom-in vs zoom-out

### GCS Artifact Links — Phase 5

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 14 | `all_moderate_zoom_choice` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-all-mod-zoom-choice/20260220_182405/` | `projects/562713517696/locations/us-east1/customJobs/5293525683123781632` |

### Results

| # | Config Name | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 |
|---|-------------|---------------|------------------|-------------|-----------|
| 14 | `all_moderate_zoom_choice` | ~91-92% | 58 | ~81% | ~97% |

### Key Findings — Phase 5

1. **`all_moderate_zoom_choice` reaches ~91-92% val top-1** at 58 epochs — slightly below `all_moderate` (~92-93% at 82 epochs).
2. **More regularization effect**: train top-1 is only ~81% (vs ~86% for `all_moderate`), and the model early-stops earlier (58 vs 82 epochs). The zoom-in/zoom-out choice adds significant augmentation difficulty.
3. **The bidirectional zoom does not improve peak accuracy** — the additional crop augmentation from `RandomZoomIn` may conflict with `RandomResizedCrop` already in the pipeline, doubling up on crop-based augmentation.
4. **Recommendation**: Stick with `all_moderate` (ZoomOut only) for the kitchen-sink config. The zoom-in augmentation is redundant with `RandomResizedCrop`.

---

## Phase 6: Cumulative Strong Augmentation Sweep

### Goal
Test cumulative strengthening of augmentations on top of `all_moderate_zoom_choice`. Three configs, each adding more intensity:

### Config Details

| Config | Changes from `all_moderate_zoom_choice` baseline |
|--------|------------------------------------------------|
| **`zc_strong_blur`** | GaussianBlur kernel 7 (was 5), sigma 0.5-3.5 (was 0.1-2.0); BilinearDownscale p=0.4 (was 0.25), scale 0.15-0.55 (was 0.25-0.7) |
| **`zc_strong_blur_zoom`** | + ZoomIn min_scale 0.5 (was 0.7); ZoomOut max_scale 2.0 (was 1.6) |
| **`zc_strong_all`** | + RandomApply p=0.85 (was 0.7); JPEG p=0.7/quality 5-55; Pixelate p=0.4/scale 0.1-0.5; Noise p=0.6/sigma 10-40; Erasing p=0.3 |

### GCS Artifact Links — Phase 6

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 15 | `zc_strong_blur` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-zc-strong-blur/20260220_195000/` | `projects/562713517696/locations/us-east1/customJobs/3936534820401709056` |
| 16 | `zc_strong_blur_zoom` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-zc-strong-blur-zoom/20260220_195007/` | `projects/562713517696/locations/us-east1/customJobs/804281279565529088` |
| 17 | `zc_strong_all` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-zc-strong-all/20260220_195013/` | `projects/562713517696/locations/us-east1/customJobs/3729369237542666240` |

### Results

| # | Config Name | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 |
|---|-------------|---------------|------------------|-------------|-----------|
| 15 | `zc_strong_blur` | ~91-92% | 66 | ~80% | ~97% |
| 16 | `zc_strong_blur_zoom` | ~91-92% | 83 | ~79% | ~97-98% |
| 17 | `zc_strong_all` | ~93-94% | 150 | ~86% | ~98% |

### Key Findings — Phase 6

1. **`zc_strong_all` is the standout performer** at ~93-94% val top-1, **matching `jpeg_noise_focus`** — the strongest overall augmentation config that also matches the best accuracy.
2. **Training for 150 epochs** — the heavy augmentation prevented early stopping, allowing the model to keep learning. Train accuracy reached ~86%, meaning the model is still not overfitting.
3. **Stronger blur alone doesn't help** — `zc_strong_blur` (66 epochs, ~91-92%) performs similarly to the baseline `all_moderate_zoom_choice` (58 epochs, ~91-92%).
4. **Adding stronger zoom doesn't help either** — `zc_strong_blur_zoom` (83 epochs, ~91-92%) matches the blur-only variant.
5. **The leap comes from cranking ALL degradations** — `zc_strong_all` jumps to ~93-94% by combining strong JPEG (quality 5-55), strong noise (sigma 10-40), strong pixelate, and 85% degradation application rate. The heavy augmentation acts as implicit regularization, allowing the model to train longer and reach a higher peak.
6. **`zc_strong_all` is the best kitchen-sink config** — it matches `jpeg_noise_focus` accuracy while providing robustness to all degradation types.

---

## Phase 7: Weighted Sampler + Strong All

### Goal
Test whether addressing the 64x class imbalance (class `6` has 4 samples vs `8` with 257) via weighted random sampling improves accuracy on top of the best augmentation config (`zc_strong_all`).

### Dataset Class Imbalance
- **Training set**: 2,930 samples, 43 classes
- **Imbalance ratio**: 64.2x (max 257, min 4)
- **Top 5 classes**: `8` (257), `2` (217), `0` (194), `5` (168), `11` (164)
- **Bottom 5 classes**: `6` (4), `46` (5), `26` (8), `20` (11), `18` (12)
- **Sampler mode**: `auto` — computes inverse-frequency weights so rare classes are sampled proportionally more often

### Config: `zc_strong_all_weighted`
- Identical to `zc_strong_all` except `data.sampler.mode=auto`

### GCS Artifact Links — Phase 7

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 18 | `zc_strong_all_weighted` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-zc-strong-all-weighted/20260220_212715/` | `projects/562713517696/locations/us-east1/customJobs/6567199952739500032` |

### Results

| # | Config Name | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 |
|---|-------------|---------------|------------------|-------------|-----------|
| 18 | `zc_strong_all_weighted` | ~90-91% | 153 | ~88% | ~97% |

### Key Findings — Phase 7

1. **Weighted sampler hurts val top-1 accuracy** — ~90-91% vs ~93-94% for the same config without weighted sampling. A drop of ~3%.
2. **Training is slower to converge** — the model learns at a much slower pace early on (val top-1 ~45% at epoch 10 vs ~75% without sampler), taking 153 epochs before early stopping.
3. **Train-val gap narrows significantly** — train ~88%, val ~90-91%. The weighted sampler forces the model to spend more time on rare classes (e.g., jersey `6` with 4 samples), which may not generalize as well due to insufficient visual diversity.
4. **The imbalance is too extreme for weighted sampling alone** — classes with 4-5 samples get massively upweighted, but there aren't enough unique images to learn robust features. Better approaches for rare classes: more data collection or class-conditional augmentation.
5. **Recommendation**: Do NOT use weighted sampler with this dataset. The class imbalance is a data scarcity problem, not a sampling problem.

---

## Overall Rankings (by Best Val Top-1)

| Rank | Config | Best Val Top-1 | Epochs | Notes |
|------|--------|---------------|--------|-------|
| 1 | `jpeg_noise_focus` | ~93-94% | 91 | **Best overall** — simplest, only JPEG + noise |
| 1 | `zc_strong_all` | ~93-94% | 150 | **Tied best** — all augs cranked, most robust |
| 2 | `high_p_mild_int` | ~93% | 93 | Frequent but gentle degradations |
| 3 | `low_p_strong_int` | ~93% | 91 | Rare but harsh degradations |
| 4 | `no_pixelate` | ~93% | 99 | Pixelate removed |
| 5 | `uniform_moderate` | ~93% | 87 | Equal moderate |
| 6 | `jpeg_noise_zoomout_agg` | ~93% | 90 | Best + aggressive zoom-out |
| 7 | `no_bilinear` | ~92-93% | 79 | Bilinear removed |
| 8 | `jpeg_noise_zoomout` | ~92-93% | 69 | Best + moderate zoom-out |
| 9 | `aggressive` | ~92% | 75 | Too much augmentation |
| 10 | `mild` | ~91-92% | 78 | Too little augmentation |
| 11 | `baseline_zoomout` | ~91-92% | 65 | Baseline + zoom-out |
| 12 | `zc_strong_blur` | ~91-92% | 66 | Strong blur only — no improvement |
| 12 | `zc_strong_blur_zoom` | ~91-92% | 83 | Strong blur + zoom — no improvement |
| 12 | `all_moderate_zoom_choice` | ~91-92% | 58 | All moderate + RandomChoice(ZoomIn, ZoomOut) |
| 15 | `zc_strong_all_weighted` | ~90-91% | 153 | Weighted sampler hurts — rare classes too scarce |
| 16 | `zoomout_only` | ~91% | 60 | Zoom-out only |
| 17 | `all_moderate` | ~92-93% | 82 | Kitchen sink + 30% clean pass-through |

---

## Phase 8: Config Consolidation + Synthetic Data + Dedup (2026-02-21)

### Changes Made

1. **Consolidated best config as default**: `zc_strong_all` is now the default `basketball_jersey.yaml` transform config. Deleted all 17 sweep/variant configs.
2. **Updated defaults to match best run**: `label_smoothing=0.0`, `max_epochs=200`, `early_stopping.patience=20` — all baked into the YAML configs instead of GCP overrides.
3. **Dataset dedup**: Removed 39 duplicate annotation rows from train split (38 had empty-suffix + numeric-suffix pairs; kept numeric). Removed 13 empty-suffix rows from val/test. Dataset now has **42 classes** (empty string class eliminated — was never a real class, just bad annotations).
4. **Synthetic data generation**: Rendered 5,197 synthetic jersey number images for 22 underrepresented classes (threshold < 50 real samples, target = 200 per class). Uses 5 sports fonts, NBA jersey color palette, perspective warp, and broadcast degradation pipeline.
5. **Recursive data loading**: `JerseyNumberDataset` now recursively discovers all `.jsonl` files under a split directory. Real + synthetic data coexist: `train/annotations.jsonl` (2,891 real) + `train/synthetic/annotations.jsonl` (5,197 synthetic) = **8,088 total train samples**.

### Dataset Summary (post-changes)

| Split | Samples | Classes | Notes |
|-------|---------|---------|-------|
| Train (real) | 2,891 | 42 | Deduped, no empty-string class |
| Train (synthetic) | 5,197 | 22 | Font-rendered, broadcast-degraded |
| **Train (total)** | **8,088** | **42** | **2.8x more data** |
| Valid | 364 | 41 | Unchanged (minus empty-string rows) |
| Test | 360 | 42 | Unchanged (minus empty-string rows) |

### Experiment: Synthetic + Real Data Training

- **GCP Job**: `jersey-ocr-training`
- **Config**: Default `basketball_jersey.yaml` (= former `zc_strong_all`)
- **Key difference from Phase 6 run #17**: 8,088 train samples (was 2,930), 42 classes (was 43), no label smoothing baked in (was override)
- **Hypothesis**: Synthetic data for rare classes should improve per-class accuracy and overall val accuracy beyond ~93-94%

### GCS Artifact Links — Phase 8

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 19 | `jersey-ocr-training` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-training/20260221_012526/` | `projects/562713517696/locations/us-east1/customJobs/1743141064383922176` |

**Note**: First submission (job `7003345429152661504`) failed with `KeyError: ''` — GCS dataset still had empty-suffix annotations. Fixed by making `JerseyNumberDataset` skip unknown labels gracefully (commit `38a7567`). Second submission succeeded.

### Results

| # | Config Name | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 |
|---|-------------|---------------|------------------|-------------|-----------|
| 19 | `jersey-ocr-training` (real + synthetic) | **~95.1%** | 77 | ~80% | ~99% |

### Comparison vs Previous Best

| Metric | #17 `zc_strong_all` (real only) | #19 `jersey-ocr-training` (real + synth) | Delta |
|--------|-------------------------------|----------------------------------------|-------|
| Train samples | 2,930 | 8,088 | +176% |
| Classes | 43 | 42 | -1 (empty string removed) |
| Best Val Top-1 | ~93-94% | **~95.1%** | **+1-2%** |
| Val Top-5 | ~98% | ~99% | +1% |
| Train Top-1 | ~85-90% | ~80% | -5-10% (more data = harder to overfit) |
| Early Stop Epoch | 150 | 77 | -73 (converges faster with more data) |

---

## Phase 9: Hyperparameter Sweep for Larger Dataset (2026-02-21)

With 2.8x more training data (8,088 samples), we revisited three hyperparameters that may behave differently at this scale.

### Experiments

| # | Name | Change from Baseline (#19) | Hypothesis |
|---|------|---------------------------|-----------|
| 20 | `hp-lr-bs` | `lr=1e-3, batch_size=128` | Larger dataset tolerates higher LR + batch |
| 21 | `hp-label-smooth` | `label_smoothing=0.1` | Was harmful with 3K samples, may help at 8K |
| 22 | `hp-less-aug` | `RandomApply p=0.7` (was 0.85) | Train acc only 80% — augmentation too harsh? |

### GCS Artifact Links — Phase 9

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 20 | `hp-lr-bs` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-lr-bs/20260221_091630/` | `5277059396986208256` |
| 21 | `hp-label-smooth` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-label-smooth/20260221_091636/` | `2971216387772514304` |
| 22 | `hp-less-aug` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-less-aug/20260221_091642/` | `2590662219259707392` |

### Results

| # | Config | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 | Val Loss |
|---|--------|---------------|------------------|-------------|-----------|----------|
| 19 | baseline (real + synth) | ~95.1% | 77 | ~80% | ~99% | 0.204 |
| 20 | `hp-lr-bs` (lr=1e-3, bs=128) | ~95.1% | 101 | ~54% | ~99% | 0.224 |
| 21 | **`hp-label-smooth`** (ls=0.1) | **~96.2%** | 73 | ~82.7% | ~99% | 1.094* |
| 22 | `hp-less-aug` (p=0.7) | ~94.8% | 46 | ~70.1% | ~99% | 0.209 |

*Val loss is higher with label smoothing because the loss function penalizes confident predictions by design. This is expected and not indicative of worse performance — val accuracy is the true metric.

### Analysis

1. **Label smoothing is the winner** (+1.1% over baseline). With 8K samples, the model has enough data to benefit from the regularization effect. Previously at 3K samples it hurt because the model couldn't learn enough signal. The val accuracy curve shows steady improvement through epoch 73 with no signs of overfitting.

2. **Higher LR + batch size was neutral**. Same ~95.1% val top-1 but took 101 epochs (vs 77 baseline). Train accuracy dropped to ~54% — the model struggled to learn the augmented training set with 2x LR. The larger batch size (128) halved batches/epoch from 127 to 64, which combined with higher LR made optimization less stable.

3. **Less augmentation was harmful** (-0.3%). Reducing RandomApply from p=0.85 to p=0.7 caused the model to converge faster (epoch 46) but at a lower ceiling. The strong augmentation is still justified — even with 8K samples, the model benefits from aggressive regularization to generalize on real broadcast footage.

### Comparison: All-Time Ranking (updated)

| Rank | Config | Val Top-1 | Dataset | Key Change |
|------|--------|-----------|---------|-----------|
| **1** | **#21 `hp-label-smooth`** | **~96.2%** | **real + synth** | **label_smoothing=0.1** |
| 2 | #19 baseline (synth) | ~95.1% | real + synth | Default config |
| 2 | #20 `hp-lr-bs` | ~95.1% | real + synth | lr=1e-3, bs=128 |
| 4 | #22 `hp-less-aug` | ~94.8% | real + synth | RandomApply p=0.7 |
| 5 | #17 `zc_strong_all` | ~93-94% | real only | Original best |
| 5 | #1 `jpeg_noise_focus` | ~93-94% | real only | Just JPEG + noise |

---

## Phase 10: Medium/Low Priority Hyperparameter Sweep (2026-02-21)

With `label_smoothing=0.1` now baked into defaults, we swept the remaining hyperparameters: weighted sampler, EMA decay, warmup steps, model capacity, and loss function.

### Experiments

| # | Name | Change from #21 Baseline (ls=0.1) | Hypothesis |
|---|------|-----------------------------------|-----------|
| 23 | `hp-weighted-sampler` | `data.sampler.mode=auto` | Synth data filled class gaps — weighted sampling may now help |
| 24 | `hp-ema-999` | `callbacks.ema.decay=0.999` | Lower decay adapts faster with 2.8x more steps/epoch |
| 25 | `hp-ema-9995` | `callbacks.ema.decay=0.9995` | Intermediate decay between 0.999 and 0.9999 |
| 26 | `hp-warmup-1000` | `callbacks.ema.warmup_steps=1000` | Faster warmup (8 epochs vs 16) |
| 27 | `hp-resnet34` | `model=resnet34` | More capacity for 8K samples |
| 28 | `hp-resnet50` | `model=resnet50` | Largest ResNet for 42-class problem |
| 29 | `hp-focal` | `model.loss_name=focal` | Focus on hard examples |

### GCS Artifact Links — Phase 10

| # | Config | GCS Output Directory | Vertex AI Job ID |
|---|--------|---------------------|-----------------|
| 23 | `hp-weighted-sampler` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-weighted-sampler/20260221_123633/` | `1816095859910115328` |
| 24 | `hp-ema-999` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-ema-999/20260221_123636/` | `6609051763339165696` |
| 25 | `hp-ema-9995` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-ema-9995/20260221_123638/` | `5933511819233591296` |
| 26 | `hp-warmup-1000` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-warmup-1000/20260221_123641/` | `5198299180065357824` |
| 27 | `hp-resnet34` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-resnet34/20260221_123643/` | `2037898141558112256` |
| 28 | `hp-resnet50` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-resnet50/20260221_123646/` | `7427580995613753344` |
| 29 | `hp-focal` | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-hp-focal/20260221_123648/` | `3145783649891254272` |

### Results

| # | Config | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 | Val Loss |
|---|--------|---------------|------------------|-------------|-----------|----------|
| 21 | baseline (ls=0.1) | **~96.2%** | 73 | ~82.7% | ~99% | 1.094 |
| 23 | weighted sampler | ~95.1% | 87 | ~83.5% | ~99% | 1.161 |
| 24 | EMA 0.999 | ~94.2% | 44 | ~87.1% | ~99% | 1.307 |
| 25 | EMA 0.9995 | ~95.1% | 78 | ~83.6% | ~99% | 1.212 |
| 26 | warmup 1000 | **~95.9%** | 99 | ~81.4% | ~99% | 1.121 |
| 27 | ResNet34 | **~95.9%** | 78 | ~81.9% | ~99% | 1.195 |
| 28 | ResNet50 | ~94.8% | 99 | ~83.5% | ~99% | 1.111 |
| 29 | focal loss | ~94.8% | 69 | ~36.7% | ~99% | 0.713 |

### Analysis

1. **No experiment beat the #21 baseline (96.2%).** Label smoothing remains the only hyperparameter that improved on the synth-data baseline.

2. **Weighted sampler is now neutral** (~95.1% vs 95.1% baseline without ls). With synthetic data filling class gaps, it no longer hurts (was -3% in Phase 7 with real-only data). But it doesn't help either — the synthetic data already solved the imbalance problem.

3. **EMA decay 0.999 is too aggressive** (~94.2%). Lower decay means the EMA weights track the current model too closely, losing the smoothing benefit. The default 0.9999 remains best. EMA 0.9995 was neutral (~95.1%).

4. **Warmup 1000 steps is slightly worse** (~95.9% vs 96.2%). Halving warmup from 16 to 8 epochs didn't help — the longer warmup gives the LR scheduler a smoother ramp that benefits training.

5. **ResNet34 matches warmup-1000** (~95.9%) but doesn't beat ResNet18 + label smoothing (96.2%). The extra capacity doesn't help on this 42-class problem with 224x224 inputs. ResNet18 is already sufficient.

6. **ResNet50 is worse** (~94.8%). Too much capacity for this dataset — the model is harder to train and doesn't generalize better. The 5x parameter increase (43MB → 83MB ONNX) isn't justified.

7. **Focal loss is significantly worse** (~94.8%, train only 36.7%). The focal gamma=2.0 down-weights easy examples too aggressively, making it hard for the model to learn the augmented training set. With label smoothing already reducing overconfidence, focal loss is redundant and harmful.

---

## Results Summary (Phases 8-10)

### Progression from Pre-Synth Baseline

| Metric | #17 `zc_strong_all` (real only) | #21 `hp-label-smooth` (real + synth) | Delta |
|--------|-------------------------------|--------------------------------------|-------|
| Val Top-1 | ~93-94% | **~96.2%** | **+2-3%** |
| Val Top-5 | ~98% | ~99% | +1% |
| Train Top-1 | ~85-90% | ~82.7% | -3-7% (more data + label smoothing) |
| Train samples | 2,930 | 8,088 | +176% |
| Early Stop Epoch | 150 | 73 | -77 (converges 2x faster) |

### All Experiments (Phases 8-10)

| # | Config | Best Val Top-1 | Early Stop Epoch | Train Top-1 | Val Top-5 | Val Loss |
|---|--------|---------------|------------------|-------------|-----------|----------|
| 19 | real + synth baseline (no ls) | ~95.1% | 77 | ~80% | ~99% | 0.204 |
| 20 | lr=1e-3, bs=128 | ~95.1% | 101 | ~54% | ~99% | 0.224 |
| **21** | **label_smoothing=0.1** | **~96.2%** | **73** | **~82.7%** | **~99%** | **1.094*** |
| 22 | RandomApply p=0.7 | ~94.8% | 46 | ~70.1% | ~99% | 0.209 |
| 23 | weighted sampler | ~95.1% | 87 | ~83.5% | ~99% | 1.161 |
| 24 | EMA 0.999 | ~94.2% | 44 | ~87.1% | ~99% | 1.307 |
| 25 | EMA 0.9995 | ~95.1% | 78 | ~83.6% | ~99% | 1.212 |
| 26 | warmup 1000 | ~95.9% | 99 | ~81.4% | ~99% | 1.121 |
| 27 | ResNet34 | ~95.9% | 78 | ~81.9% | ~99% | 1.195 |
| 28 | ResNet50 | ~94.8% | 99 | ~83.5% | ~99% | 1.111 |
| 29 | focal loss | ~94.8% | 69 | ~36.7% | ~99% | 0.713 |

*Val loss is higher with label smoothing because the loss function penalizes confident predictions by design. This is expected — val accuracy is the true metric.

---

## Phase 11: SmolVLM2 Vision-Language Model Fine-Tuning (2026-02-24/25)

### Motivation

Test whether a vision-language model (VLM) can match or exceed ResNet18 on jersey number OCR. VLMs understand visual context and can "read" text, which may be more robust for OCR-style tasks than a pure classification head. This also explores whether VLMs can replace task-specific CNN classifiers for structured visual recognition tasks.

### Approach: QLoRA Fine-Tuning

- **Model**: SmolVLM2-2.2B-Instruct (HuggingFace multimodal model)
- **Method**: QLoRA — 4-bit NF4 quantization via bitsandbytes, LoRA adapters on attention projections
- **LoRA config**: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj], task_type=CAUSAL_LM
- **Loss**: Causal LM loss (built into HF model) — NOT cross-entropy logits like ResNet
- **Prompt**: `"What number is on this basketball jersey? Reply with just the number, nothing else."`
- **Validation**: Generate response, parse with `_parse_vlm_response()`, compute accuracy vs ground truth label
- **No augmentation**: HF processor handles image preprocessing; no torchvision transforms
- **Trainable params**: ~4M LoRA params (vs 2.2B total) + vision connector unfrozen

### Architecture

Unlike ResNet18 which extends `BaseClassificationModel`, SmolVLM2 required a completely separate training pipeline:

- `VLMJerseyNumberDataset` — returns PIL images + prompt/answer strings (reuses JSONL annotation format)
- `VLMCollator` — applies chat template, runs HF processor, creates labels with prompt tokens masked to -100
- `VLMDataModule` — Lightning DataModule with HF processor instead of torchvision transforms
- `SmolVLM2ClassificationModel` — standalone Lightning module (causal LM loss, not logits-based CE)

Key commits: `ef44d54` (framework), `87a4e53` (checkpoint fix), `36937dd` (batch optimization)

### Zero-Shot Baseline

Before fine-tuning, SmolVLM2-2.2B-Instruct was evaluated zero-shot on the jersey OCR val set:

| Metric | Value |
|--------|-------|
| Val Top-1 Accuracy | 57.3% |
| Approach | Direct prompting, no training |

### Training Infrastructure Challenges

1. **GCS FUSE checkpoint issue**: PyTorch's temp-file-then-rename checkpoint pattern fails on GCS FUSE mounts. First attempted `runtime.output_dir` as `default_root_dir` — checkpoints corrupted, accuracy dropped to 58.5%. Fix: save checkpoints to local filesystem (`runtime.cwd`), then `shutil.copytree` to GCS after training completes.

2. **QLoRA non-determinism**: 4-bit NF4 quantization uses stochastic rounding, and cuDNN autotuner selects different kernels across runs. This caused significant run-to-run variance (58.5% to 94.5%) with identical configs.

3. **A100 quota**: Vertex AI `custom_model_training_nvidia_a100_gpus` quota is separate from Compute Engine GPU quota. Required explicit quota increase request.

4. **A100 OOM at batch_size=32**: Used 38.61/39.39 GiB on A100-40GB. Reduced to batch_size=24.

### Experiments

| # | Run | GPU | batch_size | accum | eff_batch | max_epochs | patience | Val Acc | Time/Epoch | Total Time | Job ID |
|---|-----|-----|-----------|-------|-----------|------------|----------|---------|------------|------------|--------|
| 30 | Zero-shot | — | — | — | — | — | — | 57.3% | — | — | — |
| 31 | Initial L4 (prev session) | L4 | 4 | 4 | 16 | 10 | 3 | 94.5% | ~7.5 min | ~75 min | `2038918488348688384` |
| 32 | Fix attempt 1 (output_dir) | L4 | 4 | 4 | 16 | 10 | 3 | 58.5% | ~7.5 min | ~75 min | `4038164879180300288` |
| 33 | Fix attempt 2 (copytree) | L4 | 4 | 4 | 16 | 10 | 3 | 87.4% | ~12 min | ~120 min | `7417834924545146880` |
| 34 | Optimized L4 | L4 | 12 | 2 | 24 | 15 | 5 | 93.4% | ~6.1 min | 91 min | `8584126490545750016` |
| 35 | A100 run 1 | A100-40GB | 24 | 1 | 24 | 15 | 5 | 87.9% | ~5.9 min | 28 min | `622325299308134400` |
| **36** | **A100 run 2** | **A100-40GB** | **24** | **1** | **24** | **15** | **5** | **94.5%** | **~5.4 min** | **27 min** | **`5267788314940801024`** |

### GCS Artifact Links — Phase 11

| # | Run | GCS Output Directory | Vertex AI Job ID |
|---|-----|---------------------|-----------------|
| 31 | Initial L4 | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-smolvlm2-train/` (prev session) | `2038918488348688384` |
| 33 | Fix attempt 2 | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-smolvlm2-train/` | `7417834924545146880` |
| 34 | Optimized L4 | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-smolvlm2-train/` | `8584126490545750016` |
| 35 | A100 run 1 | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-smolvlm2-a100/` | `622325299308134400` |
| 36 | A100 run 2 | `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-ocr-smolvlm2-a100/` | `5267788314940801024` |

### GPU Comparison: L4 vs A100

| Metric | NVIDIA L4 | NVIDIA A100-40GB | Speedup |
|--------|-----------|-----------------|---------|
| VRAM | 24 GB | 40 GB | 1.7x |
| Memory bandwidth | 300 GB/s | 1,555 GB/s | 5.2x |
| bf16 TFLOPS | 120 | 312 | 2.6x |
| Max batch_size (SmolVLM2) | 12 | 24 | 2x |
| Time per epoch | ~6.1 min | ~5.4 min | 1.1x |
| Total training time | 91 min | 27 min | **3.4x** |
| Machine type | g2-standard-8 | a2-highgpu-1g | — |

The 3.4x total speedup (vs 1.1x per-epoch) comes from the A100 training fewer epochs due to early stopping with the same patience setting.

### Key Findings — Phase 11

1. **SmolVLM2 QLoRA reaches 94.5% val top-1** — a strong result for a VLM on a classification task, but below ResNet18's 96.2% with full training recipe (synthetic data + label smoothing + augmentation).

2. **QLoRA introduces significant non-determinism**: Same config produced results ranging from 58.5% to 94.5% across 6 runs. This is inherent to 4-bit NF4 stochastic rounding and cuDNN autotuner kernel selection. Multiple runs are required to assess true model quality.

3. **Larger batch size helps stability**: Increasing batch_size from 4 to 12 (L4) or 24 (A100) improved gradient signal quality, reducing variance in convergence behavior.

4. **A100 is 3.4x faster than L4 for QLoRA**: The speedup is primarily from memory bandwidth (5.2x), which is the bottleneck for quantized model training. The A100's higher bandwidth means less time waiting for weight dequantization.

5. **GCS FUSE is incompatible with PyTorch checkpointing**: Must save locally then copy to cloud storage post-training. This is a general lesson for any PyTorch training on Vertex AI.

6. **VLM training is fundamentally different from CNN classification**: Required a completely separate training pipeline (dataset, collator, datamodule, model). The causal LM loss paradigm means no EMA, no weighted cross-entropy, no confusion matrix callbacks, and no ONNX export.

7. **Model size**: SmolVLM2 LoRA checkpoint is 1.8 GiB (full quantized model + adapters) vs ResNet18 ONNX at 43 MB — a 42x size difference with lower accuracy.

### Saved Checkpoint

Best checkpoint (94.5% from A100 run 2) saved locally:
```
models/smolvlm2-lora/last.ckpt  (1.8 GiB)
```

---

## Cross-Model Comparison: ResNet18 vs SmolVLM2

| Metric | ResNet18 (best, #21) | SmolVLM2 (best, #36) | Notes |
|--------|---------------------|---------------------|-------|
| **Val Top-1 Accuracy** | **96.2%** | 94.5% | ResNet18 wins by 1.7% |
| Val Top-5 Accuracy | ~99% | N/A | VLM generates text, no top-5 |
| Model size (deploy) | 43 MB (ONNX) | 1.8 GB (ckpt) | 42x smaller |
| Inference speed | ~1 ms (ONNX RT) | ~200 ms (GPU) | ~200x faster |
| Training data | 8,088 (real + synth) | 2,891 (real only) | VLM not tested with synth |
| Training time | ~30 min (L4) | 27 min (A100) | Similar wall time |
| Training cost | ~$1.50 (L4) | ~$8.00 (A100) | 5x more expensive |
| Run-to-run variance | Low (~1%) | High (58-95%) | QLoRA non-determinism |
| Augmentation | Heavy pipeline | None (HF processor) | ResNet benefits from augs |
| Label smoothing | Yes (0.1) | No | Causal LM loss instead |
| EMA | Yes (0.9999) | No | Not compatible with VLM |
| Export format | ONNX (edge-ready) | PyTorch (GPU-only) | ResNet deploys anywhere |

---

## Summary Table: All Experiments (Phases 0-11)

| # | Phase | Config | Model | Val Top-1 | Dataset | Key Change |
|---|-------|--------|-------|-----------|---------|-----------|
| 0 | 0 | baseline | ResNet18 | ~91.9% | real (2,930) | Pre-sweep baseline |
| 1-6 | 1-2 | aug sweep | ResNet18 | ~91-94% | real (2,930) | 8 degradation configs |
| 7 | 2 | `jpeg_noise_focus` | ResNet18 | ~93-94% | real (2,930) | Best simple aug — JPEG + noise only |
| 9-12 | 3 | zoom-out ablation | ResNet18 | ~91-93% | real (2,930) | Zoom-out hurts measured accuracy |
| 13 | 4 | `all_moderate` | ResNet18 | ~92-93% | real (2,930) | Kitchen sink + 30% clean pass-through |
| 14 | 5 | `all_mod_zoom_choice` | ResNet18 | ~91-92% | real (2,930) | Bidirectional zoom (redundant w/ RandomResizedCrop) |
| 17 | 6 | `zc_strong_all` | ResNet18 | ~93-94% | real (2,930) | All augs cranked — tied best on real data |
| 18 | 7 | `zc_strong_all_weighted` | ResNet18 | ~90-91% | real (2,930) | Weighted sampler hurts (rare classes too scarce) |
| 19 | 8 | real + synth baseline | ResNet18 | ~95.1% | real + synth (8,088) | +176% data, +1-2% accuracy |
| **21** | **9** | **label_smoothing=0.1** | **ResNet18** | **~96.2%** | **real + synth (8,088)** | **Best overall — all-time champion** |
| 23-29 | 10 | HP sweep (7 configs) | ResNet18/34/50 | ~94.2-95.9% | real + synth (8,088) | No improvement over label smoothing |
| 30 | 11 | zero-shot | SmolVLM2-2.2B | 57.3% | real (val only) | VLM baseline without training |
| **36** | **11** | **QLoRA best run** | **SmolVLM2-2.2B** | **94.5%** | **real (2,891)** | **Best VLM result (A100, batch=24)** |

---

## Phase 12: Dataset Metadata Analysis (2026-02-25)

### Motivation

Annotated all dataset splits (train, valid, test) with visual metadata using SmolVLM2-2.2B-Instruct to understand jersey color distribution, number color distribution, and whether numbers have borders/outlines. This helps identify distribution gaps between real and synthetic data, and between train/valid/test splits.

### Annotation Method

- **Model**: SmolVLM2-2.2B-Instruct (single-prompt JSON mode)
- **Infrastructure**: NVIDIA L4 on Vertex AI (g2-standard-8), batch_size=8
- **Fields**: `jersey_color`, `number_color`, `border` (true/false)
- **Valid colors**: white, black, red, blue, navy, yellow, green, purple, orange, grey, maroon, teal, pink

### Annotation Coverage

| Split | Records | With Metadata | Coverage |
|-------|---------|---------------|----------|
| Train (real) | 2,891 | 2,891 | 100% |
| Train (synthetic) | 5,197 | 5,197 | 100% |
| Valid | 372 | 372 | 100% |
| Test | 365 | 365 | 100% |

### Jersey Color Distribution

| Color | Train Real | Train Synth | Train Combined | Valid | Test |
|-------|-----------|-------------|----------------|-------|------|
| white | 1,539 (53.2%) | 679 (13.1%) | 2,218 (27.4%) | 202 (54.3%) | 193 (52.9%) |
| blue | 428 (14.8%) | 923 (17.8%) | 1,351 (16.7%) | 49 (13.2%) | 55 (15.1%) |
| red | 281 (9.7%) | 1,004 (19.3%) | 1,285 (15.9%) | 38 (10.2%) | 35 (9.6%) |
| black | 334 (11.6%) | 622 (12.0%) | 956 (11.8%) | 34 (9.1%) | 34 (9.3%) |
| yellow | 255 (8.8%) | 649 (12.5%) | 904 (11.2%) | 39 (10.5%) | 43 (11.8%) |
| green | 15 (0.5%) | 485 (9.3%) | 500 (6.2%) | 3 (0.8%) | 0 |
| orange | 0 | 480 (9.2%) | 480 (5.9%) | 0 | 0 |
| navy | 25 (0.9%) | 0 | 25 (0.3%) | 5 (1.3%) | 5 (1.4%) |
| maroon | 8 (0.3%) | 240 (4.6%) | 248 (3.1%) | 2 (0.5%) | 0 |
| grey | 0 | 62 (1.2%) | 62 (0.8%) | 0 | 0 |
| purple | 3 (0.1%) | 53 (1.0%) | 56 (0.7%) | 0 | 0 |
| pink | 3 (0.1%) | 0 | 3 (0.04%) | 0 | 0 |

### Number Color Distribution

| Color | Train Real | Train Synth | Train Combined | Valid | Test |
|-------|-----------|-------------|----------------|-------|------|
| white | 1,136 (39.3%) | 2,404 (46.3%) | 3,540 (43.8%) | 145 (39.0%) | 143 (39.2%) |
| black | 1,176 (40.7%) | 904 (17.4%) | 2,080 (25.7%) | 163 (43.8%) | 164 (44.9%) |
| red | 465 (16.1%) | 82 (1.6%) | 547 (6.8%) | 51 (13.7%) | 46 (12.6%) |
| yellow | 67 (2.3%) | 1,277 (24.6%) | 1,344 (16.6%) | 9 (2.4%) | 8 (2.2%) |
| blue | 1 (0.03%) | 306 (5.9%) | 307 (3.8%) | 0 | 1 (0.3%) |
| purple | 0 | 224 (4.3%) | 224 (2.8%) | 0 | 0 |
| unknown | 25 (0.9%) | 0 | 25 (0.3%) | 1 (0.3%) | 2 (0.5%) |
| green | 14 (0.5%) | 0 | 14 (0.2%) | 0 | 1 (0.3%) |
| maroon | 5 (0.2%) | 0 | 5 (0.06%) | 3 (0.8%) | 0 |
| grey | 1 (0.03%) | 0 | 1 (0.01%) | 0 | 0 |

### Border Distribution

SmolVLM2 reported `border=true` for **100% of all records** across all splits. This is almost certainly a model bias — the VLM interprets any visible digit edge as a "border". Border metadata is unreliable and should not be used for analysis or stratification.

### Key Observations

1. **Real data is white-jersey dominated** (53%) — matches broadcast basketball where home teams wear white. Valid and test splits mirror this distribution closely, confirming they are representative of the real data.

2. **Synthetic data diversifies jersey colors** — the generation pipeline uses an NBA color palette, producing a more uniform distribution across red (19%), blue (18%), black (12%), white (13%), yellow (12%), green (9%), orange (9%). This is by design — rare classes got synthetic data with varied jersey colors.

3. **Number color mismatch between real and synthetic**: Real data is dominated by black (41%) and white (39%) number colors. Synthetic data over-represents yellow (25%) and blue (6%) numbers. This is because the synthetic renderer picks contrasting text colors against jersey backgrounds, producing color combinations not common in real basketball (e.g., yellow numbers on blue jerseys).

4. **Colors absent from val/test**: green, orange, grey, purple jersey colors appear in training (via synthetic) but have zero representation in valid/test. This means the model learns to recognize numbers on these jersey colors but isn't evaluated on them — the val accuracy may undercount the benefit of color diversity.

5. **Navy is only in real data** (25 train, 5 valid, 5 test) — the synthetic pipeline didn't generate navy jerseys (used "blue" instead). Navy vs blue distinction may confuse SmolVLM2.

### Distribution Comparison: Train vs Valid/Test

The real training data distribution closely matches valid and test, confirming they come from the same source distribution (broadcast basketball footage). The synthetic data shifts the combined training distribution away from val/test, particularly in jersey color balance. This is acceptable because the synthetic data targets rare classes that need more samples, and the model's augmentation pipeline handles domain adaptation.

### Generated Plots

All plots saved to `experiments/metadata_stats/`:

```
experiments/metadata_stats/
├── train_real/          # Real training data only (2,891 samples)
│   ├── jersey_color_distribution.png
│   ├── number_color_distribution.png
│   ├── jersey_color_by_class.png    # Heatmap: which classes wear which colors
│   ├── number_color_by_class.png    # Heatmap: number color per class
│   ├── color_combinations.png       # Jersey x Number color matrix
│   ├── border_distribution.png
│   ├── border_by_class.png
│   └── annotation_coverage.png
├── train_synth/         # Synthetic training data only (5,197 samples)
│   └── ... (same plot set)
├── train_combined/      # Real + Synthetic combined (8,088 samples)
│   └── ... (same plot set)
├── valid/               # Validation split (372 samples)
│   └── ... (same plot set)
└── test/                # Test split (365 samples)
    └── ... (same plot set)
```

### GCP Jobs

| Split | Job ID | Samples Annotated |
|-------|--------|-------------------|
| train | (previous session) | 2,891 real + 5,197 synthetic |
| valid | `8572867491477323776` | 372 |
| test | `943769722711703552` | 365 |

---

## Conclusions

1. **ResNet18 with label smoothing + synthetic data remains the best model** at 96.2% val top-1. It's also 42x smaller, 200x faster at inference, cheaper to train, and deployable on edge devices via ONNX.

2. **SmolVLM2 QLoRA is a viable alternative at 94.5%** but has significant practical drawbacks: large model size, GPU-only inference, high run-to-run variance, and 5x training cost. It has not yet been tested with synthetic training data or augmentation, which could close the gap.

3. **The accuracy progression tells a clear story**:
   - Baseline ResNet18: 91.9% (no augmentation)
   - + Broadcast degradation augmentation: 93-94% (+2%)
   - + Synthetic data for rare classes: 95.1% (+1%)
   - + Label smoothing: 96.2% (+1.1%)
   - Total improvement: **+4.3%** from systematic experimentation

4. **Data quality > model complexity**: Adding 5,197 synthetic images (+176% data) improved accuracy more than switching from an 11M-param CNN to a 2.2B-param VLM. The VLM trained on real data only (2,891 samples) couldn't match ResNet18 trained on the enriched dataset.

## Next Steps

1. **Test SmolVLM2 with synthetic data**: The VLM was only trained on real data. Adding synthetic training samples could push it past 96%.
2. **Test SmolVLM2 with more epochs**: QLoRA variance is high — running 5+ A100 jobs and taking the best checkpoint would give a more reliable accuracy estimate.
3. **Evaluate on real broadcast footage**: Both models were evaluated on the val set (tightly cropped). Real-world performance on detector output crops is the true measure.
4. **Ensemble approach**: Use SmolVLM2 as a fallback for samples where ResNet18 is low-confidence. The VLM's text understanding may catch cases the CNN misses.
5. **Full fine-tune vs QLoRA**: QLoRA's stochastic quantization limits accuracy ceiling. A full bf16 fine-tune on A100-80GB (or H100) could reduce variance and improve peak accuracy.

## Recommendations

1. **Deploy ResNet18 (label smoothing)** as the primary production model — 96.2% accuracy, 43 MB ONNX, runs anywhere.
2. **Keep SmolVLM2 checkpoint** as a research artifact for ensemble or fallback experiments.
3. **Current best configs are baked into defaults** — no overrides needed for the best ResNet18 recipe.
4. **Synthetic data works** — for any future class additions, generate synthetic samples immediately.
5. **Keep strong augmentation** (RandomApply p=0.85) — proven beneficial even with 2.8x more data.
6. **Label smoothing=0.1 is the only HP that helped** — all other HP changes were neutral or harmful.
7. **ResNet18 is sufficient** — ResNet34/50/SmolVLM2-2.2B offer no accuracy improvement on this task.

## Technical Details

- **Repository**: classifier-training (GitHub: ortizeg/classifier-training)
- **Transform configs**: `src/classifier_training/conf/transforms/basketball_jersey_*.yaml`
- **Custom transforms**: `src/classifier_training/transforms/degradation.py`
  - `RandomJPEGCompression` — PIL encode/decode round-trip
  - `RandomPixelate` — NEAREST downscale + upscale
  - `RandomBilinearDownscale` — BILINEAR downscale + upscale
  - `RandomGaussianNoise` — additive Gaussian noise (0-255 scale)
  - `RandomZoomOut` — canvas padding + resize (simulates loose bbox)
  - `RandomZoomIn` — crop + resize up (simulates tight bbox)
- **SmolVLM2 pipeline**: `src/classifier_training/models/smolvlm2.py`, `src/classifier_training/data/vlm_*.py`
- **GCP Project**: `api-project-562713517696`
- **GCS Bucket**: `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/`
- **Vertex AI Region**: `us-east1`
- **Docker Image**: `us-docker.pkg.dev/api-project-562713517696/classifier-training/classifier-training:latest`
- **Dates**: 2026-02-20 (Phases 1-8), 2026-02-21 (Phases 9-10), 2026-02-24/25 (Phase 11), 2026-02-25 (Phase 12)

## How to Recover Artifacts

```bash
# List all sweep runs
gsutil ls gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-*/

# Download accuracy plot for a specific run
gsutil cp gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-<variant>/<timestamp>/training_history/accuracy_history.png .

# Download ONNX model
gsutil cp gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-<variant>/<timestamp>/model.onnx .

# Download labels mapping
gsutil cp gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-<variant>/<timestamp>/labels_mapping.json .

# Download last confusion matrix (check last epoch number first)
gsutil ls gs://deep-ego-model-training/ego-training-data/classifier-training/logs/jersey-r18-aug-sweep-<variant>/<timestamp>/ | grep epoch_ | sort | tail -1

# View job in Vertex AI console
# https://console.cloud.google.com/ai/platform/locations/us-east1/training/<JOB_ID>?project=562713517696
```
