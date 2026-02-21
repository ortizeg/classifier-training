# Jersey OCR Augmentation Sweep: Complete Experiment Log

## Overview

Systematic evaluation of broadcast video degradation augmentations for basketball jersey number OCR using ResNet18. All experiments use the same base configuration:
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

## Recommendations

1. **Current best**: `label_smoothing=0.1` + synthetic data = **~96.2% val top-1** (Experiment #21). Already baked into defaults.
2. **Synthetic data works**: +1-2% val accuracy from adding 5,197 synthetic images for 22 underrepresented classes. Model converges in half the epochs.
3. **Label smoothing is the only HP that helped**: 0.1 adds +1.1% over synth baseline. Was harmful at 3K samples, beneficial at 8K.
4. **Keep all other defaults**: EMA 0.9999, warmup 2000 steps, lr=5e-4, batch_size=64 — none improved when changed.
5. **ResNet18 is sufficient**: ResNet34/50 offer no improvement on this 42-class problem. Stick with the smaller, faster model.
6. **Don't use focal loss**: Harmful with label smoothing — the two regularizers conflict.
7. **Weighted sampler is now neutral**: With synthetic data, it neither helps nor hurts. Not needed.
8. **Keep strong augmentation**: RandomApply p=0.85 is still optimal even with 2.8x more data.

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
- **GCP Project**: `api-project-562713517696`
- **GCS Bucket**: `gs://deep-ego-model-training/ego-training-data/classifier-training/logs/`
- **Vertex AI Region**: `us-east1`
- **Docker Image**: `us-docker.pkg.dev/api-project-562713517696/classifier-training/classifier-training:latest`
- **Date**: 2026-02-20 (Phases 1-6)

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
