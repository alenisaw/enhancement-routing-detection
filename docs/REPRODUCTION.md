# Technical Reproduction & Pipeline Execution Guide

> This guide covers the full environment setup, data acquisition, pipeline execution, policy evaluation, and figure generation for the experiment supporting **"Adaptive Preprocessing Routing for Object Detection under Image Degradations"** (IEEE DG 2026). All commands are run from the repository root.

## Table of Contents

- [1. Complete Directory Layout](#1-complete-directory-layout)
- [2. Production Environment Setup](#2-production-environment-setup)
- [3. Data Acquisition & Splitting](#3-data-acquisition--splitting)
- [4. Execution Pipeline & Model Inferences](#4-execution-pipeline--model-inferences)
- [5. Diagnostics & Asset Plotting](#5-diagnostics--asset-plotting)

---

## 1. Complete Directory Layout

```text
enhancement-routing-detection/
├── README.md                            ← Research-oriented landing page (preprint showcase)
├── LICENSE                              ← MIT License
├── CITATION.cff                         ← Machine-readable citation metadata
├── pyproject.toml                       ← Package build configuration (setuptools)
├── requirements.txt                     ← Pinned Python dependency list
├── .gitignore
│
├── configs/
│   └── experiment.yaml                  ← Single source of truth for all pipeline parameters
│
├── docs/
│   ├── REPRODUCTION.md                  ← This file
│   └── assets/
│       ├── condition_action_delta.png   ← Fig. 1: condition-action mAP50-95 change matrix
│       └── oracle_rf_distributions.png  ← Fig. 2: Oracle vs. RF routing distributions
│
├── src/
│   └── enhancement_routing/
│       ├── __init__.py
│       ├── data.py                      ← Degradation synthesis and manifest generation
│       ├── methods.py                   ← No-reference feature extraction and YOLO inference
│       ├── metrics.py                   ← COCO evaluation, oracle selection, policy comparison
│       ├── plots.py                     ← Figure rendering
│       └── utils.py                     ← Config loading, CSV/JSON I/O helpers
│
├── scripts/
│   ├── download_coco_subset.py          ← Stage 1: data acquisition (stdlib only)
│   ├── prepare_data.py                  ← Stage 2: degradation synthesis and manifest
│   ├── run_experiments.py               ← Stage 3: feature extraction + detector inference
│   ├── evaluate.py                      ← Stage 4: policy evaluation and router training
│   └── make_figures.py                  ← Stage 5: paper figure generation
│
├── results/
│   ├── tables/                          ← Generated CSV evaluation tables (gitignored)
│   ├── figures/                         ← Generated PNG figures (gitignored)
│   └── models/                          ← Serialized Random Forest router (gitignored)
│
└── .agent/                              ← Internal implementation notes (remove before release)
    ├── README.md
    ├── CODEX_PROMPT.md
    ├── IMPLEMENTATION_CHECKLIST.md
    └── PAPER_NOTES.md
```

> **Note on `results/`.** All generated artifacts under `results/` are listed in `.gitignore` and are not committed to the repository. They are reproduced locally by running the five pipeline stages described below. The `.gitkeep` placeholder files that maintain the directory structure are tracked.

> **Note on `.agent/`.** This folder contains paper-drafting guidance and implementation checklists for the development authors. It can be deleted entirely before public release without affecting the experimental pipeline.

---

## 2. Production Environment Setup

### Prerequisites

| Requirement | Version |
| :--- | :--- |
| Python | 3.10 or higher |
| pip | 22+ |
| GPU (optional) | CUDA-capable; the pipeline runs on CPU if no GPU is available |

### Step 1 — Create and activate a virtual environment

```bash
# Create
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\activate
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the following packages:

| Package | Minimum version | Purpose |
| :--- | :---: | :--- |
| `numpy` | 1.24 | Array operations |
| `pandas` | 2.0 | Manifest and result tables |
| `opencv-python` | 4.8 | Image I/O, no-reference feature extraction |
| `Pillow` | 10.0 | Auxiliary image utilities |
| `PyYAML` | 6.0 | Config file parsing |
| `tqdm` | 4.66 | Progress bars |
| `matplotlib` | 3.7 | Figure rendering |
| `scikit-learn` | 1.3 | Random Forest router training |
| `ultralytics` | 8.2 | YOLOv8n inference |
| `pycocotools` | 2.0 | COCO metric computation |
| `joblib` | 1.3 | Parallel feature extraction with `return_as="generator"` |

### Step 3 — Install PyTorch

Install PyTorch separately to match your CUDA version. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct wheel. CPU-only build (always works, slower inference):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify the setup

```bash
python -c "import cv2, ultralytics, sklearn, joblib; print('Setup OK')"
```

---

## 3. Data Acquisition & Splitting

### Stage 1 — Download COCO val2017

`scripts/download_coco_subset.py` fetches COCO val2017 images (~6.4 GB) and `instances_val2017.json` (~241 MB) using only the Python standard library — no `wget`, `curl`, or external download tool is required. The script is **idempotent**: already-present files are skipped on re-runs. Zip archives are cached in `data/raw/coco/.cache/` and can be safely deleted after extraction to reclaim disk space.

```bash
# Full download — ~6.4 GB images + ~241 MB annotations
python scripts/download_coco_subset.py

# Annotations only — skip the 6.4 GB image archive (useful before committing to full download)
python scripts/download_coco_subset.py --skip-images

# Smoke test — extract only 50 images from the archive
python scripts/download_coco_subset.py --limit 50

# Custom storage root
python scripts/download_coco_subset.py --dest /mnt/data/coco
```

#### CLI Reference — `download_coco_subset.py`

| Flag | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--dest` | `str` | `data/raw/coco` | Root directory for COCO data; must match `paths.coco_images_dir` in `configs/experiment.yaml` |
| `--limit` | `int` | `None` | Extract only the first N images from the archive; all annotation entries are still extracted |
| `--skip-images` | flag | `False` | Download the annotations zip only; skip the 6.4 GB image archive entirely |

#### Expected Layout After Download

```text
data/raw/coco/
├── val2017/
│   ├── 000000000139.jpg
│   └── ...                    ← 5,000 JPEG images (or N if --limit was used)
├── annotations/
│   └── instances_val2017.json ← Required annotation file
└── .cache/                    ← Zip archives — safe to delete after extraction
```

---

### Stage 2 — Generate Degraded Image Variants

`scripts/prepare_data.py` selects the configured COCO subset (default: 5,000 images, filtered to 4,952 after requiring usable annotations), applies six controlled degradations to each original image in-place under `data/processed/`, and writes the manifest CSV that all downstream stages use.

```bash
# Full preparation — 5,000-image subset (produces 29,712 variants)
python scripts/prepare_data.py --config configs/experiment.yaml

# Smoke test — limit to 20 original images (produces 120 variants)
python scripts/prepare_data.py --config configs/experiment.yaml --limit 20
```

#### CLI Reference — `prepare_data.py`

| Flag | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Path to the experiment YAML configuration file |
| `--limit` | `int` | `None` | Override `dataset.subset_size` from the config; caps the number of original images processed |

#### Degradation Presets (from `configs/experiment.yaml`)

| Condition | Transform | Key Parameters |
| :--- | :--- | :--- |
| `clean` | None (copy) | — |
| `low_light` | Intensity scale + gamma | `factor=0.38`, `gamma=1.65` |
| `low_light_noise` | Low-light + additive Gaussian noise | `factor=0.38`, `gamma=1.65`, `noise_sigma=13.0` |
| `low_light_jpeg` | Low-light + JPEG re-encode | `factor=0.38`, `gamma=1.65`, `jpeg_quality=24` |
| `blur` | Gaussian blur | `kernel_size=7`, `sigma=1.8` |
| `contrast_reduction` | Linear contrast scale + shift | `alpha=0.55`, `beta=22` (clipped to valid range) |

**Primary output:** `data/processed/coco_enhancement_routing/manifest.csv` — one row per image-condition pair (29,712 rows for the full 4,952-image retained subset), recording `image_id`, `file_name`, `condition`, and `image_path`.

---

## 4. Execution Pipeline & Model Inferences

### Stage 3 — No-Reference Feature Extraction + Detector Inference

`scripts/run_experiments.py` performs two sequential operations:

**3a. No-reference feature extraction** — computes 14 quality statistics per image variant using OpenCV (Laplacian, Sobel, Canny, histogram CDF, HSV saturation, Gaussian residual, local variance, blockiness proxy). Runs in parallel across all available CPU cores using `joblib.Parallel(n_jobs=-1, return_as="generator")`. Progress is **checkpointed to `results/tables/quality_features.csv` every 500 images**: if the run is interrupted (e.g., by Ctrl+C or a system event), re-running the exact same command resumes automatically — already-processed `image_id` entries are detected and skipped.

**3b. YOLO inference** — runs YOLOv8n with pretrained COCO weights (downloaded automatically on first use) over every image-condition-action combination defined in `configs/experiment.yaml`. Stores per-detection results in `results/tables/detections.csv` (~2.6 GB for the full run).

```bash
# Full run — feature extraction + all detector inference
python scripts/run_experiments.py --config configs/experiment.yaml

# Feature extraction only — skip YOLO inference (fast; useful for testing routing logic)
python scripts/run_experiments.py --config configs/experiment.yaml --skip-detector

# Smoke test — limit manifest rows (combine with --limit on prepare_data.py)
python scripts/run_experiments.py --config configs/experiment.yaml --limit 20
```

#### CLI Reference — `run_experiments.py`

| Flag | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Path to the experiment YAML configuration file |
| `--limit` | `int` | `None` | Limit manifest to the first N rows; for smoke tests only |
| `--skip-detector` | flag | `False` | Compute and cache the quality feature table only; skip all YOLO inference |

#### Preprocessing Action Presets (from `configs/experiment.yaml:preprocessing`)

These parameters are fixed before evaluation and are not tuned per image. They are applied during YOLO inference for every image-condition variant.

| Action | Transform | Key Parameters |
| :--- | :--- | :--- |
| `none` | Identity — input passed through unchanged | — |
| `gamma` | Gamma correction | γ = 0.62 |
| `clahe` | CLAHE on luminance channel (LAB color space) | `clip_limit=2.0`, tile grid 8×8 |
| `retinex` | Multi-scale Retinex-like transform | Gaussian scales = [15, 80, 250]; per-channel 1st/99th percentile normalization |

> These preprocessing actions are distinct from the degradation presets in Stage 2. Stage 2 conditions describe how the *input image was corrupted*; Stage 3 actions describe what the *routing policy applies* before the detector runs.

#### No-Reference Features (14 total, extracted per `image_id`)

| Group | Features |
| :--- | :--- |
| Intensity statistics | `mean_intensity`, `median_intensity`, `dark_pixel_ratio`, `rms_contrast`, `histogram_spread` |
| Sharpness / frequency | `laplacian_variance`, `tenengrad`, `edge_density`, `gradient_magnitude` |
| Noise / artifact | `noise_estimate`, `local_variance_estimate`, `blockiness_proxy` |
| Color (HSV) | `saturation_mean`, `saturation_std` |

> **Checkpoint behaviour.** If `results/tables/quality_features.csv` already exists when the script starts, the feature extractor loads it, reads the set of `image_id` values already present, and processes only the remaining images. No recomputation occurs for finished images.

#### Detector Configuration (`configs/experiment.yaml:detector`)

| Parameter | Value |
| :--- | :--- |
| Model | `yolov8n` (pretrained COCO weights, auto-downloaded to `runs/`) |
| Input image size | 640 px |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.7 |
| Batch size | 8 |
| Max detections per image | 300 |
| Device | `auto` (CUDA GPU if available, else CPU) |

---

### Stage 4 — Policy Evaluation & Router Training

`scripts/evaluate.py` reads `results/tables/quality_features.csv` and `results/tables/detections.csv` produced by Stage 3 and computes the complete evaluation suite in the following order:

1. **COCO metrics per condition-action group** — mAP50, mAP50:95, mAP75, AP-small/medium/large, recall, mean confidence using `pycocotools`; intermediate JSON prediction files are written to `results/tables/coco_eval_json/` (gitignored).
2. **Per-image recall at IoU 0.50** — true positives matched against ground-truth annotations for every image-condition-action triple; used as the primary policy score throughout.
3. **Oracle actions** — hindsight-optimal action $a^*(i,c) = \arg\max_{a} S(i,c,a)$ per image-condition pair.
4. **Rule-based router predictions** — interpretable threshold policy over the 14 no-reference features (brightness, blur, noise, contrast, blockiness thresholds; see `src/enhancement_routing/methods.py:rule_based_router`).
5. **Random Forest router** — trained on (features → `oracle_action`) pairs using grouped shuffle split (train fraction: 0.75), 200 trees, balanced class weights, all CPU cores, random state 42; serialized to `results/models/random_forest_router.joblib`.
6. **Policy comparison table** — all seven policies evaluated on mean score, policy gain, oracle gap, harmful preprocessing rate, routing accuracy, and runtime.

```bash
python scripts/evaluate.py --config configs/experiment.yaml
```

#### CLI Reference — `evaluate.py`

| Flag | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Path to the experiment YAML configuration file |

#### Output Files

| File | Description |
| :--- | :--- |
| `results/tables/preprocessing_results.csv` | COCO metrics (mAP50, mAP50:95, recall, …) per condition-action group |
| `results/tables/per_image_scores.csv` | Per-image recall at IoU 0.50 for every image-condition-action triple |
| `results/tables/oracle_actions.csv` | Oracle-best action and score per image-condition pair |
| `results/tables/router_predictions.csv` | Rule-based and Random Forest routing decisions per image-condition pair |
| `results/tables/policy_comparison.csv` | Aggregated policy metrics — Table II in the paper |
| `results/tables/runtime_overhead.csv` | Per-action runtime statistics: mean, median, std in ms |
| `results/models/random_forest_router.joblib` | Serialized trained Random Forest classifier (scikit-learn) |

---

## 5. Diagnostics & Asset Plotting

### Stage 5 — Generate Diagnostic Figures

`scripts/make_figures.py` reads `results/tables/policy_comparison.csv` and renders four supplementary diagnostic figures to `results/figures/` (configured by `paths.figures_dir` in `configs/experiment.yaml`).

```bash
python scripts/make_figures.py --config configs/experiment.yaml
```

#### CLI Reference — `make_figures.py`

| Flag | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Path to the experiment YAML configuration file |

#### Generated Figures

| Output file | Description |
| :--- | :--- |
| `results/figures/policy_comparison.png` | Horizontal bar chart — mean per-image recall@0.50 per policy, sorted ascending |
| `results/figures/oracle_gap.png` | Horizontal bar chart — oracle gap (distance from oracle) per deployable policy |
| `results/figures/harmful_enhancement_cases.png` | Horizontal bar chart — harmful preprocessing rate per policy |
| `results/figures/routing_pipeline.png` | Text-box flow diagram — routing pipeline: input → features → policy → action → detector → metrics |

> **Note on paper figures.** The two figures embedded in `README.md` (the condition-action mAP50–95 change heatmap and the Oracle vs. RF stacked bar distributions by condition) are **not** outputs of `make_figures.py`. They are placed manually into `docs/assets/` — see the section below.

---

### Placing Paper Figures into `docs/assets/`

The two figures displayed in `README.md` — the condition-action mAP50–95 change heatmap (Fig. 1) and the Oracle vs. RF routing distributions (Fig. 2) — must be placed into `docs/assets/` manually. They are not produced by `make_figures.py`.

**Expected filenames** (must match the paths referenced in `README.md`):

```
docs/assets/Condition-action delta vs. no preprocessing.png   ← Fig. 1 heatmap
docs/assets/Oracle and RF action distributions.png            ← Fig. 2 stacked bars
```

Copy these from your saved figure screenshots or re-export them from the source that produced them.

---

### Full Reproduction — All Five Stages

```bash
# Stage 1 — Download COCO val2017 (~6.6 GB total)
python scripts/download_coco_subset.py

# Stage 2 — Generate 29,712 degraded image variants
python scripts/prepare_data.py --config configs/experiment.yaml

# Stage 3 — Extract 14 no-reference features + run YOLOv8n inference
#           (parallel; checkpoints to results/tables/quality_features.csv every 500 images)
python scripts/run_experiments.py --config configs/experiment.yaml

# Stage 4 — Evaluate all policies; train and serialize the Random Forest router
python scripts/evaluate.py --config configs/experiment.yaml

# Stage 5 — Render diagnostic figures
python scripts/make_figures.py --config configs/experiment.yaml

# Place paper figures manually into docs/assets/ (see §5 for required filenames)
```

---

### Smoke Test — End-to-End in Minutes

For a quick pipeline validation that completes in under five minutes (no GPU required):

```bash
python scripts/download_coco_subset.py --limit 20
python scripts/prepare_data.py --config configs/experiment.yaml --limit 20
python scripts/run_experiments.py --config configs/experiment.yaml --limit 20
python scripts/evaluate.py --config configs/experiment.yaml
python scripts/make_figures.py --config configs/experiment.yaml
```

> With `--limit 20`, the manifest contains 20 × 6 = 120 image-condition variants. All five pipeline stages execute without error. Expect reduced Random Forest routing accuracy (small training sample), but `results/tables/policy_comparison.csv` will be fully populated and figures will render correctly.
