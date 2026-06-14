# Degradation-Aware Enhancement Routing for Object Detection

This repository contains an independent experimental pipeline for the paper **"Degradation-Aware Enhancement Routing for Object Detection"**. The project studies whether a lightweight degradation-aware router can select a better preprocessing action for object detection than applying one fixed enhancement method to every image.

The repository is designed for a 5,000-image COCO val2017 subset by default. It generates its own degraded image variants, runs its own detector experiments, computes its own oracle and routing metrics, and does not import result tables from the other degradation-aware detection papers.

## Paper

- **Title:** Degradation-Aware Enhancement Routing for Object Detection
- **Repository:** `enhancement-routing-detection`
- **Package:** `enhancement_routing`
- **Research area:** Computational Imaging / Image Processing
- **Target venue:** IEEE DG 2026
- **Status:** experimental repository scaffold

## Research Question

Can degradation-aware routing select a better preprocessing action for object detection than applying one fixed enhancement method to all images?

## Method Overview

The pipeline uses a controlled COCO val2017 subset and creates image conditions where preprocessing decisions matter:

- clean
- low-light
- low-light + noise
- low-light + JPEG compression
- blur
- contrast reduction

For each degraded image, the repository evaluates several preprocessing actions:

- none
- gamma correction
- CLAHE
- Retinex / MSRCR-style enhancement
- optional Zero-DCE wrapper
- reject / warn as a non-processing action

The main comparison is between fixed preprocessing policies, an oracle upper bound, a rule-based degradation-aware router, and an optional Random Forest router trained to imitate oracle actions.

## Repository Structure

```text
enhancement-routing-detection/
  README.md
  LICENSE
  CITATION.cff
  pyproject.toml
  requirements.txt
  .gitignore

  configs/
    experiment.yaml

  src/
    enhancement_routing/
      __init__.py
      data.py
      methods.py
      metrics.py
      plots.py
      utils.py

  scripts/
    prepare_data.py
    run_experiments.py
    evaluate.py
    make_figures.py

  results/
    tables/
    figures/

  .agent/
    README.md
    CODEX_PROMPT.md
    IMPLEMENTATION_CHECKLIST.md
    PAPER_NOTES.md
```

The public repository is intentionally compact. The `.agent/` folder is for implementation instructions and can be removed before public release if needed.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Install PyTorch separately if your CUDA setup requires a specific wheel.

## Data

Download COCO val2017 images and annotations manually. The repository does not redistribute COCO.

Expected local layout:

```text
data/raw/coco/
  val2017/
    000000000139.jpg
    ...
  annotations/
    instances_val2017.json
```

The default config uses 5,000 original COCO images. Degraded variants are generated inside `data/processed/`.

## Reproduce Experiments

Prepare the deterministic 5k subset and synthetic degradation variants:

```bash
python scripts/prepare_data.py --config configs/experiment.yaml
```

Run YOLO detector inference for each preprocessing action:

```bash
python scripts/run_experiments.py --config configs/experiment.yaml
```

Evaluate oracle actions, routing policies and policy-level metrics:

```bash
python scripts/evaluate.py --config configs/experiment.yaml
```

Create figures for the paper:

```bash
python scripts/make_figures.py --config configs/experiment.yaml
```

For a small smoke test, override the limit:

```bash
python scripts/prepare_data.py --config configs/experiment.yaml --limit 50
python scripts/run_experiments.py --config configs/experiment.yaml --limit 50
python scripts/evaluate.py --config configs/experiment.yaml
python scripts/make_figures.py --config configs/experiment.yaml
```

## Metrics

The main policy metrics are:

- mAP@50
- mAP@50:95
- recall
- mean confidence
- runtime overhead
- harmful enhancement rate
- oracle gap
- routing accuracy
- policy gain

Definitions used in this repository:

```text
Harmful Enhancement Rate = cases where preprocessing lowers detection score / all preprocessed cases
Oracle Gap = score(oracle) - score(router)
Policy Gain = score(router) - score(no preprocessing)
```

## Expected Outputs

```text
results/tables/dataset_summary.csv
results/tables/preprocessing_results.csv
results/tables/oracle_actions.csv
results/tables/router_predictions.csv
results/tables/policy_comparison.csv
results/tables/runtime_overhead.csv

results/figures/policy_comparison.png
results/figures/oracle_gap.png
results/figures/harmful_enhancement_cases.png
results/figures/routing_pipeline.png
```

## Literature and Background

This project is related to:

- COCO object detection evaluation.
- Common corruption and robustness evaluation for computer vision.
- Low-light enhancement methods such as gamma correction, CLAHE, Retinex and Zero-DCE.
- Task-aware image enhancement for downstream vision models.
- Studies showing that enhancement can improve or harm object detection depending on the degradation.

Full bibliography should be maintained in Zotero or Overleaf, not inside this repository.

## Citation

```bibtex
@software{issayev_2026_enhancement_routing_detection,
  author = {Issayev, Alen},
  title = {Degradation-Aware Enhancement Routing for Object Detection},
  year = {2026},
  url = {https://github.com/alenisaw/enhancement-routing-detection}
}
```

## License

This repository is released under the MIT License unless changed by the author before public release.
