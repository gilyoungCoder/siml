# Unlearning: Training-Free Unsafe Concept Removal from Text-to-Image Diffusion Models

Research on removing unsafe concepts (e.g., nudity) from T2I diffusion models without retraining.

## Core Approach
**Example-based When + Where Guidance** — deciding when (CAS) and where (Spatial) to apply safety guidance based on sample-level signals.

## Methods

| Folder | Description |
|--------|-------------|
| `CAS_SpatialCFG/` | CAS + Spatial CFG (SLD/DAG style, noise-based spatial guidance) |
| `AMG/` | Activation Matching Guidance (h-space feature matching) |
| `SAFREE/` | SAFREE baseline + extensions (spatial CG, dual classifier, multi-class) |
| `SoftDelete+CG/` | Soft deletion with classifier guidance |
| `SDErasure/` | Stable Diffusion concept erasure |
| `z0_clf_guidance/` | z0 classifier guidance |

## Classification Variants
- `3_classification_sd1.4/` — 3-class SD1.4
- `3_classification_sd1.4TC/` — 3-class SD1.4 with textual concepts
- `5_classificaiton/` — 5-class
- `10_classificaiton/` — 10-class

## Evaluation
- `vlm/` — NudeNet + Qwen3-VL evaluation scripts
- Metrics: Safety Rate (SR), NudeNet Unsafe%, COCO FP rate

## Datasets
4 nudity benchmarks: Ring-A-Bell, MMA, P4DN, UnlearnDiff + COCO (benign FP check)

## Model
`CompVis/stable-diffusion-v1-4` with `safety_checker=None`
