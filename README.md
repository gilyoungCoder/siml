# Unlearning: Training-Free Unsafe Concept Removal from Text-to-Image Diffusion Models

Research on removing unsafe concepts (e.g., nudity) from T2I diffusion models without retraining.

For navigation, start with:
- `docs/repo-taxonomy.md` — active vs baseline vs legacy folders
- `docs/repo-layout.md` — quick orientation to the canonical workflow
- `docs/active-workflow.md` — canonical current edit/generate/evaluate flow
- `docs/runtime-config.md` — repo/env path configuration for active scripts
- `docs/output-storage.md` — artifact hygiene and storage guidance
- `docs/archive/legacy-experiments.md` — labeled legacy experiment families

## Core Approach
**Example-based When + Where Guidance** — deciding when (CAS) and where (Spatial) to apply safety guidance based on sample-level signals.

## Active Core

The current canonical working set is:

- `CAS_SpatialCFG/` — current training-free method line
- `SAFREE/` — baseline plus local extensions
- `vlm/` — evaluation and aggregation
- `scripts/` — multi-method orchestration
- `docs/` — living methodology notes

## Methods

| Folder | Description |
|--------|-------------|
| `CAS_SpatialCFG/` | CAS + Spatial CFG (SLD/DAG style, noise-based spatial guidance) |
| `AMG/` | Activation Matching Guidance (h-space feature matching) |
| `SAFREE/` | SAFREE baseline + extensions (spatial CG, dual classifier, multi-class) |
| `SoftDelete+CG/` | Soft deletion with classifier guidance |
| `SDErasure/` | Stable Diffusion concept erasure |
| `z0_clf_guidance/` | z0 classifier guidance |

## Secondary Baselines

- `AMG/`
- `SoftDelete+CG/`

## Classification Variants
- `3_classification_sd1.4/` — 3-class SD1.4
- `3_classification_sd1.4TC/` — 3-class SD1.4 with textual concepts
- `5_classificaiton/` — 5-class
- `10_classificaiton/` — 10-class

These and many other top-level classifier/SDXL branches should generally be
treated as legacy or archived experiment families unless your task explicitly
targets them.

## Evaluation
- `vlm/` — NudeNet + Qwen3-VL evaluation scripts
- Metrics: Safety Rate (SR), NudeNet Unsafe%, COCO FP rate

## Datasets
4 nudity benchmarks: Ring-A-Bell, MMA, P4DN, UnlearnDiff + COCO (benign FP check)

## Model
`CompVis/stable-diffusion-v1-4` with `safety_checker=None`
