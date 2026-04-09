# Repository Taxonomy

This repository is a research monorepo. It mixes active method development,
baseline integrations, historical experiment branches, and large generated
artifacts. Use the categories below to decide where to work first.

## Active Core

These folders are the canonical working set for current nudity-erasing
research and evaluation:

- `CAS_SpatialCFG/` — current training-free method line (`v4` through `v13`)
- `SAFREE/` — major baseline plus local extensions used in comparisons
- `vlm/` — shared evaluation, aggregation, and judge-analysis scripts
- `scripts/` — top-level orchestration across methods and datasets
- `docs/` — living methodology and evaluation notes

If you are starting new work, prefer to begin in this set unless you have a
clear reason to touch a baseline or archived branch.

## Secondary Baselines

These folders still matter for comparisons, but they are not the primary
development surface:

- `AMG/`
- `SoftDelete+CG/`

Treat them as baseline or comparative subsystems rather than the default place
for new method development.

## Legacy / Archive

These folders appear to contain older classifier families, earlier experiment
branches, or heavily duplicated variants:

- `3_classification_sd1.4/`
- `3_classification_sd1.4TC/`
- `3_classification_sd1.4_csv/`
- `4_classification_sd1.4/`
- `5_classificaiton/`
- `10_classificaiton/`
- `three_classificaiton/`
- `three_classificaiton_new/`
- `three_classificaiton_scale/`
- `sdxl-lightening-*`
- `sdxl-nag-3classification/`
- `z0_clf_guidance/`
- `dit_clf_guidance/`
- `rae_clf_guidance/`
- `repa_clf_guidance/`
- `rab_grid_search/`

Do not use these as the default starting point for new work unless the task is
explicitly about reproducing, comparing against, or extracting assets from a
legacy branch.

## External / Imported Projects

Some top-level folders behave more like imported or side projects than the
canonical core:

- `guided-diffusion/`
- `geodiffusion-features-code_refactor/`
- `Scale-RAE/`
- `Safe_Denoiser/`
- `Claude-Code-Usage-Monitor/`

Treat these as separate codebases with their own local concerns.

## Outputs and Artifacts

Large generated outputs are mixed into several top-level method folders,
especially image-heavy result trees under:

- `CAS_SpatialCFG/outputs/`
- `SAFREE/results/`
- `SoftDelete+CG/scg_outputs/`
- `logs/`
- `outputs/`

When making source changes, avoid treating these artifact directories as the
primary source of truth for code organization.

