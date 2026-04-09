# Legacy Experiment Families

This document labels top-level folders that should usually be treated as legacy
 or archive branches rather than the default surface for new work.

## Legacy classifier families

- `3_classification_sd1.4/`
- `3_classification_sd1.4TC/`
- `3_classification_sd1.4_csv/`
- `4_classification_sd1.4/`
- `5_classificaiton/`
- `10_classificaiton/`
- `three_classificaiton/`
- `three_classificaiton_new/`
- `three_classificaiton_scale/`
- `three_classificaiton_Clip/`

These appear to contain earlier classifier pipelines and multiple copy-derived
 variants.

## Legacy SDXL / variant branches

- `sdxl-lightening-11classification/`
- `sdxl-lightening-11classificationNegativeLearning/`
- `sdxl-lightening-31classification/`
- `sdxl-lightening-3classification/`
- `sdxl-lightening-3classificationNegtiveLearning/`
- `sdxl-lightening-4classification/`
- `sdxl-lightening-5classification/`
- `sdxl-lightening-5classification_hier/`
- `sdxl-nag-3classification/`

These should not be the default starting point unless the task explicitly asks
for SDXL-family reproduction or comparison.

## Older guidance / analysis branches

- `z0_clf_guidance/`
- `dit_clf_guidance/`
- `rae_clf_guidance/`
- `repa_clf_guidance/`
- `rab_grid_search/`

Use these only when reproducing historical results, mining assets, or comparing
against older guidance ideas.

## How to work with legacy branches

If you must touch a legacy area:

1. confirm it is the intended target
2. avoid copying new code patterns back into the active core by default
3. prefer documenting the relationship to the active core in `docs/`

For current work, prefer the active core defined in `docs/repo-taxonomy.md`.

