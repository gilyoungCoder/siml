# Current Optional Artifact Prep Playbook

This note explains the non-GPU preparation state for concept-pack optional artifacts.

## Goal

For each concept pack under:

```text
docs/neurips_plan/multi_concept/concept_packs/<concept>/
```

we eventually want these optional artifacts:

- `exemplar_images/`
- `concept_directions.pt`
- `clip_patch_tokens.pt`
- `contrastive_embeddings.pt`

## What was fixed

The prep pipeline was normalized so it can now be driven pack-by-pack:

- `CAS_SpatialCFG/prepare_concept_subspace.py`
  - now supports:
    - `--target_prompt_file`
    - `--anchor_prompt_file`
    - `--target_image_prefix`
    - `--anchor_image_prefix`
- `CAS_SpatialCFG/prepare_clip_patch_tokens.py`
  - now supports:
    - `--target_prefix`
    - `--anchor_prefix`
- `CAS_SpatialCFG/prepare_contrastive_direction.py`
  - now supports:
    - `--target_prefix`
    - `--anchor_prefix`
- `CAS_SpatialCFG/prepare_multi_concept.py`
  - now passes pack-specific prompt files and image-prefix settings through the stack
  - supports `--ensure_exemplar_images`

## Recommended prefixes

The normalized pack-local convention is:

- target images: `target_00.png`, `target_01.png`, ...
- anchor images: `anchor_00.png`, `anchor_01.png`, ...

## Recommended commands

### Prepare only concept directions + exemplar images for all packs

```bash
python3 CAS_SpatialCFG/prepare_multi_concept.py \
  --all \
  --skip_clip \
  --skip_contrastive \
  --ensure_exemplar_images
```

### Prepare all optional artifacts for all packs

```bash
python3 CAS_SpatialCFG/prepare_multi_concept.py \
  --all \
  --ensure_exemplar_images
```

### Prepare one pack only

```bash
python3 CAS_SpatialCFG/prepare_multi_concept.py \
  --pack_dirs docs/neurips_plan/multi_concept/concept_packs/violence \
  --ensure_exemplar_images
```

## Current practical status

- All 7 expected packs are now text/metadata-complete.
- The optional artifact layer is still unprepared.
- No GPU jobs were launched in this cleanup pass; this document only makes the prep pipeline coherent and ready.

## Next operator expectation

When GPU capacity is available and it is safe to do so, run the prep pipeline above and then refresh:

- `docs/omc_reports/current_concept_pack_completeness.md`
- `docs/omc_reports/current_gridsearch_best_config_summary.md`
