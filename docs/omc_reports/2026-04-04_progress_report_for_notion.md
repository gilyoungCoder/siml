# 2026-04-04 Progress Report

## Executive summary
- Stabilized multi-concept plumbing so concept packs can be consumed by existing v14-v19 generator paths without GPU-side changes.
- Rebuilt result readers so legacy judge outputs (`results.txt`, `categories_qwen2_vl.json`) and canonical outputs are both readable.
- Added automated reporting for best configs, Pareto frontier, and concept-pack completeness.
- Normalized the active v15-v19 run scripts onto the shared env/result helper pattern.
- Filled the missing required `harassment` and `hate` concept-pack files.

## What was completed

### A. Multi-concept compatibility
- Updated `CAS_SpatialCFG/concept_pack_loader.py` to support dict-like access used by `generate_v14.py` ~ `generate_v19.py`.
- Added prompt/metadata passthrough and compatibility properties such as `target_concepts`, `anchor_concepts`, and `target_words`.

### B. Result reading and aggregation
- Updated `vlm/result_paths.py` and `scripts/lib/repo_env.sh` to recognize both canonical and legacy Qwen result filenames.
- Updated `vlm/opensource_vlm_nudity.py` to emit canonical nudity result filenames while keeping legacy compatibility for Qwen2-VL.
- Rebuilt `scripts/aggregate_nudity_results.py` and `CAS_SpatialCFG/scripts/analyze_results.py` so they recover SR / Relevant_SR from historical outputs.

### C. Script cleanup
- Updated `CAS_SpatialCFG/scripts/run_remaining_siml04.sh`, `run_v14.sh`, `run_v15.sh`, `run_v16.sh`, `run_v16_siml04.sh`, `run_v17.sh`, `run_v18.sh`, `run_v19.sh`, and `run_v19_siml04.sh` to use shared env/result helper conventions.
- Fixed prep-script argument mismatches in the v15/v16 launchers.

### D. Multi-concept pack completeness
- Added missing required files for `harassment` and `hate` concept packs:
  - `metadata.json`
  - `families.json`
  - `target_prompts.txt`
  - `anchor_prompts.txt`
  - `target_keywords_primary.txt`
  - `target_keywords_secondary.txt`
  - `anchor_keywords.txt`

### E. OMC-facing reporting
- Added `CAS_SpatialCFG/scripts/report_gridsearch_best_configs.py`
- Added `CAS_SpatialCFG/scripts/check_concept_pack_completeness.py`
- Refreshed handoff and report docs under `docs/omc_reports/` and `docs/omc_handoff_multiconcept_and_results.md`

## Verification
- `pytest -q` -> 39 passed
- `python3 -m py_compile` on edited Python files -> passed
- `bash -n` on edited shell scripts -> passed
- Direct readback of real existing outputs via:
  - `CAS_SpatialCFG/scripts/analyze_results.py`
  - `scripts/aggregate_nudity_results.py`
  - `CAS_SpatialCFG/scripts/report_gridsearch_best_configs.py`
  - `CAS_SpatialCFG/scripts/check_concept_pack_completeness.py`

## Current experiment readout

### Grid-search coverage
- v14: 72 configs, 72 with NudeNet, 67 with SR
- v15: 24 configs, 24 with NudeNet, 21 with SR
- v16: 0 configs
- v17: 144 configs, 144 with NudeNet, 117 with SR
- v18: 218 configs, 218 with NudeNet, 0 with SR
- v19: 0 configs

### Best configs by version
- v14
  - Best NN: `image_dag_adaptive_ss5.0_st0.2` -> NN 3.2 / SR 13.6
  - Best SR: `both_dag_adaptive_ss2.0_st0.2` -> NN 15.8 / SR 69.0
  - Best balanced: `both_dag_adaptive_ss3.0_st0.4` -> NN 11.1 / SR 66.8
- v15
  - Best NN: `text_dag_adaptive_ss5.0_st0.2_np16` -> NN 3.8 / SR 27.2
  - Best SR: `text_dag_adaptive_ss3.0_st0.4_np16` -> NN 19.9 / SR 72.8
  - Best balanced: `text_dag_adaptive_ss5.0_st0.4_np16` -> NN 13.0 / SR 70.6
- v17
  - Best NN: `both_dag_adaptive_ss5.0_st0.2_fused` -> NN 2.9
  - Best SR: `image_hybrid_ss5.0_st0.3_fused` -> NN 28.5 / SR 77.8
  - Best balanced: `image_hybrid_ss5.0_st0.2_fused` -> NN 27.9 / SR 76.6
- v18
  - Best NN: `both_dag_adaptive_ss5.0_st0.2_none_sb0.5` -> NN 2.9
  - SR currently unreadable because judge outputs are absent, not because the reader is broken

### Global Pareto frontier
Current frontier now contains 14 configs, mostly from v14, v15, and v17.

## Multi-concept pack status
- Required text/metadata files now exist for all 7 expected packs:
  - `sexual`, `violence`, `shocking`, `self-harm`, `illegal_activity`, `harassment`, `hate`
- Remaining optional artifact gap for every pack:
  - `concept_directions.pt`
  - `clip_patch_tokens.pt`
  - `contrastive_embeddings.pt`
  - `exemplar_images/`

## Main risks / limitations
- Missing SR in some versions means missing judge outputs, not failed parsing.
- Multi-concept work was stabilized at the plumbing layer; no new GPU generation was triggered in this pass.
- The remaining multi-concept gap is optional image/precomputed artifact preparation, not required-file completeness.

## Repo artifacts produced for OMC
- `docs/omc_handoff_multiconcept_and_results.md`
- `docs/omc_reports/current_gridsearch_best_config_summary.md`
- `docs/omc_reports/current_gridsearch_config_index.csv`
- `docs/omc_reports/current_gridsearch_pareto_frontier.csv`
- `docs/omc_reports/current_gridsearch_pareto_frontier.md`
- `docs/omc_reports/current_concept_pack_completeness.md`
- `docs/omc_reports/current_concept_pack_completeness.json`
- `docs/omc_reports/2026-04-04_progress_report_for_notion.md`
- `docs/omc_reports/2026-04-04_notion_publish_receipt.md`

## Recommended next steps
1. Prepare optional image / precomputed artifacts for the 7 concept packs.
2. When running experiments finish producing judge outputs, rerun the reporting scripts to refresh SR / Pareto summaries.
3. Use the refreshed reports to choose the next multi-concept pilot configs.
