# Read this first: final paper-writer handoff

Use this file for paper updates:

- `MLWRITER_FINAL_TABLE_VALUES_20260501.md`

Auxiliary focused tables:

- `i2p_multi_sr_full_nr_tables_20260501.md`
- `CROSSBACKBONE_MJA_TABLE4_WITH_SAFEDENOISER_SGF_20260501.md`
- `CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md`

Do **not** use deprecated files under `_deprecated_do_not_use/`. In particular, older notes referring to `phase_safree_v2` used a mismatched prompt split for SAFREE multi-concept and were archived to prevent confusion.

Current final SAFREE multi source:

- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_safree_multi_q16top60`

Current final reviewer-ready code/repro candidate:

- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE`

Main convention:

- I2P uses q16 top-60 per concept, seed 42, one image per prompt.
- VLM metric is Qwen3-VL v5; SR = Safe + Partial.
- Multi-concept compact cells are SR / Full / NR.

- `MLWRITER_BESTMODE_FINAL_WITH_CONFIGS_20260501.md` — best-of-mode/best-config values, per-class cells where available, config/result paths.

- `CROSSBACKBONE_OURS_SR_SOURCE_AND_TABLE_UPDATE_20260501.md` — source audit for Ours SD3/FLUX/MJA SR values and exact table-update guidance.

- `CROSSBACKBONE_OURS_I2P_IMAGE_BACKED_BREAKDOWN_20260501.md` — image-backed Ours SD3/FLUX I2P q16 top-60 full breakdown with image/result/args paths.
