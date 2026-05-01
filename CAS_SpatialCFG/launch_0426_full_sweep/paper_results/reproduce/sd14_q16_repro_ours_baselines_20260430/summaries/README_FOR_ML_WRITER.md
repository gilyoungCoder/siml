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
