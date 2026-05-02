# Paper-aligned release/reproduction bundle (2026-05-02)

Root: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502`

Use `summaries/CANONICAL_PAPER_TABLES_20260502.md` first. It supersedes stale files that report Ours violence as 81.7.

Key checks:
- Final Ours violence single-concept is hardlink-copied at `outputs/sd14_i2p_single/ours_best/violence/sh20_tau04_txt030_img010/` and verifies SR=88.3.
- Old/confusing violence sweeps moved to `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/_deprecated_confusing_violence_20260502` and old probe summaries moved to `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/_deprecated_confusing_violence_20260502`.
- This bundle uses hardlink copies (`cp -al`) to avoid duplicating disk usage while presenting a clean folder.

Important summaries:
- `summaries/CANONICAL_SD14_I2P_SINGLE_20260502.csv`
- `summaries/probe_ablation_i2p_q16_top60_7concept_UPDATED_violence88p3_20260502.md`
- `summaries/i2p_multi_sr_full_nr_tables_20260501.md`
- `summaries/TABLE1_NUDITY_BREAKDOWN_RELIABLE_HANDOFF_20260501.md`
- `summaries/WRITER_SGF_P4DN_AND_NFE_CONFIGS_20260502.md`
- `configs/OMC_NFE_PER_CONCEPT_CONFIGS_SD14_I2P_Q16_TOP60.md`
