# ML writer final best-mode/best-config handoff (2026-05-01)
This is the writer-facing source for **best-of-mode / best-of-config** reporting. Metric cell format is **SR / Safe / Partial / Full / NR** (%). SR = Safe + Partial.

## Rules to use in paper
- Ours may use **best-of-mode** (anchor vs hybrid) and per-concept tuned best config. This is the intended final reporting rule.
- I2P split is **q16 top-60**, 60 prompts per concept, seed 42.
- SD1.4 Ours configs below are verified from actual `results_qwen3_vl_*_v5.txt` files and config JSONs.
- Cross-backbone Ours has verified SR/config summaries, but raw Ours per-class VLM files were not found in the current result tree; Safe/Partial/Full/NR stay `TBD(raw not found)` there. Official baselines have full breakdown in `CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md`.
- Machine-readable CSV: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/mlwriter_bestmode_final_with_configs_20260501.csv`

## SD1.4 I2P q16 top-60 — Ours best config with full breakdown
| Concept | Cell SR/Safe/Partial/Full/NR | Best variant | Key args | Prompt path | Result path | Family pack |
|---|---:|---|---|---|---|---|
| sexual | 98.3/91.7/6.7/1.7/0.0 | `hybrid_best_tau05_cas0.5` | `mode=hybrid, probe=both, family=True, ss=20.0, cas=0.5, th_text=0.1, th_img=0.3` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/sexual_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_tune/i2p_q16/sexual/hybrid_best_tau05_cas0.5/results_qwen3_vl_nudity_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt` |
| violence | 88.3/73.3/15.0/5.0/6.7 | `sh20_tau04_txt030_img010` | `mode=hybrid, probe=both, family=True, ss=20.0, cas=0.4, th_text=0.30, th_img=0.10` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/violence_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260501_both_violence_adaptive_clean005/sh20_tau04_txt030_img010/results_qwen3_vl_violence_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt` |
| self-harm | 51.7/8.3/43.3/23.3/25.0 | `hybrid_best_tau05_cas0.5` | `mode=hybrid, probe=both, family=True, ss=7.0, cas=0.5, th_text=0.1, th_img=0.1` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/self-harm_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_tune/i2p_q16/self-harm/hybrid_best_tau05_cas0.5/results_qwen3_vl_self_harm_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt` |
| shocking | 93.3/88.3/5.0/3.3/3.3 | `hybrid_best_ss125_ss27.5` | `mode=hybrid, probe=both, family=True, ss=27.5, cas=0.6, th_text=0.15, th_img=0.1` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/shocking_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_tune/i2p_q16/shocking/hybrid_best_ss125_ss27.5/results_qwen3_vl_shocking_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/shocking/clip_grouped.pt` |
| illegal_activity | 46.7/31.7/15.0/15.0/38.3 | `hybrid_best_ss125_ss25.0` | `mode=hybrid, probe=both, family=True, ss=25.0, cas=0.6, th_text=0.1, th_img=0.5` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_tune/i2p_q16/illegal_activity/hybrid_best_ss125_ss25.0/results_qwen3_vl_illegal_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/illegal_activity/clip_grouped.pt` |
| harassment | 68.3/56.7/11.7/13.3/18.3 | `hybrid_best_ss125_ss31.25` | `mode=hybrid, probe=both, family=True, ss=31.25, cas=0.5, th_text=0.1, th_img=0.5` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/harassment_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_tune/i2p_q16/harassment/hybrid_best_ss125_ss31.25/results_qwen3_vl_harassment_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/harassment/clip_grouped.pt` |
| hate | 73.3/56.7/16.7/10.0/16.7 | `hybrid_best_img075_img0.0375` | `mode=hybrid, probe=both, family=True, ss=28.0, cas=0.6, th_text=0.25, th_img=0.0375` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/hate_q16_top60.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_tune/i2p_q16/hate/hybrid_best_img075_img0.0375/results_qwen3_vl_hate_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt` |

**Avg:** SR 74.2, Safe 58.1, Partial 16.2, Full 10.2, NR 15.5.

## Cross-backbone I2P q16 top-60 — Ours best config/SR

### SD3
| Concept | SR | Best config label | Prompt path | Result/source |
|---|---:|---|---|---|
| sexual | 96.7 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `$CAS/prompts/i2p_q16_top60/sexual_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| violence | 73.3 | `sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/violence_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| self-harm | 43.3 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `$CAS/prompts/i2p_q16_top60/self-harm_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| shocking | 90.0 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `$CAS/prompts/i2p_q16_top60/shocking_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| illegal_activity | 50.0 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `$CAS/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| harassment | 36.7 | `sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/harassment_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| hate | 56.7 | `sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/hate_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |

**Avg SR:** 63.8. Full breakdown TBD(raw not found).

### FLUX1
| Concept | SR | Best config label | Prompt path | Result/source |
|---|---:|---|---|---|
| sexual | 100.0 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/sexual_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| violence | 60.0 | `flux_xlow_hybrid_ss0.25_thr0.15_imgthr0.1_cas0.45_both` | `$CAS/prompts/i2p_q16_top60/violence_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| self-harm | 65.0 | `flux_low_hybrid_ss0.75_thr0.15_imgthr0.1_cas0.45_both` | `$CAS/prompts/i2p_q16_top60/self-harm_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| shocking | 100.0 | `flux_high_hybrid_ss2.5_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/shocking_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| illegal_activity | 60.0 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| harassment | 68.3 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/harassment_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |
| hate | 80.0 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `$CAS/prompts/i2p_q16_top60/hate_q16_top60.txt` | `SR from crossbackbone_sd3_flux1_i2p_status_20260501.md; raw per-class file not found` |

**Avg SR:** 76.2. Full breakdown TBD(raw not found).

## Cross-backbone MJA — Ours best-of-mode config/SR

### SD3
| Concept | SR | Best mode | Hyperparameters | Result/source |
|---|---:|---|---|---|
| sexual | 84.0 | `hybrid` | `sh=15 tau=0.6 theta_text=0.10 theta_img=0.30` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |
| violence | 58.0 | `anchor` | `sa=1.5 tau=0.6 theta_text=0.10 theta_img=0.20` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |
| illegal | 67.0 | `hybrid` | `sh=20 tau=0.3 theta_text=0.15 theta_img=0.10` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |
| disturbing | 90.0 | `hybrid` | `sh=20 tau=0.4 theta_text=0.15 theta_img=0.10` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |

**Avg SR:** 74.8. Full breakdown TBD(raw not found).

### FLUX1
| Concept | SR | Best mode | Hyperparameters | Result/source |
|---|---:|---|---|---|
| sexual | 97.0 | `hybrid` | `sh=2.5 tau=0.6 theta_text=0.10 theta_img=0.10` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |
| violence | 89.0 | `anchor` | `sa=2.0 tau=0.6 theta_text=0.10 theta_img=0.10` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |
| illegal | 86.0 | `anchor` | `sa=3.0 tau=0.6 theta_text=0.10 theta_img=0.10` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |
| disturbing | 98.0 | `anchor` | `sa=1.5 tau=0.6 theta_text=0.10 theta_img=0.10` | `SR from crossbackbone_sd3_flux1_official_final_20260501.md / paper appendix Table 17; raw per-class file not found` |

**Avg SR:** 92.5. Full breakdown TBD(raw not found).

## Related files writer should open
- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md`
- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/CROSSBACKBONE_MJA_TABLE4_WITH_SAFEDENOISER_SGF_20260501.md`
- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/MLWRITER_FINAL_TABLE_VALUES_20260501.md`
- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/i2p_multi_sr_full_nr_tables_20260501.md`
- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/TABLE1_NUDITY_BREAKDOWN_RELIABLE_HANDOFF_20260501.md`
- `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/ours_best_configs_i2p_q16_top60.json`

## IMAGE-BACKED CROSS-BACKBONE I2P UPDATE

Use this stricter image-backed source when paper table needs SR/Safe/Partial/Full/NR for Ours SD3/FLUX I2P.

### SD3 Ours best image-backed

| Concept | SR/Safe/Partial/Full/NR | Config | Image dir | Result file |
|---|---:|---|---|---|
| sexual | 93.3/86.7/6.7/5.0/1.7 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/sexual/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/sexual/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_nudity_v5.txt` |
| violence | 76.7/70.0/6.7/23.3/0.0 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/violence/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/violence/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_violence_v5.txt` |
| self-harm | 46.7/43.3/3.3/5.0/48.3 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both/results_qwen3_vl_self_harm_v5.txt` |
| shocking | 85.0/75.0/10.0/10.0/5.0 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/shocking/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/shocking/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_shocking_v5.txt` |
| illegal_activity | 56.7/51.7/5.0/10.0/33.3 | `hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/illegal_activity/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/illegal_activity/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both/results_qwen3_vl_illegal_v5.txt` |
| harassment | 40.0/31.7/8.3/23.3/36.7 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both/results_qwen3_vl_harassment_v5.txt` |
| hate | 56.7/51.7/5.0/18.3/25.0 | `hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_hate_v5.txt` |

**Avg:** SR 65.0, Safe 58.6, Partial 6.4, Full 13.6, NR 21.4.

### FLUX1 Ours best image-backed

| Concept | SR/Safe/Partial/Full/NR | Config | Image dir | Result file |
|---|---:|---|---|---|
| sexual | 100.0/98.3/1.7/0.0/0.0 | `hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/sexual/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/sexual/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_nudity_v5.txt` |
| violence | 86.7/61.7/25.0/13.3/0.0 | `hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/violence/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/violence/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_violence_v5.txt` |
| self-harm | 65.0/23.3/41.7/10.0/25.0 | `hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/self-harm/hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/self-harm/hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_self_harm_v5.txt` |
| shocking | 100.0/96.7/3.3/0.0/0.0 | `hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/shocking/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/shocking/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_shocking_v5.txt` |
| illegal_activity | 60.0/43.3/16.7/11.7/28.3 | `hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/illegal_activity/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/illegal_activity/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_illegal_v5.txt` |
| harassment | 68.3/45.0/23.3/16.7/15.0 | `hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/harassment/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/harassment/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_harassment_v5.txt` |
| hate | 83.3/76.7/6.7/13.3/3.3 | `hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/hate/hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/hate/hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_hate_v5.txt` |

**Avg:** SR 80.5, Safe 63.6, Partial 16.9, Full 9.3, NR 10.2.

