# Paper update package — reproduction status (2026-04-30 KST)

Root: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430`

## 0. Immediate status

- Human survey DB recovered: 1662 rows / 20 annotators; `/api/results` now paginates and returns all rows.
- COCO FID/CLIP for **Safe Denoiser** and **SGF** nudity configs is running on `siml-07` GPU1. Logs: `logs/coco_fid_official/nohup_siml07_gpu1.log`.
- Concept-specific SafeDenoiser/SGF I2P reference generation is still running on `siml-09`; old I2P SafeDenoiser/SGF rows below are nudity/sexual-reference preliminary rows, not final concept-specific rows.

## 1. Nudity SD v1.4 VLM SR table update

Use these SR values for SafeDenoiser/SGF rows; COCO utility for these two is pending the running job.

| Method | UD | RAB | MMA | P4DN | COCO FID | COCO CLIP | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 71.1 | 48.1 | 35.4 | 33.1 | – | 0.2675 | draft/reference |
| SAFREE | 87.3 | 83.5 | 75.4 | 70.9 | 8.96 | 0.264 | reproduced |
| SAFREE + Safe Denoiser | 95.1 | 81.0 | 73.4 | 62.9 | RUNNING | RUNNING | nudity official config; COCO launched |
| SAFREE + SGF | 92.3 | 83.5 | 76.9 | 74.2 | RUNNING | RUNNING | nudity official config; COCO launched |
| EBSG hybrid | 97.2 | 96.2 | 84.2 | 97.4 | 6.31/6.78 | 0.263 | MMA differs from draft 84.4 by -0.2pp; FID depends on current eval artifact |

## 2. I2P q16 top-60 single-concept table (7 concepts incl. sexual)

| concept | baseline | SAFREE | SafeDenoiser | SGF | ours main | ours best | best variant | winner |
|---|---:|---:|---:|---:|---:|---:|---|---|
| sexual | 68.3 | 83.3 | 91.7 | 95.0 | 90.0 | 98.3 | `hybrid_best_tau05_cas0.5` | ours_best |
| violence | 36.7 | 73.3 | 43.3 | 41.7 | 75.0 | 81.7 | `hybrid_best_img075_img0.225` | ours_best |
| self-harm | 43.3 | 36.7 | 41.7 | 35.0 | 50.0 | 51.7 | `hybrid_best_tau05_cas0.5` | ours_best |
| shocking | 15.0 | 81.7 | 20.0 | 16.7 | 85.0 | 93.3 | `hybrid_best_ss125_ss27.5` | ours_best |
| illegal_activity | 31.7 | 35.0 | 41.7 | 25.0 | 41.7 | 46.7 | `hybrid_best_ss125_ss25.0` | ours_best |
| harassment | 25.0 | 28.3 | 25.0 | 28.3 | 46.7 | 68.3 | `hybrid_best_ss125_ss31.25` | ours_best |
| hate | 25.0 | 43.3 | 28.3 | 25.0 | 70.0 | 73.3 | `hybrid_best_img075_img0.0375` | ours_best |

**6-concept avg excl sexual**

- baseline: 29.4
- safree: 49.7
- safedenoiser: 33.3
- sgf: 28.6
- ours: 61.4
- ours_best: 69.2

**7-concept avg incl sexual**

- baseline: 35.0
- safree: 54.5
- safedenoiser: 41.7
- sgf: 38.1
- ours: 65.5
- ours_best: 73.3

**Recommended paper row:** report `ours_best` or per-concept best config; if reporting one fixed config, use `ours main` and state tuned best in appendix.

## 3. I2P q16 top-60 multi-concept: SAFREE vs EBSG 7-concept

| concept | SAFREE multi | EBSG multi C3 tau0.7 | Δ |
|---|---:|---:|---:|
| sexual | 6.7 | 85.0 | +78.3 |
| violence | 5.0 | 63.3 | +58.3 |
| self-harm | 1.7 | 65.0 | +63.3 |
| shocking | 8.3 | 71.7 | +63.4 |
| illegal_activity | 0.0 | 56.7 | +56.7 |
| harassment | 0.0 | 45.0 | +45.0 |
| hate | 1.7 | 61.7 | +60.0 |

Avg incl sexual: SAFREE 3.3, EBSG 64.1, Δ +60.7.
Avg excl sexual: SAFREE 2.8, EBSG 60.6, Δ +57.8.

## 4. Best config file paths

- JSON: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/ours_best_configs_i2p_q16_top60.json`
- Single table CSV: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/i2p_q16_top60_v5_5method_comparison_with_ours_best.csv`
- Multi table CSV: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/multi_7c_safree_vs_ebsg_c3_q16top60.csv`

## 5. Running COCO job

- Script: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/scripts/run_coco_fid_safedenoiser_sgf.sh`
- Prompt CSV: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/coco/coco_250_x4.csv`
- SafeDenoiser out: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safedenoiser/nudity_coco/coco250x4/all`
- SGF out: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/sgf/nudity_coco/coco250x4/all`
- Logs: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/coco_fid_official`

## 6. Human survey

- Full dump: `/mnt/c/users/yhgil/human_survey_rescue_20260430_205717/full_dump/human_agreement_all_results.json`
- Summary: `/mnt/c/users/yhgil/human_survey_rescue_20260430_205717/full_dump/FULL_RECOVERY_SUMMARY.md`
- `/api/results` verified to return 1662 rows after Vercel redeploy.


## 7. UPDATED: concept-specific SafeDenoiser/SGF I2P rows

# I2P q16 top-60 final table with concept-specific SafeDenoiser/SGF

| concept | Baseline | SAFREE | SafeDenoiser-CS | SGF-CS | Ours main | Ours best | best config | winner |
|---|---:|---:|---:|---:|---:|---:|---|---|
| sexual | 68.3 | 83.3 | 91.7 | 96.7 | 90.0 | 98.3 | `hybrid_best_tau05_cas0.5` | ours_best |
| violence | 36.7 | 73.3 | 40.0 | 40.0 | 75.0 | 81.7 | `hybrid_best_img075_img0.225` | ours_best |
| self-harm | 43.3 | 36.7 | 38.3 | 36.7 | 50.0 | 51.7 | `hybrid_best_tau05_cas0.5` | ours_best |
| shocking | 15.0 | 81.7 | 21.7 | 18.3 | 85.0 | 93.3 | `hybrid_best_ss125_ss27.5` | ours_best |
| illegal_activity | 31.7 | 35.0 | 41.7 | 30.0 | 41.7 | 46.7 | `hybrid_best_ss125_ss25.0` | ours_best |
| harassment | 25.0 | 28.3 | 26.7 | 23.3 | 46.7 | 68.3 | `hybrid_best_ss125_ss31.25` | ours_best |
| hate | 25.0 | 43.3 | 28.3 | 21.7 | 70.0 | 73.3 | `hybrid_best_img075_img0.0375` | ours_best |

**Avg 7-concept incl sexual**

- baseline: 35.0
- safree: 54.5
- safedenoiser_cs: 41.2
- sgf_cs: 38.1
- ours: 65.5
- ours_best: 73.3

**Avg 6-concept excl sexual**

- baseline: 29.4
- safree: 49.7
- safedenoiser_cs: 32.8
- sgf_cs: 28.3
- ours: 61.4
- ours_best: 69.2

Notes: CS means concept-specific reference examples/caches were used for SafeDenoiser/SGF instead of the earlier nudity/sexual-only reference.
