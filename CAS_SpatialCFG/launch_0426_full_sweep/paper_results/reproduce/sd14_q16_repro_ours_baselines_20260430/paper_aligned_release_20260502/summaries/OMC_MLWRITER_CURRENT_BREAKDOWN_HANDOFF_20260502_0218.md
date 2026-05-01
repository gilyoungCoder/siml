# OMC / ML Writer handoff — current paper-aligned results with breakdowns

Generated: 2026-05-02 KST

## Canonical release / result roots

- Release bundle: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502`
- Main previous full handoff: `.../paper_aligned_release_20260502/summaries/FINAL_MLWRITER_OMC_HANDOFF_ALL_TABLES_CONFIGS_20260502_TBD_RESOLVED.md`
- Current short handoff: this file

## 1. SD v1.4 nudity — Baseline re-evaluated with Qwen3-VL V5

Source images: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/{unlearndiff,rab,mma,p4dn}`
Evaluator: `/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py`, concept=`nudity`, model=`qwen`.

| Dataset | n | SR | Safe | Partial | Full | NR/NotPeople | result file |
|---|---:|---:|---:|---:|---:|---:|---|
| UD / UnlearnDiff | 142 | 64.1 | 21.8 | 42.3 | 33.1 | 2.8 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/unlearndiff/results_qwen3_vl_nudity_v5.txt` |
| RAB | 79 | 45.6 | 20.3 | 25.3 | 50.6 | 3.8 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/rab/results_qwen3_vl_nudity_v5.txt` |
| MMA | 1000 | 35.9 | 12.3 | 23.6 | 63.8 | 0.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/mma/results_qwen3_vl_nudity_v5.txt` |
| P4DN | 151 | 25.2 | 7.3 | 17.9 | 74.2 | 0.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/p4dn/results_qwen3_vl_nudity_v5.txt` |

Note: These V5 values differ from older Table 9 baseline values; for consistency with V5 evaluator, use this row if replacing baseline breakdowns.

## 2. SD v1.4 nudity — Ours hybrid family breakdown

Source/result root: `.../paper_aligned_release_20260502/outputs/sd14_nudity/ours_hybrid_family/{dataset}/results_qwen3_vl_nudity_v5.txt`

| Dataset | n | SR | Safe | Partial | Full | NR/NotPeople | result dir |
|---|---:|---:|---:|---:|---:|---:|---|
| UD / UnlearnDiff | 142 | 97.2 | 73.9 | 23.2 | 1.4 | 1.4 | `.../outputs/sd14_nudity/ours_hybrid_family/unlearndiff/` |
| RAB | 79 | 96.2 | 89.9 | 6.3 | 2.5 | 1.3 | `.../outputs/sd14_nudity/ours_hybrid_family/rab/` |
| MMA | 1000 | 84.2 | 66.9 | 17.3 | 15.4 | 0.4 | `.../outputs/sd14_nudity/ours_hybrid_family/mma/` |
| P4DN | 151 | 97.4 | 92.1 | 5.3 | 2.6 | 0.0 | `.../outputs/sd14_nudity/ours_hybrid_family/p4dn/` |

Important: if manuscript says MMA Ours 84.4, the located V5 result file says 84.2. Use 84.2 unless there is a different canonical result file.

## 3. SD v1.4 nudity — SGF breakdown status

Use breakdown-backed values only. Current verified files:

| Dataset | n | SR | Safe | Partial | Full | NR/NotPeople | result file |
|---|---:|---:|---:|---:|---:|---:|---|
| UD | 142 | 92.3 | 69.0 | 23.2 | 2.8 | 4.9 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full/sgf/nudity/ud/all/results_qwen3_vl_nudity_v5.txt` |
| RAB | 79 | 84.8 | 64.6 | 20.3 | 7.6 | 7.6 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full/sgf/nudity/rab/all/results_qwen3_vl_nudity_v5.txt` |
| MMA | 1000 | 77.6 | 51.6 | 26.0 | 18.6 | 3.8 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full/sgf/nudity/mma/all/results_qwen3_vl_nudity_v5.txt` |
| P4DN | 151 | 70.2 | 37.7 | 32.5 | 25.8 | 4.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/sgf/nudity/p4dn/all/results_qwen3_vl_nudity_v5.txt` |

Caveat: old SGF P4DN 74.2 came from an incomplete n=147 run; do not use it. Full n=151 repaired result is 70.2.

## 4. Probe ablation — q16 top-60, 7 concepts, SD v1.4

Use file: `.../paper_aligned_release_20260502/summaries/probe_ablation_i2p_q16_top60_7concept_UPDATED_violence88p3_20260502.md`

| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 70.0 | 55.0 | 65.7 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both / Ours | 98.3 | 88.3 | 51.7 | 91.7 | 43.3 | 70.0 | 73.3 | 73.8 |

Violence final config/result: `.../outputs/sd14_i2p_single/ours_best/violence/sh20_tau04_txt030_img010/`, SR 88.3.

## 5. I2P multi-concept, q16 top-60, SD v1.4

Use file: `.../paper_aligned_release_20260502/summaries/i2p_multi_sr_full_nr_tables_20260501.md`

Main SR averages:
- 2c (sexual+violence): SAFREE 65.0, SafeDenoiser 77.5, SGF 72.5, Ours 76.7
- 3c (sexual+violence+shocking): SAFREE 52.8, SafeDenoiser 68.9, SGF 61.1, Ours 81.7
- 7c: SAFREE 41.9, SafeDenoiser 50.7, SGF 48.3, Ours 73.1

For paper table, include SR / Full / NR from the same summary file, not SR-only.

## 6. Cross-backbone SD3 / FLUX

Use main cross-backbone outputs under:
- `.../paper_aligned_release_20260502/outputs/crossbackbone_0501/`
- Ours q16 outputs under: `.../paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/`
- Summary/config details are in `FINAL_MLWRITER_OMC_HANDOFF_ALL_TABLES_CONFIGS_20260502_TBD_RESOLVED.md`.

Use best-of-mode for Ours where applicable; record full breakdown from result files when inserting appendix tables.

## 7. Still not done / pending

1. COCO FID/CLIP for SGF 10k/9966 is still generating. It was paused for baseline V5 evaluation and relaunched on siml-06 GPUs 0-3 at 2026-05-02 ~02:17 KST.
   - Logs: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/coco10k_9966_sgf_accel_siml06_20260502/`
   - Output chunks: `.../outputs/sgf_coco10k_9966/chunks/{2400_1892,4292_1892,6184_1892,8076_1890}`
2. SafeDenoiser/SGF COCO FID/CLIP final table should not be treated as final until 10k/9966 generation and FID/CLIP calculation finish.
3. If replacing main Table 1 baseline breakdown with V5 baseline_v2, update all four baseline cells consistently (UD/RAB/MMA/P4DN), because old baseline row and V5 rerun differ.
4. SGF P4DN: use 70.2 n=151, not old 74.2 n=147.

## 8. Ours cross-backbone I2P q16 top-60 breakdowns (SD3 / FLUX.1)

Root: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/
Selection rule below: best SR among complete n=60 configs; all are hybrid/both/family.

### SD3 Ours best breakdown

| Concept | SR | Safe | Partial | Full | NR | Config |
|---|---:|---:|---:|---:|---:|---|
| sexual | 96.7 | 90.0 | 6.7 | 1.7 | 1.7 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| violence | 73.3 | 68.3 | 5.0 | 25.0 | 1.7 | sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both |
| self-harm | 43.3 | 35.0 | 8.3 | 10.0 | 46.7 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| shocking | 90.0 | 78.3 | 11.7 | 10.0 | 0.0 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| illegal | 50.0 | 40.0 | 10.0 | 20.0 | 30.0 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| harassment | 36.7 | 26.7 | 10.0 | 25.0 | 38.3 | sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both |
| hate | 56.7 | 51.7 | 5.0 | 18.3 | 25.0 | sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both |
| Avg | 63.8 | - | - | - | - | - |

### FLUX.1 Ours best breakdown

| Concept | SR | Safe | Partial | Full | NR | Config |
|---|---:|---:|---:|---:|---:|---|
| sexual | 100.0 | 93.3 | 6.7 | 0.0 | 0.0 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| violence | 60.0 | 30.0 | 30.0 | 38.3 | 1.7 | flux_xlow_hybrid_ss0.25_thr0.15_imgthr0.1_cas0.45_both |
| self-harm | 65.0 | 21.7 | 43.3 | 10.0 | 25.0 | flux_low_hybrid_ss0.75_thr0.15_imgthr0.1_cas0.45_both |
| shocking | 100.0 | 96.7 | 3.3 | 0.0 | 0.0 | flux_high_hybrid_ss2.5_thr0.15_imgthr0.1_cas0.5_both |
| illegal | 60.0 | 43.3 | 16.7 | 11.7 | 28.3 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| harassment | 68.3 | 45.0 | 23.3 | 16.7 | 15.0 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| hate | 80.0 | 75.0 | 5.0 | 11.7 | 8.3 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| Avg | 76.2 | - | - | - | - | - |

Note: this SD3 avg 63.8 differs from earlier text value 65.0 because this table uses strict complete n=60 best-config files found in the release bundle. Earlier 65.0 likely used a different/partial selection; prefer the breakdown-backed table unless another canonical file is specified.

## 9. Ours cross-backbone MJA breakdowns located — HYBRID-ONLY paper setting

Important policy: for the paper version, Ours should be reported as hybrid-only. Do not use anchor-inpaint rows for MJA even if their SR is higher.

Located roots:
- SD3 Ours MJA: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/`
- FLUX.1 Ours MJA: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/`

Selection below: best complete n=100 V5 result per concept among directories containing `hybrid` only.

### SD3 Ours MJA hybrid-only breakdown

| Concept | SR | Safe | Partial | Full | NR | Config / result file |
|---|---:|---:|---:|---:|---:|---|
| Sexual | 84.0 | 56.0 | 28.0 | 16.0 | 0.0 | `outputs/launch_0420/ours_sd3/mja_sexual/hybrid_ss15_thr0.1_imgthr0.3_both/results_qwen3_vl_nudity_v5.txt` |
| Violent | 23.0 | 18.0 | 5.0 | 72.0 | 5.0 | `outputs/launch_0420/ours_sd3/mja_violent/cas0.5_ss25_thr0.1_imgthr0.1_hybrid_both/results_qwen3_vl_violence_v5.txt` |
| Illegal | 50.0 | 37.0 | 13.0 | 36.0 | 14.0 | `outputs/launch_0420/ours_sd3/mja_illegal/cas0.45_ss25_thr0.1_imgthr0.1_hybrid_both/results_qwen3_vl_illegal_v5.txt` |
| Disturbing | 47.0 | 11.0 | 36.0 | 53.0 | 0.0 | `outputs/launch_0420/ours_sd3/mja_disturbing/cas0.6_ss25_thr0.1_imgthr0.1_hybrid_both/results_qwen3_vl_disturbing_v5.txt` |
| Avg | 51.0 | - | - | - | - | - |

This is much lower than the previous best-of-mode/anchor-mixed SD3 MJA number. If the manuscript currently claims SD3 MJA Ours around 69–70 while also saying hybrid-only, that is inconsistent with the located V5 hybrid-only files and must be resolved.

### FLUX.1 Ours MJA hybrid-only breakdown

| Concept | SR | Safe | Partial | Full | NR | Config / result file |
|---|---:|---:|---:|---:|---:|---|
| Sexual | 97.0 | 85.0 | 12.0 | 3.0 | 0.0 | `outputs/launch_0420/ours_flux1/mja_sexual/cas0.6_ss2.5_thr0.1_hybrid_both/results_qwen3_vl_nudity_v5.txt` |
| Violent | 67.0 | 57.0 | 10.0 | 20.0 | 13.0 | `outputs/launch_0420/ours_flux1/mja_violent/cas0.6_ss2.0_thr0.1_hybrid_both/results_qwen3_vl_violence_v5.txt` |
| Illegal | 58.0 | 33.0 | 25.0 | 35.0 | 7.0 | `outputs/launch_0420/ours_flux1/mja_illegal/cas0.6_ss2.0_thr0.1_hybrid_both/results_qwen3_vl_illegal_v5.txt` |
| Disturbing | 96.0 | 74.0 | 22.0 | 4.0 | 0.0 | `outputs/launch_0420/ours_flux1/mja_disturbing/cas0.6_ss3.0_thr0.1_hybrid_both/results_qwen3_vl_disturbing_v5.txt` |
| Avg | 79.5 | - | - | - | - | - |

This matches the earlier manuscript FLUX hybrid avg 79.5.

## 10. Hybrid-only verification for I2P Ours

- SD1.4 I2P single Ours in release is hybrid-only. The violence directory name lacks `hybrid`, but its `args.json` contains `"how_mode": "hybrid"` and `"safety_scale": 20.0`.
- Cross-backbone I2P Ours SD3/FLUX configs are also hybrid-only; selected config directories contain `hybrid` and no anchor/inpaint directories were used for the I2P cross-backbone table.
