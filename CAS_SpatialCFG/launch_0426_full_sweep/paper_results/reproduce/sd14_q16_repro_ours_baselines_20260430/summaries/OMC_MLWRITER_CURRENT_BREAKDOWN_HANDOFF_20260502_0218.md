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

