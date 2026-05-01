# Cross-backbone final breakdown: I2P q16 top-60 and MJA (best-mode Ours)

Metric cell format: **SR / Safe / Partial / Full / NR** (%). SR = Safe + Partial. NR is NotRelevant, or NotPeople for nudity/sexual rubric.

Important: **Ours is reported as best-of-mode where both anchor/hybrid exist**, because per-backbone/per-concept mode tuning is allowed. For I2P cross-backbone, the final available Ours row is the verified best row from the sweep; only SR is currently summarized in handoff, so detailed Safe/Partial/Full/NR cells remain TBD unless raw Ours result files are supplied.

## SD3 I2P q16 top-60

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg SR | Avg Safe | Avg Partial | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 93.3/73.3/20.0/5.0/1.7 | 41.7/31.7/10.0/58.3/0.0 | 38.3/8.3/30.0/30.0/31.7 | 36.7/25.0/11.7/63.3/0.0 | 41.7/30.0/11.7/30.0/28.3 | 35.0/26.7/8.3/31.7/33.3 | 48.3/43.3/5.0/43.3/8.3 | 47.9 | 34.0 | 13.8 | 37.4 | 14.8 |
| SAFREE + SafeDenoiser | 91.7/68.3/23.3/3.3/5.0 | 41.7/31.7/10.0/58.3/0.0 | 35.0/11.7/23.3/33.3/31.7 | 31.7/20.0/11.7/68.3/0.0 | 40.0/33.3/6.7/30.0/30.0 | 30.0/20.0/10.0/38.3/31.7 | 48.3/35.0/13.3/46.7/5.0 | 45.5 | 31.4 | 14.0 | 39.7 | 14.8 |
| SAFREE + SGF | 91.7/71.7/20.0/5.0/3.3 | 35.0/28.3/6.7/65.0/0.0 | 38.3/8.3/30.0/33.3/28.3 | 26.7/18.3/8.3/73.3/0.0 | 36.7/28.3/8.3/38.3/25.0 | 31.7/23.3/8.3/41.7/26.7 | 40.0/36.7/3.3/50.0/10.0 | 42.9 | 30.7 | 12.1 | 43.8 | 13.3 |
| **Ours best** | 96.7/TBD/TBD/TBD/TBD | 73.3/TBD/TBD/TBD/TBD | 43.3/TBD/TBD/TBD/TBD | 90.0/TBD/TBD/TBD/TBD | 50.0/TBD/TBD/TBD/TBD | 36.7/TBD/TBD/TBD/TBD | 56.7/TBD/TBD/TBD/TBD | **63.8** | TBD | TBD | TBD | TBD |

## FLUX1 I2P q16 top-60

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg SR | Avg Safe | Avg Partial | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 95.0/56.7/38.3/3.3/1.7 | 56.7/36.7/20.0/36.7/6.7 | 51.7/13.3/38.3/31.7/16.7 | 33.3/25.0/8.3/66.7/0.0 | 43.3/28.3/15.0/20.0/36.7 | 38.3/21.7/16.7/28.3/33.3 | 45.0/40.0/5.0/41.7/13.3 | 51.9 | 31.7 | 20.2 | 32.6 | 15.5 |
| SAFREE + SafeDenoiser | 91.7/58.3/33.3/6.7/1.7 | 45.0/25.0/20.0/53.3/1.7 | 46.7/15.0/31.7/35.0/18.3 | 35.0/18.3/16.7/65.0/0.0 | 45.0/23.3/21.7/18.3/36.7 | 36.7/20.0/16.7/35.0/28.3 | 48.3/36.7/11.7/43.3/8.3 | 49.8 | 28.1 | 21.7 | 36.7 | 13.6 |
| SAFREE + SGF | 93.3/60.0/33.3/3.3/3.3 | 46.7/25.0/21.7/51.7/1.7 | 51.7/10.0/41.7/28.3/20.0 | 31.7/23.3/8.3/68.3/0.0 | 48.3/26.7/21.7/18.3/33.3 | 35.0/20.0/15.0/38.3/26.7 | 46.7/36.7/10.0/46.7/6.7 | 50.5 | 28.8 | 21.7 | 36.4 | 13.1 |
| **Ours best** | 100.0/TBD/TBD/TBD/TBD | 60.0/TBD/TBD/TBD/TBD | 65.0/TBD/TBD/TBD/TBD | 100.0/TBD/TBD/TBD/TBD | 60.0/TBD/TBD/TBD/TBD | 68.3/TBD/TBD/TBD/TBD | 80.0/TBD/TBD/TBD/TBD | **76.2** | TBD | TBD | TBD | TBD |

## SD3 MJA

| Method | sexual | violence | illegal | disturbing | Avg SR | Avg Safe | Avg Partial | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 67.0/24.0/43.0/31.0/2.0 | 0.0/0.0/0.0/99.0/1.0 | 24.0/9.0/15.0/74.0/2.0 | 40.0/3.0/37.0/60.0/0.0 | 32.8 | 9.0 | 23.8 | 66.0 | 1.2 |
| SAFREE + SafeDenoiser | 76.0/24.0/52.0/22.0/2.0 | 1.0/0.0/1.0/99.0/0.0 | 22.0/11.0/11.0/75.0/3.0 | 40.0/2.0/38.0/60.0/0.0 | 34.8 | 9.2 | 25.5 | 64.0 | 1.2 |
| SAFREE + SGF | 59.0/17.0/42.0/40.0/1.0 | 0.0/0.0/0.0/100.0/0.0 | 20.0/9.0/11.0/77.0/3.0 | 44.0/1.0/43.0/56.0/0.0 | 30.8 | 6.8 | 24.0 | 68.2 | 1.0 |
| **Ours best-of-mode** | 84.0/TBD/TBD/TBD/TBD (hybrid) | 58.0/TBD/TBD/TBD/TBD (anchor) | 67.0/TBD/TBD/TBD/TBD (hybrid) | 90.0/TBD/TBD/TBD/TBD (hybrid) | **74.8** | TBD | TBD | TBD | TBD |

## FLUX1 MJA

| Method | sexual | violence | illegal | disturbing | Avg SR | Avg Safe | Avg Partial | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 69.0/12.0/57.0/28.0/3.0 | 11.0/5.0/6.0/88.0/1.0 | 42.0/11.0/31.0/48.0/10.0 | 62.0/3.0/59.0/37.0/1.0 | 46.0 | 7.8 | 38.2 | 50.2 | 3.8 |
| SAFREE + SafeDenoiser | 58.0/5.0/53.0/41.0/1.0 | 4.0/1.0/3.0/96.0/0.0 | 34.0/11.0/23.0/64.0/2.0 | 59.0/3.0/56.0/41.0/0.0 | 38.8 | 5.0 | 33.8 | 60.5 | 0.8 |
| SAFREE + SGF | 58.0/6.0/52.0/41.0/1.0 | 2.0/0.0/2.0/98.0/0.0 | 36.0/8.0/28.0/61.0/3.0 | 56.0/2.0/54.0/43.0/1.0 | 38.0 | 4.0 | 34.0 | 60.8 | 1.2 |
| **Ours best-of-mode** | 97.0/TBD/TBD/TBD/TBD (hybrid) | 89.0/TBD/TBD/TBD/TBD (anchor) | 86.0/TBD/TBD/TBD/TBD (anchor) | 98.0/TBD/TBD/TBD/TBD (anchor) | **92.5** | TBD | TBD | TBD | TBD |

## Compact avg table for paper

| Benchmark | Method | SD3 Avg SR | FLUX1 Avg SR |
|---|---|---:|---:|
| I2P q16 top-60 | SAFREE | 47.9 | 51.9 |
| I2P q16 top-60 | SAFREE + SafeDenoiser | 45.5 | 49.8 |
| I2P q16 top-60 | SAFREE + SGF | 42.9 | 50.5 |
| I2P q16 top-60 | **Ours best** | **63.8** | **76.2** |
| MJA | SAFREE | 32.8 | 46.0 |
| MJA | SAFREE + SafeDenoiser | 34.8 | 38.8 |
| MJA | SAFREE + SGF | 30.8 | 38.0 |
| MJA | **Ours best-of-mode** | **74.8** | **92.5** |

## Source files

- i2p sd3 SAFREE sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/sexual/all/results_qwen3_vl_nudity_v5.txt`
- i2p sd3 SAFREE violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/violence/all/results_qwen3_vl_violence_v5.txt`
- i2p sd3 SAFREE self-harm: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/self-harm/all/results_qwen3_vl_self_harm_v5.txt`
- i2p sd3 SAFREE shocking: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/shocking/all/results_qwen3_vl_shocking_v5.txt`
- i2p sd3 SAFREE illegal_activity: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/illegal_activity/all/results_qwen3_vl_illegal_v5.txt`
- i2p sd3 SAFREE harassment: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/harassment/all/results_qwen3_vl_harassment_v5.txt`
- i2p sd3 SAFREE hate: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/i2p_q16/hate/all/results_qwen3_vl_hate_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/sexual/all/results_qwen3_vl_nudity_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/violence/all/results_qwen3_vl_violence_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser self-harm: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/self-harm/all/results_qwen3_vl_self_harm_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser shocking: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/shocking/all/results_qwen3_vl_shocking_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser illegal_activity: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/illegal_activity/all/results_qwen3_vl_illegal_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser harassment: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/harassment/all/results_qwen3_vl_harassment_v5.txt`
- i2p sd3 SAFREE + SafeDenoiser hate: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/i2p_q16/hate/all/results_qwen3_vl_hate_v5.txt`
- i2p sd3 SAFREE + SGF sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/sexual/all/results_qwen3_vl_nudity_v5.txt`
- i2p sd3 SAFREE + SGF violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/violence/all/results_qwen3_vl_violence_v5.txt`
- i2p sd3 SAFREE + SGF self-harm: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/self-harm/all/results_qwen3_vl_self_harm_v5.txt`
- i2p sd3 SAFREE + SGF shocking: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/shocking/all/results_qwen3_vl_shocking_v5.txt`
- i2p sd3 SAFREE + SGF illegal_activity: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/illegal_activity/all/results_qwen3_vl_illegal_v5.txt`
- i2p sd3 SAFREE + SGF harassment: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/harassment/all/results_qwen3_vl_harassment_v5.txt`
- i2p sd3 SAFREE + SGF hate: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/i2p_q16/hate/all/results_qwen3_vl_hate_v5.txt`
- i2p flux1 SAFREE sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/sexual/all/results_qwen3_vl_nudity_v5.txt`
- i2p flux1 SAFREE violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/violence/all/results_qwen3_vl_violence_v5.txt`
- i2p flux1 SAFREE self-harm: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/self-harm/all/results_qwen3_vl_self_harm_v5.txt`
- i2p flux1 SAFREE shocking: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/shocking/all/results_qwen3_vl_shocking_v5.txt`
- i2p flux1 SAFREE illegal_activity: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/illegal_activity/all/results_qwen3_vl_illegal_v5.txt`
- i2p flux1 SAFREE harassment: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/harassment/all/results_qwen3_vl_harassment_v5.txt`
- i2p flux1 SAFREE hate: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/i2p_q16/hate/all/results_qwen3_vl_hate_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/sexual/all/results_qwen3_vl_nudity_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/violence/all/results_qwen3_vl_violence_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser self-harm: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/self-harm/all/results_qwen3_vl_self_harm_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser shocking: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/shocking/all/results_qwen3_vl_shocking_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser illegal_activity: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/illegal_activity/all/results_qwen3_vl_illegal_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser harassment: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/harassment/all/results_qwen3_vl_harassment_v5.txt`
- i2p flux1 SAFREE + SafeDenoiser hate: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/i2p_q16/hate/all/results_qwen3_vl_hate_v5.txt`
- i2p flux1 SAFREE + SGF sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/sexual/all/results_qwen3_vl_nudity_v5.txt`
- i2p flux1 SAFREE + SGF violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/violence/all/results_qwen3_vl_violence_v5.txt`
- i2p flux1 SAFREE + SGF self-harm: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/self-harm/all/results_qwen3_vl_self_harm_v5.txt`
- i2p flux1 SAFREE + SGF shocking: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/shocking/all/results_qwen3_vl_shocking_v5.txt`
- i2p flux1 SAFREE + SGF illegal_activity: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/illegal_activity/all/results_qwen3_vl_illegal_v5.txt`
- i2p flux1 SAFREE + SGF harassment: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/harassment/all/results_qwen3_vl_harassment_v5.txt`
- i2p flux1 SAFREE + SGF hate: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/i2p_q16/hate/all/results_qwen3_vl_hate_v5.txt`
- mja sd3 SAFREE sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt`
- mja sd3 SAFREE violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt`
- mja sd3 SAFREE illegal: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt`
- mja sd3 SAFREE disturbing: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safree/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt`
- mja sd3 SAFREE + SafeDenoiser sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt`
- mja sd3 SAFREE + SafeDenoiser violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt`
- mja sd3 SAFREE + SafeDenoiser illegal: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt`
- mja sd3 SAFREE + SafeDenoiser disturbing: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt`
- mja sd3 SAFREE + SGF sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt`
- mja sd3 SAFREE + SGF violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt`
- mja sd3 SAFREE + SGF illegal: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt`
- mja sd3 SAFREE + SGF disturbing: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/sd3/sgf/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt`
- mja flux1 SAFREE sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt`
- mja flux1 SAFREE violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt`
- mja flux1 SAFREE illegal: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt`
- mja flux1 SAFREE disturbing: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safree/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt`
- mja flux1 SAFREE + SafeDenoiser sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt`
- mja flux1 SAFREE + SafeDenoiser violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt`
- mja flux1 SAFREE + SafeDenoiser illegal: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt`
- mja flux1 SAFREE + SafeDenoiser disturbing: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt`
- mja flux1 SAFREE + SGF sexual: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt`
- mja flux1 SAFREE + SGF violence: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt`
- mja flux1 SAFREE + SGF illegal: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt`
- mja flux1 SAFREE + SGF disturbing: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/crossbackbone_0501/flux1/sgf/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt`
