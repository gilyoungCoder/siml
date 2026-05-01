# Final curated paper data handoff — hybrid-only, V5, reproducible paths

Generated: 2026-05-02 KST

- Curated folder: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_final_curated_hybrid_only_20260502`
- Release source: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502`
- Rule: include only complete, parsed V5 result files. Ours rows are hybrid-only. Anchor/best-of-mode rows are excluded from final Ours claims.

## Validation status

OK: all curated result files exist, parse, match expected n, and Ours hybrid rows pass args/path hybrid checks.

## Table 1 nudity baseline V5 breakdown

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd14 | Baseline SD1.4 | UD | 142 | 64.1 | 21.8 | 42.3 | 33.1 | 2.8 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/unlearndiff/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Baseline SD1.4 | RAB | 79 | 45.6 | 20.3 | 25.3 | 50.6 | 3.8 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/rab/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Baseline SD1.4 | MMA | 1000 | 35.9 | 12.3 | 23.6 | 63.8 | 0.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/mma/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Baseline SD1.4 | P4DN | 151 | 25.2 | 7.3 | 17.9 | 74.2 | 0.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/baselines_v2/p4dn/results_qwen3_vl_nudity_v5.txt` |

## Table 1 nudity Ours hybrid-family breakdown

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd14 | Ours hybrid family | UD | 142 | 97.2 | 73.9 | 23.2 | 1.4 | 1.4 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_nudity/ours_hybrid_family/unlearndiff/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Ours hybrid family | RAB | 79 | 96.2 | 89.9 | 6.3 | 2.5 | 1.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_nudity/ours_hybrid_family/rab/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Ours hybrid family | MMA | 1000 | 84.2 | 66.9 | 17.3 | 15.4 | 0.4 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_nudity/ours_hybrid_family/mma/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Ours hybrid family | P4DN | 151 | 97.4 | 92.1 | 5.3 | 2.6 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_nudity/ours_hybrid_family/p4dn/results_qwen3_vl_nudity_v5.txt` |

## Table 1 nudity SGF verified breakdown

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd14 | SGF | UD | 142 | 92.3 | 69.0 | 23.2 | 2.8 | 4.9 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full/sgf/nudity/ud/all/results_qwen3_vl_nudity_v5.txt` |
| sd14 | SGF | RAB | 79 | 84.8 | 64.6 | 20.3 | 7.6 | 7.6 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full/sgf/nudity/rab/all/results_qwen3_vl_nudity_v5.txt` |
| sd14 | SGF | MMA | 1000 | 77.6 | 51.6 | 26.0 | 18.6 | 3.8 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full/sgf/nudity/mma/all/results_qwen3_vl_nudity_v5.txt` |
| sd14 | SGF | P4DN | 151 | 70.2 | 37.7 | 32.5 | 25.8 | 4.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/sgf/nudity/p4dn/all/results_qwen3_vl_nudity_v5.txt` |

## SD1.4 I2P q16 top-60 Ours single-concept hybrid

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd14 | Ours hybrid | sexual | 60 | 98.3 | 91.7 | 6.7 | 1.7 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/sexual/hybrid_best_tau05_cas0.5/results_qwen3_vl_nudity_v5.txt` |
| sd14 | Ours hybrid | violence | 60 | 88.3 | 73.3 | 15.0 | 5.0 | 6.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/violence/sh20_tau04_txt030_img010/results_qwen3_vl_violence_v5.txt` |
| sd14 | Ours hybrid | self-harm | 60 | 51.7 | 8.3 | 43.3 | 23.3 | 25.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/self-harm/hybrid_best_tau05_cas0.5/results_qwen3_vl_self_harm_v5.txt` |
| sd14 | Ours hybrid | shocking | 60 | 93.3 | 88.3 | 5.0 | 3.3 | 3.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/shocking/hybrid_best_ss125_ss27.5/results_qwen3_vl_shocking_v5.txt` |
| sd14 | Ours hybrid | illegal_activity | 60 | 46.7 | 31.7 | 15.0 | 15.0 | 38.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/illegal_activity/hybrid_best_ss125_ss25.0/results_qwen3_vl_illegal_v5.txt` |
| sd14 | Ours hybrid | harassment | 60 | 68.3 | 56.7 | 11.7 | 13.3 | 18.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/harassment/hybrid_best_ss125_ss31.25/results_qwen3_vl_harassment_v5.txt` |
| sd14 | Ours hybrid | hate | 60 | 73.3 | 56.7 | 16.7 | 10.0 | 16.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/sd14_i2p_single/ours_best/hate/hybrid_best_img075_img0.0375/results_qwen3_vl_hate_v5.txt` |

## MJA cross-backbone Ours hybrid-only

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd3 | Ours hybrid | sexual | 100 | 84.0 | 56.0 | 28.0 | 16.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/mja_sexual/hybrid_ss15_thr0.1_imgthr0.3_both/results_qwen3_vl_nudity_v5.txt` |
| sd3 | Ours hybrid | violent | 100 | 23.0 | 18.0 | 5.0 | 72.0 | 5.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/mja_violent/cas0.5_ss25_thr0.1_imgthr0.1_hybrid_both/results_qwen3_vl_violence_v5.txt` |
| sd3 | Ours hybrid | illegal | 100 | 50.0 | 37.0 | 13.0 | 36.0 | 14.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/mja_illegal/cas0.45_ss25_thr0.1_imgthr0.1_hybrid_both/results_qwen3_vl_illegal_v5.txt` |
| sd3 | Ours hybrid | disturbing | 100 | 47.0 | 11.0 | 36.0 | 53.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/mja_disturbing/cas0.6_ss25_thr0.1_imgthr0.1_hybrid_both/results_qwen3_vl_disturbing_v5.txt` |
| flux1 | Ours hybrid | sexual | 100 | 97.0 | 85.0 | 12.0 | 3.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/mja_sexual/cas0.6_ss2.5_thr0.1_hybrid_both/results_qwen3_vl_nudity_v5.txt` |
| flux1 | Ours hybrid | violent | 100 | 67.0 | 57.0 | 10.0 | 20.0 | 13.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/mja_violent/cas0.6_ss2.0_thr0.1_hybrid_both/results_qwen3_vl_violence_v5.txt` |
| flux1 | Ours hybrid | illegal | 100 | 58.0 | 33.0 | 25.0 | 35.0 | 7.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/mja_illegal/cas0.6_ss2.0_thr0.1_hybrid_both/results_qwen3_vl_illegal_v5.txt` |
| flux1 | Ours hybrid | disturbing | 100 | 96.0 | 74.0 | 22.0 | 4.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/mja_disturbing/cas0.6_ss3.0_thr0.1_hybrid_both/results_qwen3_vl_disturbing_v5.txt` |

## MJA cross-backbone baselines

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd3 | safree | sexual | 100 | 67.0 | 24.0 | 43.0 | 31.0 | 2.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safree/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt` |
| sd3 | safree | violent | 100 | 0.0 | 0.0 | 0.0 | 99.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safree/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt` |
| sd3 | safree | illegal | 100 | 24.0 | 9.0 | 15.0 | 74.0 | 2.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safree/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt` |
| sd3 | safree | disturbing | 100 | 40.0 | 3.0 | 37.0 | 60.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safree/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt` |
| sd3 | safedenoiser | sexual | 100 | 76.0 | 24.0 | 52.0 | 22.0 | 2.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt` |
| sd3 | safedenoiser | violent | 100 | 1.0 | 0.0 | 1.0 | 99.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt` |
| sd3 | safedenoiser | illegal | 100 | 22.0 | 11.0 | 11.0 | 75.0 | 3.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt` |
| sd3 | safedenoiser | disturbing | 100 | 40.0 | 2.0 | 38.0 | 60.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/safedenoiser/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt` |
| sd3 | sgf | sexual | 100 | 59.0 | 17.0 | 42.0 | 40.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/sgf/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt` |
| sd3 | sgf | violent | 100 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/sgf/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt` |
| sd3 | sgf | illegal | 100 | 20.0 | 9.0 | 11.0 | 77.0 | 3.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/sgf/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt` |
| sd3 | sgf | disturbing | 100 | 44.0 | 1.0 | 43.0 | 56.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/sd3/sgf/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt` |
| flux1 | safree | sexual | 100 | 69.0 | 12.0 | 57.0 | 28.0 | 3.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safree/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt` |
| flux1 | safree | violent | 100 | 11.0 | 5.0 | 6.0 | 88.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safree/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt` |
| flux1 | safree | illegal | 100 | 42.0 | 11.0 | 31.0 | 48.0 | 10.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safree/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt` |
| flux1 | safree | disturbing | 100 | 62.0 | 3.0 | 59.0 | 37.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safree/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt` |
| flux1 | safedenoiser | sexual | 100 | 58.0 | 5.0 | 53.0 | 41.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt` |
| flux1 | safedenoiser | violent | 100 | 4.0 | 1.0 | 3.0 | 96.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt` |
| flux1 | safedenoiser | illegal | 100 | 34.0 | 11.0 | 23.0 | 64.0 | 2.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt` |
| flux1 | safedenoiser | disturbing | 100 | 59.0 | 3.0 | 56.0 | 41.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/safedenoiser/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt` |
| flux1 | sgf | sexual | 100 | 58.0 | 6.0 | 52.0 | 41.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/sgf/mja/mja_sexual/all/results_qwen3_vl_nudity_v5.txt` |
| flux1 | sgf | violent | 100 | 2.0 | 0.0 | 2.0 | 98.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/sgf/mja/mja_violent/all/results_qwen3_vl_violence_v5.txt` |
| flux1 | sgf | illegal | 100 | 36.0 | 8.0 | 28.0 | 61.0 | 3.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/sgf/mja/mja_illegal/all/results_qwen3_vl_illegal_v5.txt` |
| flux1 | sgf | disturbing | 100 | 56.0 | 2.0 | 54.0 | 43.0 | 1.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/crossbackbone_0501/flux1/sgf/mja/mja_disturbing/all/results_qwen3_vl_disturbing_v5.txt` |

## I2P q16 top-60 cross-backbone Ours hybrid

| Backbone | Method | Dataset | n | SR | Safe | Partial | Full | NR | Result path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd3 | Ours hybrid | sexual | 60 | 96.7 | 90.0 | 6.7 | 1.7 | 1.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/sexual/sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both/results_qwen3_vl_nudity_v5.txt` |
| sd3 | Ours hybrid | violence | 60 | 73.3 | 68.3 | 5.0 | 25.0 | 1.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/violence/sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_violence_v5.txt` |
| sd3 | Ours hybrid | self-harm | 60 | 43.3 | 35.0 | 8.3 | 10.0 | 46.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/self-harm/sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both/results_qwen3_vl_self_harm_v5.txt` |
| sd3 | Ours hybrid | shocking | 60 | 90.0 | 78.3 | 11.7 | 10.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/shocking/sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both/results_qwen3_vl_shocking_v5.txt` |
| sd3 | Ours hybrid | illegal_activity | 60 | 50.0 | 40.0 | 10.0 | 20.0 | 30.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/illegal_activity/sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both/results_qwen3_vl_illegal_v5.txt` |
| sd3 | Ours hybrid | harassment | 60 | 36.7 | 26.7 | 10.0 | 25.0 | 38.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/harassment/sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_harassment_v5.txt` |
| sd3 | Ours hybrid | hate | 60 | 56.7 | 51.7 | 5.0 | 18.3 | 25.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/sd3/hate/sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_hate_v5.txt` |
| flux1 | Ours hybrid | sexual | 60 | 100.0 | 93.3 | 6.7 | 0.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/sexual/flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_nudity_v5.txt` |
| flux1 | Ours hybrid | violence | 60 | 60.0 | 30.0 | 30.0 | 38.3 | 1.7 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/violence/flux_xlow_hybrid_ss0.25_thr0.15_imgthr0.1_cas0.45_both/results_qwen3_vl_violence_v5.txt` |
| flux1 | Ours hybrid | self-harm | 60 | 65.0 | 21.7 | 43.3 | 10.0 | 25.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/self-harm/flux_low_hybrid_ss0.75_thr0.15_imgthr0.1_cas0.45_both/results_qwen3_vl_self_harm_v5.txt` |
| flux1 | Ours hybrid | shocking | 60 | 100.0 | 96.7 | 3.3 | 0.0 | 0.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/shocking/flux_high_hybrid_ss2.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_shocking_v5.txt` |
| flux1 | Ours hybrid | illegal_activity | 60 | 60.0 | 43.3 | 16.7 | 11.7 | 28.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/illegal_activity/flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_illegal_v5.txt` |
| flux1 | Ours hybrid | harassment | 60 | 68.3 | 45.0 | 23.3 | 16.7 | 15.0 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/harassment/flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_harassment_v5.txt` |
| flux1 | Ours hybrid | hate | 60 | 80.0 | 75.0 | 5.0 | 11.7 | 8.3 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502/outputs/ours_crossbackbone_i2p_q16/sd3_flux1_q16_7concept_20260430/flux1/hate/flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_hate_v5.txt` |

## Paper update guidance

- I2P Ours rows are hybrid-only. The SD1.4 violence folder name does not include `hybrid`, but `args.json` has `how_mode=hybrid`.
- MJA must be reported as hybrid-only if the paper text says hybrid-only. Do not use the higher anchor/best-of-mode MJA values.
- Hybrid-only MJA Ours averages from curated files: SD3 = 51.0, FLUX.1 = 79.5. SD3 still beats SAFREE/SafeDenoiser/SGF averages (32.8/34.8/30.8) but is much lower than the old mixed-mode 69.x claim.
- FLUX.1 MJA hybrid-only Ours avg 79.5 beats SAFREE/SafeDenoiser/SGF averages (46.0/38.8/38.0).
- SGF P4DN old 74.2 n=147 is excluded; use repaired n=151 SR 70.2 if including SGF nudity breakdown.
- COCO FID/CLIP is still pending and should not be listed as final here.

## Clarification on OMC-reported residual TBDs

OMC reported three residual TBD groups. Current status after filesystem verification:

### A. Table 1 SLD-Weak/Medium/Strong/Max P4DN

Status: **still genuinely missing / not curated**.

I did not find a complete Qwen3-VL V5 P4DN result for SLD-Weak/Medium/Strong/Max under CAS. Keep these cells as TBD unless we rerun SLD P4DN and evaluate with V5.

### B. Cross-backbone full SD1.4 SafeDenoiser/SGF MJA cells

Status: **out of scope, not a TBD**.

For the paper, MJA cross-backbone comparison is **SD3.0 and FLUX1.0 only**. We are not including SD1.4 SafeDenoiser/SGF MJA cells, so these should be removed from the table/request rather than marked TBD. SD3/FLUX SafeDenoiser/SGF MJA results exist under `paper_aligned_release_20260502/outputs/crossbackbone_0501/{sd3,flux1}/{safedenoiser,sgf}/mja/...`.

### C. I2P cross-backbone SD3/FLUX SafeDenoiser/SGF baselines

Status: **not missing; found and can be filled**.

Root: `paper_aligned_release_20260502/outputs/crossbackbone_0501/{sd3,flux1}/{safedenoiser,sgf}/i2p_q16/{concept}/all/results_qwen3_vl_*_v5.txt`

SR values:

| Backbone | Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SD3 | SafeDenoiser | 91.7 | 41.7 | 35.0 | 31.7 | 40.0 | 30.0 | 48.3 | 45.5 |
| SD3 | SGF | 91.7 | 35.0 | 38.3 | 26.7 | 36.7 | 31.7 | 40.0 | 42.9 |
| FLUX.1 | SafeDenoiser | 91.7 | 45.0 | 46.7 | 35.0 | 45.0 | 36.7 | 48.3 | 49.8 |
| FLUX.1 | SGF | 93.3 | 46.7 | 51.7 | 31.7 | 48.3 | 35.0 | 46.7 | 50.5 |

These can be added to cross-backbone I2P appendix/table if needed. The curated final file originally emphasized Ours hybrid, but these baseline result files are present and parse correctly.

## Canonical decision: I2P cross-backbone Ours uses image-backed rows

Use the image-backed document as canonical for I2P cross-backbone Ours:

- Source: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/CROSSBACKBONE_OURS_I2P_IMAGE_BACKED_BREAKDOWN_20260501.md`
- Reason: each cell has an image directory, 60 PNGs, Qwen3-VL v5 result file, and args.json. This supersedes earlier SR-only/status handoffs.

Canonical compact row:

| Benchmark | Method | SD3 Avg SR | FLUX1 Avg SR |
|---|---|---:|---:|
| I2P q16 top-60 | Ours hybrid/image-backed | 65.0 | 80.5 |

SD3 image-backed per concept: sexual 93.3, violence 76.7, self-harm 46.7, shocking 85.0, illegal 56.7, harassment 40.0, hate 56.7, Avg 65.0.

FLUX1 image-backed per concept: sexual 100.0, violence 86.7, self-harm 65.0, shocking 100.0, illegal 60.0, harassment 68.3, hate 83.3, Avg 80.5.

The older 63.8 / 76.2 rows came from stricter release-bundle/status selection and should not be used when full image-backed breakdown is needed.

## MJA scope clarification

MJA cross-backbone table should include **SD3.0 and FLUX1.0 only**. Do not include SD1.4 SafeDenoiser/SGF MJA cells, and do not mark them as TBD; they are outside the intended table scope.
