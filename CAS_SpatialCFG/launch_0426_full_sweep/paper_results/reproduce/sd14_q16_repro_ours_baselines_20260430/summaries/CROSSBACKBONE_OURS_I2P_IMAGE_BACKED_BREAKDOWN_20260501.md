# Image-backed cross-backbone Ours I2P q16 top-60 audit (2026-05-01)

Metric cell format: **SR / Safe / Partial / Full / NR** (%). All rows below point to an existing image directory with 60 generated images and a Qwen3-VL v5 result file.

CSV: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/crossbackbone_ours_i2p_image_backed_best_20260501.csv`

## Important correction vs earlier SR-only handoff

Earlier `crossbackbone_sd3_flux1_i2p_status_20260501.md` stored SR/config labels but did not include image paths. The table below is stricter: it is backed by current on-disk image folders + result files. Use this for verifiable paper updates if full SR/Safe/Partial/Full/NR is required.

## SD3 Ours best — image-backed

| Concept | SR/Safe/Partial/Full/NR | Config dir | #imgs | Image dir | Result file | args.json |
|---|---:|---|---:|---|---|---|
| sexual | 93.3/86.7/6.7/5.0/1.7 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/sexual/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/sexual/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_nudity_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/sexual/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| violence | 76.7/70.0/6.7/23.3/0.0 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/violence/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/violence/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_violence_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/violence/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| self-harm | 46.7/43.3/3.3/5.0/48.3 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both/results_qwen3_vl_self_harm_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both/args.json` |
| shocking | 85.0/75.0/10.0/10.0/5.0 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/shocking/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/shocking/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_shocking_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/shocking/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| illegal_activity | 56.7/51.7/5.0/10.0/33.3 | `hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/illegal_activity/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/illegal_activity/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both/results_qwen3_vl_illegal_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/illegal_activity/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both/args.json` |
| harassment | 40.0/31.7/8.3/23.3/36.7 | `hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both/results_qwen3_vl_harassment_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both/args.json` |
| hate | 56.7/51.7/5.0/18.3/25.0 | `hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_hate_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |

**Avg:** SR 65.0, Safe 58.6, Partial 6.4, Full 13.6, NR 21.4.

## FLUX1 Ours best — image-backed

| Concept | SR/Safe/Partial/Full/NR | Config dir | #imgs | Image dir | Result file | args.json |
|---|---:|---|---:|---|---|---|
| sexual | 100.0/98.3/1.7/0.0/0.0 | `hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/sexual/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/sexual/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_nudity_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/sexual/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| violence | 86.7/61.7/25.0/13.3/0.0 | `hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/violence/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/violence/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_violence_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/violence/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| self-harm | 65.0/23.3/41.7/10.0/25.0 | `hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/self-harm/hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/self-harm/hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_self_harm_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/self-harm/hybrid_ss0.75_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| shocking | 100.0/96.7/3.3/0.0/0.0 | `hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/shocking/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/shocking/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_shocking_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/shocking/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| illegal_activity | 60.0/43.3/16.7/11.7/28.3 | `hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/illegal_activity/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/illegal_activity/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_illegal_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/illegal_activity/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| harassment | 68.3/45.0/23.3/16.7/15.0 | `hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/harassment/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/harassment/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_harassment_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/harassment/hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both/args.json` |
| hate | 83.3/76.7/6.7/13.3/3.3 | `hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both` | 60 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/hate/hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/hate/hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both/results_qwen3_vl_hate_v5.txt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/hate/hybrid_ss1.0_thr0.15_imgthr0.1_cas0.5_both/args.json` |

**Avg:** SR 80.5, Safe 63.6, Partial 16.9, Full 9.3, NR 10.2.

## Recommended paper update

If the table needs full breakdown, replace the Ours cross-backbone I2P rows with the image-backed rows above. Compact avg row becomes:

| Benchmark | Method | SD3 Avg SR | FLUX1 Avg SR |
|---|---|---:|---:|
| I2P q16 top-60 | **Ours best (image-backed)** | **65.0** | **80.5** |

If preserving the earlier SR-only handoff values (63.8/76.2), explicitly label them as summary-only and do not attach per-class breakdown.
