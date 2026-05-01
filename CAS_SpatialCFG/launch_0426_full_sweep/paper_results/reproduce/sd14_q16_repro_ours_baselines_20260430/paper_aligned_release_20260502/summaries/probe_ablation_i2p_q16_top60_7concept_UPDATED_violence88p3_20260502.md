# Probe-channel ablation: SD v1.4 I2P q16 top-60 (7 concepts incl. sexual) -- UPDATED violence both

Metric: Qwen3-VL v5 SR = Safe + Partial (%). Same family/hybrid setting; violence `both` is updated to the verified best coarse-grid config `sh=20, tau=0.40, theta_text=0.30, theta_img=0.10`.

| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 70.0 | 55.0 | 65.7 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (Ours) | 98.3 | 88.3 | 51.7 | 91.7 | 43.3 | 70.0 | 73.3 | 73.8 |

## Updated violence both evidence
- Result: SR 88.3 = Safe 73.3 + Partial 15.0; Full 5.0; NotRelevant 6.7.
- Config: `sh=20, tau=0.40, theta_text=0.30, theta_img=0.10, probe=both, how=hybrid, family=True`.
- Output path: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260501_both_violence_adaptive_clean005/sh20_tau04_txt030_img010`
- Result file: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260501_both_violence_adaptive_clean005/sh20_tau04_txt030_img010/results_qwen3_vl_violence_v5.txt`
