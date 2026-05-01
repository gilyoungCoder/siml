# Table 1 nudity breakdown reliable handoff (2026-05-01)

Purpose: give ML writer a non-guessing source for Table 1 per-class cells. Values below are parsed from saved `results_qwen3_vl_nudity_v5.txt` files unless explicitly marked canonical-SR-only.

## Decision summary

- **Fill per-class Sa/Pa/Fu/NR now** for: `SAFREE`, `SAFREE + SafeDenoiser`, `Ours/EBSG hybrid` from preserved result files.
- **Do not fill per-class breakdown** for `Baseline` unless regenerated or an actual result file is provided. Current repo state only has canonical SR for baseline nudity benchmarks.
- **Do not fill per-class breakdown** for `SAFREE + SGF`: the SGF nudity result files/images are not preserved under the current reproduction root; only SR handoff exists. Use SR-only or rerun SGF nudity before claiming breakdown.
- SGF SR conflict: existing handoff says `92.3 / 83.5 / 76.9 / 74.2`; observed/mentioned result-file SR was `92.3 / 84.8 / 77.6 / 72.8` with P4DN `n=147`. For paper consistency, keep canonical SR **only if** you accept the existing handoff; otherwise rerun SGF nudity and replace all four cells.

## Parsed per-class breakdown (SR / Sa / Pa / Fu / NR)

| Method | UD SR | RAB SR | MMA SR | P4DN SR | UD Sa/Pa/Fu/NR | RAB Sa/Pa/Fu/NR | MMA Sa/Pa/Fu/NR | P4DN Sa/Pa/Fu/NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 87.3 | 83.5 | 75.4 | 70.9 | 62.7/24.6/4.9/7.7 | 49.4/34.2/11.4/5.1 | 51.6/23.8/20.2/4.4 | 41.1/29.8/21.2/7.9 |
| SAFREE+SafeDenoiser | 95.1 | 81.0 | 73.4 | 62.9 | 69.0/26.1/2.1/2.8 | 50.6/30.4/13.9/5.1 | 47.2/26.2/24.1/2.5 | 31.8/31.1/32.5/4.6 |
| Ours/EBSG hybrid | 97.2 | 96.2 | 84.2 | 97.4 | 73.9/23.2/1.4/1.4 | 89.9/6.3/2.5/1.3 | 66.9/17.3/15.4/0.4 | 92.1/5.3/2.6/0.0 |
| Baseline | 71.1 | 48.1 | 35.4 | 33.1 | TBD | TBD | TBD | TBD |
| SAFREE+SGF | 92.3 | 83.5 | 76.9 | 74.2 | TBD | TBD | TBD | TBD |

## 23-column LaTeX row snippets for rows with trustworthy breakdown

Column order assumed: `Method & UD Sa & UD Pa & UD Fu & UD NR & RAB Sa & ... & P4DN NR`.

```tex
SAFREE & 62.7 & 24.6 & 4.9 & 7.7 & 49.4 & 34.2 & 11.4 & 5.1 & 51.6 & 23.8 & 20.2 & 4.4 & 41.1 & 29.8 & 21.2 & 7.9 \\
SAFREE + Safe Denoiser & 69.0 & 26.1 & 2.1 & 2.8 & 50.6 & 30.4 & 13.9 & 5.1 & 47.2 & 26.2 & 24.1 & 2.5 & 31.8 & 31.1 & 32.5 & 4.6 \\
EBSG (hybrid, family) & 73.9 & 23.2 & 1.4 & 1.4 & 89.9 & 6.3 & 2.5 & 1.3 & 66.9 & 17.3 & 15.4 & 0.4 & 92.1 & 5.3 & 2.6 & 0.0 \\
% Baseline and SAFREE+SGF per-class cells: keep TBD unless regenerated/result files are supplied.
```

## Source files used

- SAFREE UD: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safree/nudity/unlearndiff/generated/results_qwen3_vl_nudity_v5.txt`
- SAFREE RAB: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safree/nudity/rab/generated/results_qwen3_vl_nudity_v5.txt`
- SAFREE MMA: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safree/nudity/mma/generated/results_qwen3_vl_nudity_v5.txt`
- SAFREE P4DN: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safree/nudity/p4dn/generated/results_qwen3_vl_nudity_v5.txt`
- SAFREE+SafeDenoiser UD: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safedenoiser/nudity/unlearndiff/all/results_qwen3_vl_nudity_v5.txt`
- SAFREE+SafeDenoiser RAB: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safedenoiser/nudity/rab/all/results_qwen3_vl_nudity_v5.txt`
- SAFREE+SafeDenoiser MMA: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safedenoiser/nudity/mma/all/results_qwen3_vl_nudity_v5.txt`
- SAFREE+SafeDenoiser P4DN: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/safedenoiser/nudity/p4dn/all/results_qwen3_vl_nudity_v5.txt`
- Ours/EBSG hybrid UD: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours/nudity/unlearndiff/results_qwen3_vl_nudity_v5.txt`
- Ours/EBSG hybrid RAB: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours/nudity/rab/results_qwen3_vl_nudity_v5.txt`
- Ours/EBSG hybrid MMA: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours/nudity/mma/results_qwen3_vl_nudity_v5.txt`
- Ours/EBSG hybrid P4DN: OK — `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours/nudity/p4dn/results_qwen3_vl_nudity_v5.txt`

## COCO FID/CLIP note

Earlier COCO FID/CLIP for SafeDenoiser/SGF was marked unstable/deprioritized in the final experiment handoff. Do not use the high/unstable COCO numbers as final main-table claims unless rerun on the agreed 9966 split against the same SD1.4 baseline.

## Conclusion text replacement

Use the updated seven-concept language: `+18.8%p (73.3 vs 54.5)` for single-concept I2P and `73.1% vs 41.9% (+31.2%p)` for 7-concept multi. Replace `six I2P harmful concepts` with `seven I2P harmful concepts`.
