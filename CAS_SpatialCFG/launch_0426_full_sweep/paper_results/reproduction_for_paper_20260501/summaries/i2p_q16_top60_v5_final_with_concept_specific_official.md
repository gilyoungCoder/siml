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
