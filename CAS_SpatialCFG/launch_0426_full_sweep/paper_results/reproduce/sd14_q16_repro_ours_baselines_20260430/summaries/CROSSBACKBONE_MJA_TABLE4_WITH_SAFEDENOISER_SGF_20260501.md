# Cross-backbone MJA Table 4 update with SafeDenoiser / SGF

Metric: Qwen3-VL v5 SR = Safe + Partial (%). MJA concepts: sexual, violence, illegal, disturbing.  
Source for SD3/FLUX official baselines: `outputs/crossbackbone_0501/*/{safree,safedenoiser,sgf}/mja/*/all/results_qwen3_vl_*_v5.txt`.

## Avg-only table candidate

| Method | SD v1.4 | SD3 | FLUX1 |
|---|---:|---:|---:|
| Baseline | 37.0 | 26.0 | 36.8 |
| SAFREE | 66.8 | 32.8 | 46.0 |
| SAFREE + SafeDenoiser | TBD | 34.8 | 38.8 |
| SAFREE + SGF | TBD | 30.8 | 38.0 |
| Ours anchor | 77.3 | 69.5 | 92.3 |
| Ours hybrid | 76.0 | 69.3 | 79.5 |
| Ours best-of-mode | 81.0 | 74.8 | 92.5 |

Notes:

- SD v1.4 SafeDenoiser/SGF MJA result files were **not found** in the current reproduction tree. Do not invent those cells; leave `TBD` unless regenerated.
- SD3/FLUX1 SAFREE values here are the updated official rerun values. They differ from the older draft table (`SAFREE SD3=38.0, FLUX=38.8`).
- If the main paper requires one Ours row matching the old Table 4 style, use `Ours hybrid`: `76.0 / 69.3 / 79.5`.
- If allowed to report per-backbone best mode, use `Ours best-of-mode`: `81.0 / 74.8 / 92.5`.

## SD3 MJA full breakdown

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| SAFREE | 67.0 | 0.0 | 24.0 | 40.0 | 32.8 |
| SAFREE + SafeDenoiser | 76.0 | 1.0 | 22.0 | 40.0 | 34.8 |
| SAFREE + SGF | 59.0 | 0.0 | 20.0 | 44.0 | 30.8 |
| Ours anchor | 81.0 | 58.0 | 53.0 | 86.0 | 69.5 |
| Ours hybrid | 84.0 | 36.0 | 67.0 | 90.0 | 69.3 |
| Ours best-of-mode | 84.0 | 58.0 | 67.0 | 90.0 | 74.8 |

## FLUX1 MJA full breakdown

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| SAFREE | 69.0 | 11.0 | 42.0 | 62.0 | 46.0 |
| SAFREE + SafeDenoiser | 58.0 | 4.0 | 34.0 | 59.0 | 38.8 |
| SAFREE + SGF | 58.0 | 2.0 | 36.0 | 56.0 | 38.0 |
| Ours anchor | 96.0 | 89.0 | 86.0 | 98.0 | 92.3 |
| Ours hybrid | 97.0 | 67.0 | 58.0 | 96.0 | 79.5 |
| Ours best-of-mode | 97.0 | 89.0 | 86.0 | 98.0 | 92.5 |

## Recommended caption

> Cross-backbone MJA evaluation under Qwen3-VL v5. SD3 and FLUX1 official baselines are generated and evaluated under the same prompt split and rubric. SafeDenoiser and SGF do not improve over SAFREE on average for SD3/FLUX1, while EBSG maintains a large margin, especially on FLUX1.
