# Cross-backbone official baseline results — final generated/evaluated (2026-05-01)

Metric: Qwen3-VL v5 SR = Safe + Partial (%). Prompt split: I2P q16 top-60 per concept; MJA 100 per concept.

## Completion status

- SD3 SAFREE / SafeDenoiser / SGF: generation complete, VLM v5 evaluation complete.
- FLUX1 SAFREE / SafeDenoiser / SGF: generation complete, VLM v5 evaluation complete.
- No missing SD3/FLUX official baseline cells under `outputs/crossbackbone_0501`.

## SD3 I2P q16 top-60

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 93.3 | 41.7 | 38.3 | 36.7 | 41.7 | 35.0 | 48.3 | 47.9 |
| SafeDenoiser | 91.7 | 41.7 | 35.0 | 31.7 | 40.0 | 30.0 | 48.3 | 45.5 |
| SGF | 91.7 | 35.0 | 38.3 | 26.7 | 36.7 | 31.7 | 40.0 | 42.9 |
| Ours best | 96.7 | 73.3 | 43.3 | 90.0 | 50.0 | 36.7 | 56.7 | 63.8 |

## SD3 MJA

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| SAFREE | 67.0 | 0.0 | 24.0 | 40.0 | 32.8 |
| SafeDenoiser | 76.0 | 1.0 | 22.0 | 40.0 | 34.8 |
| SGF | 59.0 | 0.0 | 20.0 | 44.0 | 30.8 |
| Ours anchor | 81.0 | 58.0 | 53.0 | 86.0 | 69.5 |
| Ours hybrid | 84.0 | 36.0 | 67.0 | 90.0 | 69.3 |

## FLUX1 I2P q16 top-60

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 95.0 | 56.7 | 51.7 | 33.3 | 43.3 | 38.3 | 45.0 | 51.9 |
| SafeDenoiser | 91.7 | 45.0 | 46.7 | 35.0 | 45.0 | 36.7 | 48.3 | 49.8 |
| SGF | 93.3 | 46.7 | 51.7 | 31.7 | 48.3 | 35.0 | 46.7 | 50.5 |
| Ours best | 100.0 | 60.0 | 65.0 | 100.0 | 60.0 | 68.3 | 80.0 | 76.2 |

## FLUX1 MJA

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| SAFREE | 69.0 | 11.0 | 42.0 | 62.0 | 46.0 |
| SafeDenoiser | 58.0 | 4.0 | 34.0 | 59.0 | 38.8 |
| SGF | 58.0 | 2.0 | 36.0 | 56.0 | 38.0 |
| Ours anchor | 96.0 | 89.0 | 86.0 | 98.0 | 92.3 |
| Ours hybrid | 97.0 | 67.0 | 58.0 | 96.0 | 79.5 |

## Notes

- SD3 official baselines are consistently below Ours on average. SafeDenoiser/SGF do not improve over SAFREE here except small isolated cells.
- FLUX1 official baselines are also below Ours by a large margin on both I2P and MJA.
- Current siml-07 GPU1 has an unrelated timing job under `outputs/phase_timing_isolated/safedenoiser_violence`; SD3/FLUX official generation/eval jobs are done.
