# SD3 / FLUX1 I2P q16 top-60 cross-backbone results status (2026-05-01)

Metric: Qwen3-VL v5 SR = Safe + Partial (%).


## SD3 Ours best

| Concept | Ours best SR | Config |
|---|---:|---|
| sexual | 96.7 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| violence | 73.3 | sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both |
| self-harm | 43.3 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| shocking | 90.0 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| illegal_activity | 50.0 | sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both |
| harassment | 36.7 | sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both |
| hate | 56.7 | sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both |
| **Avg** | **63.8** |  |

## FLUX1 Ours best

| Concept | Ours best SR | Config |
|---|---:|---|
| sexual | 100.0 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| violence | 60.0 | flux_xlow_hybrid_ss0.25_thr0.15_imgthr0.1_cas0.45_both |
| self-harm | 65.0 | flux_low_hybrid_ss0.75_thr0.15_imgthr0.1_cas0.45_both |
| shocking | 100.0 | flux_high_hybrid_ss2.5_thr0.15_imgthr0.1_cas0.5_both |
| illegal_activity | 60.0 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| harassment | 68.3 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| hate | 80.0 | flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both |
| **Avg** | **76.2** |  |

## FLUX1 baseline/official methods

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg completed | Completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 95.0 | 56.7 | 51.7 | 33.3 | 43.3 | 38.3 | 45.0 | 51.9 | 7/7 |
| SafeDenoiser | 91.7 | 45.0 | 46.7 | 35.0 | 45.0 | 36.7 | 48.3 | 49.8 | 7/7 |
| SGF | 93.3 | 46.7 | 51.7 | 31.7 | 48.3 | 35.0 | 46.7 | 50.5 | 7/7 |

## SD3 baseline/official methods

Current `crossbackbone_0501/sd3` generated-image counts are 0 for i2p q16 top-60; the run failed at tokenizer/sentencepiece initialization, so SAFREE/SafeDenoiser/SGF SD3 cells must be regenerated before paper use.

