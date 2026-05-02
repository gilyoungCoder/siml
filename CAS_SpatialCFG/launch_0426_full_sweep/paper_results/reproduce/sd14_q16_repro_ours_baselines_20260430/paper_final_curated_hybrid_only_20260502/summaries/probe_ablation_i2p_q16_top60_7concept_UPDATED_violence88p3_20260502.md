# Probe-channel ablation: SD v1.4 I2P q16 top-60 (7 concepts incl. sexual) -- FINAL rounded canonical

Metric: Qwen3-VL v5 SR = Safe + Partial (%). Per-concept-best hyperparameters from Appendix Table 7; each row varies only the probe channel (text-only, image-only, both). The Both row is aligned to main Table 2 EBSG single-concept SR under the final rounded hybrid configs.

| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 65.0 | 55.0 | 65.0 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (EBSG) | 98.3 | 88.3 | 58.3 | 93.3 | 46.7 | 63.3 | 66.7 | 73.6 |

## Evidence notes

- Text sexual = 98.3 is confirmed and unchanged from the maintained probe-ablation result.
- Text harassment is the rounded sh=30 rerun: SR 65.0 = Safe 55.0 + Partial 10.0; Full 11.7; NotRelevant 23.3.
  - Result file: `outputs/probe_ablation_q16top60_20260502_text_harassment_sh30/text/harassment/results_qwen3_vl_harassment_v5.txt`
- Both row follows the final Table 2 EBSG single-concept rounded canonical values: sexual 98.3, violence 88.3, self-harm 58.3, shocking 93.3, illegal 46.7, harassment 63.3, hate 66.7, Avg 73.6.
- Do not use the older sub-optimal Both row `98.3/88.3/51.7/91.7/43.3/70.0/73.3` (Avg 73.8); it was a probe sweep snapshot and is superseded by final Table 2 alignment.
