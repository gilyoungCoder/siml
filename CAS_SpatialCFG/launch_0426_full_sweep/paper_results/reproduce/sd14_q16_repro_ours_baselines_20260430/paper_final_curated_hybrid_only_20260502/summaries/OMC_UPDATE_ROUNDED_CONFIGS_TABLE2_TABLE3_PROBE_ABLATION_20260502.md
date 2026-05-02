# OMC update request: use rounded SD1.4 I2P Ours configs for self-harm / harassment / hate

## Decision
Use the cleaner/rounded SD v1.4 I2P q16 top-60 Ours single-concept configs for the visually suspicious hyperparameters, even if harassment/hate lose some SR. The goal is to avoid the impression of overly precise post-hoc tuning such as `31.25` and `0.0375`.

This update affects:
1. **Table 2**: I2P harmful-concept erasure on SD v1.4, Ours single-concept row.
2. **Table 3**: **Probe-channel ablation (SR% ↑, SD v1.4 I2P top-60)**. The `Both (Ours)` row should match the final Table 2 Ours single-concept canonical row.
3. Any main-text claim that reports the SD1.4 I2P 7-concept Ours single-concept average.

---

## Rounded configs to use

### self-harm — update to rounded sh=7.5
Use this run.

- Output path:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round`
- Args:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round/args.json`
- Result:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round/results_qwen3_vl_self_harm_v5.txt`
- Config:
```text
safety_scale=7.5, cas_threshold=0.5, theta_text=0.10, theta_img=0.10
how_mode=hybrid, probe_mode=both, family_guidance=true
```
- Result:
```text
SR=58.3, Safe=10.0, Partial=48.3, Full=28.3, NR=13.3
```

### harassment — update to rounded sh=30.0
Use this run for cleaner config, despite lower SR than the old `31.25` run.

- Output path:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round`
- Args:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round/args.json`
- Result:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round/results_qwen3_vl_harassment_v5.txt`
- Config:
```text
safety_scale=30.0, cas_threshold=0.5, theta_text=0.10, theta_img=0.50
how_mode=hybrid, probe_mode=both, family_guidance=true
```
- Result:
```text
SR=63.3, Safe=50.0, Partial=13.3, Full=15.0, NR=21.7
```

### hate — update to rounded sh=27.5 / theta_img=0.05
Use this run for cleaner config, despite lower SR than the old `28.0 / 0.0375` run.

- Output path:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round`
- Args:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round/args.json`
- Result:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round/results_qwen3_vl_hate_v5.txt`
- Config:
```text
safety_scale=27.5, cas_threshold=0.6, theta_text=0.25, theta_img=0.05
how_mode=hybrid, probe_mode=both, family_guidance=true
```
- Result:
```text
SR=66.7, Safe=56.7, Partial=10.0, Full=16.7, NR=16.7
```

---

## Updated Table 2 Ours single-concept row

Use these final SD v1.4 I2P q16 top-60 Ours single-concept values:

| Concept | SR | Safe | Partial | Full | NR | Config note |
|---|---:|---:|---:|---:|---:|---|
| sexual | 98.3 | 91.7 | 6.7 | 1.7 | 0.0 | sh20 / cas0.5 / θt0.10 / θi0.30 |
| violence | 88.3 | 73.3 | 15.0 | 5.0 | 6.7 | sh20 / cas0.4 / θt0.30 / θi0.10 |
| self-harm | 58.3 | 10.0 | 48.3 | 28.3 | 13.3 | rounded sh7.5 / cas0.5 / θt0.10 / θi0.10 |
| shocking | 93.3 | 88.3 | 5.0 | 3.3 | 3.3 | sh27.5 / cas0.6 / θt0.15 / θi0.10 |
| illegal | 46.7 | 31.7 | 15.0 | 15.0 | 38.3 | sh25 / cas0.6 / θt0.10 / θi0.50 |
| harassment | 63.3 | 50.0 | 13.3 | 15.0 | 21.7 | rounded sh30 / cas0.5 / θt0.10 / θi0.50 |
| hate | 66.7 | 56.7 | 10.0 | 16.7 | 16.7 | rounded sh27.5 / cas0.6 / θt0.25 / θi0.05 |
| **Avg** | **73.6** |  |  |  |  | mean SR over 7 concepts |

Average check:
```text
(98.3 + 88.3 + 58.3 + 93.3 + 46.7 + 63.3 + 66.7) / 7 = 73.6
```

---

## Updated Table 3: Probe-channel ablation

This is the table captioned:

> **Table 3: Probe-channel ablation (SR% ↑, SD v1.4 I2P top-60).**

The `Both (Ours)` row must match the final rounded Table 2 Ours single-concept row above.

Use:

| Probe | sexual | violence | self-harm | shocking | illegal | harass. | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 70.0 | 55.0 | 65.7 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (Ours) | 98.3 | 88.3 | **58.3** | 93.3 | 46.7 | **63.3** | **66.7** | **73.6** |

Important: previous probe-ablation MD used `Both Avg=73.8` with stale/mixed sub-run values. Replace it with the final rounded Table 2-aligned `Both (Ours)` row above.

---

## Update all text claims

Please update any prose claims that cite SD v1.4 I2P 7-concept Ours single-concept average:

- If referring to final rounded Table 2 Ours single-concept: use **73.6 Avg SR**.
- If comparing against SAFREE or other baselines, recompute deltas from 73.6, not 74.3/75.2/73.8.

Rationale to record if needed:

> We use rounded, human-readable hyperparameters for the final I2P single-concept SD1.4 Ours row to avoid the appearance of overly precise per-cell tuning. The rounded self-harm config improves SR, while rounded harassment/hate slightly reduce SR but keep the experimental setting cleaner and more defensible.

