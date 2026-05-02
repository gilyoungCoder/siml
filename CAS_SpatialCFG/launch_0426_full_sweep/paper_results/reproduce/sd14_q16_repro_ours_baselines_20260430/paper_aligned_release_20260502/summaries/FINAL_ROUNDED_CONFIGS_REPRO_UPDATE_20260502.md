# Final rounded-config reproducibility update (2026-05-02)

This note supersedes earlier SD1.4 I2P Ours/probe-ablation handoffs where they conflict. The final paper preference is to use cleaner rounded hyperparameters for suspiciously precise cells, while keeping every reported value image-backed and Qwen3-VL V5-evaluated.

## Scope
- Backbone: SD v1.4
- Dataset: I2P q16 top-60, 7 concepts including sexual
- Method row affected: Ours / EBSG, single-concept
- Probe ablation affected: **Table 3: Probe-channel ablation (SR% ↑, SD v1.4 I2P top-60)**
- Evaluator: Qwen3-VL V5
- All paths below have generated PNGs + args/result files.

---

## Final Table 2 Ours single-concept row after rounded-config update

| Concept | SR | Safe | Partial | Full | NR | Canonical config/result path |
|---|---:|---:|---:|---:|---:|---|
| sexual | 98.3 | 91.7 | 6.7 | 1.7 | 0.0 | `outputs/ours_tune/i2p_q16/sexual/hybrid_best_tau05_cas0.5` |
| violence | 88.3 | 73.3 | 15.0 | 5.0 | 6.7 | `outputs/probe_ablation_q16top60_20260501_both_violence_adaptive_clean005/sh20_tau04_txt030_img010` |
| self-harm | 58.3 | 10.0 | 48.3 | 28.3 | 13.3 | `outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round` |
| shocking | 93.3 | 88.3 | 5.0 | 3.3 | 3.3 | `outputs/ours_tune/i2p_q16/shocking/hybrid_best_ss125_ss27.5` |
| illegal | 46.7 | 31.7 | 15.0 | 15.0 | 38.3 | `outputs/ours_tune/i2p_q16/illegal_activity/hybrid_best_ss125_ss25.0` |
| harassment | 63.3 | 50.0 | 13.3 | 15.0 | 21.7 | `outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round` |
| hate | 66.7 | 56.7 | 10.0 | 16.7 | 16.7 | `outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round` |
| **Avg** | **73.6** |  |  |  |  | mean SR |

Average check:
```text
(98.3 + 88.3 + 58.3 + 93.3 + 46.7 + 63.3 + 66.7) / 7 = 73.6
```

---

## Final Table 3 Probe-channel ablation after rounded-config update

Caption target:

> **Table 3: Probe-channel ablation (SR% ↑, SD v1.4 I2P top-60).**

Use this final table:

| Probe | sexual | violence | self-harm | shocking | illegal | harass. | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | **65.0** | 55.0 | **65.0** |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (EBSG/Ours) | 98.3 | 88.3 | **58.3** | 93.3 | 46.7 | **63.3** | **66.7** | **73.6** |

Changes vs older probe-ablation MD:
- Text harassment: `70.0 -> 65.0` using rounded `sh=30` text-only run.
- Text Avg: `65.7 -> 65.0`.
- Both row now matches final rounded Table 2 Ours single-concept row.
- Both Avg: final `73.6`.

---

## New rounded runs and evidence

### self-harm Both rounded
- Output:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7.5_cas0.5_txt0.10_img0.10_round`
- Args: `args.json`
- Result: `results_qwen3_vl_self_harm_v5.txt`
- Config: `sh=7.5, cas=0.5, theta_text=0.10, theta_img=0.10, probe=both, how=hybrid`
- Result: `SR=58.3, Safe=10.0, Partial=48.3, Full=28.3, NR=13.3`

### harassment Both rounded
- Output:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round`
- Args: `args.json`
- Result: `results_qwen3_vl_harassment_v5.txt`
- Config: `sh=30.0, cas=0.5, theta_text=0.10, theta_img=0.50, probe=both, how=hybrid`
- Result: `SR=63.3, Safe=50.0, Partial=13.3, Full=15.0, NR=21.7`

### hate Both rounded
- Output:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round`
- Args: `args.json`
- Result: `results_qwen3_vl_hate_v5.txt`
- Config: `sh=27.5, cas=0.6, theta_text=0.25, theta_img=0.05, probe=both, how=hybrid`
- Result: `SR=66.7, Safe=56.7, Partial=10.0, Full=16.7, NR=16.7`

### harassment Text rounded for Table 3
- Output:
  `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260502_text_harassment_sh30/text/harassment`
- Args: `args.json`
- Result: `results_qwen3_vl_harassment_v5.txt`
- Config: `sh=30.0, cas=0.5, theta_text=0.10, theta_img=0.50, probe=text, how=hybrid`
- Result: `SR=65.0, Safe=55.0, Partial=10.0, Full=11.7, NR=23.3`

---

## OMC / ml-writer instruction

1. Replace SD1.4 I2P Ours single Avg with **73.6** where the final rounded single-concept row is discussed.
2. Replace Table 3 Probe ablation with the table above.
3. Do not use old `Both Avg=73.8`, old Table2 `74.3`, or the intermediate `75.2` self-harm-only update.
4. Keep a note that the rounded configs were selected for defensibility/readability of hyperparameters, not for maximizing every cell.
5. If deltas against SAFREE/SafeDenoiser/SGF are stated in prose, recompute them from the rounded Ours Avg **73.6**.
