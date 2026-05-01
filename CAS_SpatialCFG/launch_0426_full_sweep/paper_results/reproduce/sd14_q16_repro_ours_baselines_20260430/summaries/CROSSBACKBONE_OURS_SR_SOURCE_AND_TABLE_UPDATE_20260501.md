# Cross-backbone Ours SR source audit + paper table update plan (2026-05-01)

## What was found

The Ours SD3/FLUX I2P SR values and best-config labels are recorded in:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/crossbackbone_sd3_flux1_i2p_status_20260501.md
```

The same SR rows were propagated into:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/crossbackbone_sd3_flux1_official_final_20260501.md
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/MLWRITER_FINAL_TABLE_VALUES_20260501.md
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/MLWRITER_BESTMODE_FINAL_WITH_CONFIGS_20260501.md
```

## Important distinction

- **SR values and best config labels are found and should be used.**
- **Raw Ours Qwen VLM per-class files for cross-backbone SD3/FLUX were not found in the current result tree** after searching `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG`. Therefore do not invent Safe/Partial/Full/NR for those Ours cross-backbone rows unless raw eval files are recovered or re-evaluated.
- Official baselines (SAFREE / SafeDenoiser / SGF) do have full SR/Safe/Partial/Full/NR breakdown in `CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md`.

## I2P q16 top-60 — Ours best SR/config source

### SD3 Ours best

| Concept | SR | Best config label | Source line file |
|---|---:|---|---|
| sexual | 96.7 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| violence | 73.3 | `sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| self-harm | 43.3 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| shocking | 90.0 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| illegal_activity | 50.0 | `sd3_mild_hybrid_ss15.0_thr0.15_imgthr0.1_cas0.45_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| harassment | 36.7 | `sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| hate | 56.7 | `sd3_strong_hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| **Avg** | **63.8** |  |  |

### FLUX1 Ours best

| Concept | SR | Best config label | Source line file |
|---|---:|---|---|
| sexual | 100.0 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| violence | 60.0 | `flux_xlow_hybrid_ss0.25_thr0.15_imgthr0.1_cas0.45_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| self-harm | 65.0 | `flux_low_hybrid_ss0.75_thr0.15_imgthr0.1_cas0.45_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| shocking | 100.0 | `flux_high_hybrid_ss2.5_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| illegal_activity | 60.0 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| harassment | 68.3 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| hate | 80.0 | `flux_mid_hybrid_ss1.5_thr0.15_imgthr0.1_cas0.5_both` | `crossbackbone_sd3_flux1_i2p_status_20260501.md` |
| **Avg** | **76.2** |  |  |

## MJA cross-backbone Ours best-of-mode SR source

The MJA Ours best-of-mode rows are recorded in:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/CROSSBACKBONE_MJA_TABLE4_WITH_SAFEDENOISER_SGF_20260501.md
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/MLWRITER_FINAL_TABLE_VALUES_20260501.md
```

| Backbone | sexual | violence | illegal | disturbing | Avg | Mode choices |
|---|---:|---:|---:|---:|---:|---|
| SD3 | 84.0 | 58.0 | 67.0 | 90.0 | 74.8 | sexual/hybrid, violence/anchor, illegal/hybrid, disturbing/hybrid |
| FLUX1 | 97.0 | 89.0 | 86.0 | 98.0 | 92.5 | sexual/hybrid, violence/anchor, illegal/anchor, disturbing/anchor |

## How to update paper tables

### Main/appendix compact cross-backbone table
Use SR-only for Ours cross-backbone if the table is compact:

| Benchmark | Method | SD3 Avg SR | FLUX1 Avg SR |
|---|---|---:|---:|
| I2P q16 top-60 | SAFREE | 47.9 | 51.9 |
| I2P q16 top-60 | SAFREE + SafeDenoiser | 45.5 | 49.8 |
| I2P q16 top-60 | SAFREE + SGF | 42.9 | 50.5 |
| I2P q16 top-60 | **Ours best** | **63.8** | **76.2** |
| MJA | SAFREE | 32.8 | 46.0 |
| MJA | SAFREE + SafeDenoiser | 34.8 | 38.8 |
| MJA | SAFREE + SGF | 30.8 | 38.0 |
| MJA | **Ours best-of-mode** | **74.8** | **92.5** |

### Full breakdown table
- For SAFREE / SafeDenoiser / SGF, use `CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md` because it has SR/Safe/Partial/Full/NR.
- For Ours cross-backbone, use SR/config labels from this file, and leave Safe/Partial/Full/NR blank/TBD unless raw Qwen files are recovered or re-evaluated.
- If paper requires every row to have Full/NR, then rerun only VLM eval on the Ours cross-backbone image folders if those image folders are found; otherwise regenerate/evaluate.

