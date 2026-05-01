# Final ML Writer / OMC Handoff: Paper-aligned tables, paths, configs (2026-05-02)

This is the single handoff document to use for paper updates. It supersedes stale violence-best files and older SGF P4DN SR-only notes.

**Canonical release root**

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/paper_aligned_release_20260502
```

Use this release folder first. It hardlink-copies the actual output folders/results while keeping stale/confusing runs out of the main path.

---

## 0. Global conventions

- Metric: Qwen3-VL v5 four-class rubric.
- SR = Safe + Partial.
- NR = NotRelevant for non-nudity concepts; for nudity/sexual rubric the file label is `NotPeople`.
- SD1.4 generation: seed 42, one image per prompt, CFG 7.5, 512x512, DDIM 50 unless noted.
- I2P split: **q16 top-60**, 60 prompts per concept.
- Seven I2P concepts: sexual, violence, self-harm, shocking, illegal_activity, harassment, hate.

Main verification manifests:

```text
$REL/manifests/vlm_result_files.txt
$REL/manifests/files.txt
$REL/manifests/violence_final_88p3_verification.txt
```

---

## 1. SD1.4 I2P q16 top-60 single-concept table

### 1.1 Final SR table

| concept | Baseline | SAFREE | SAFREE+SafeDenoiser-CS | SAFREE+SGF-CS | Ours best | Ours config |
|---|---:|---:|---:|---:|---:|---|
| sexual | 68.3 | 83.3 | 91.7 | 96.7 | **98.3** | sh=20.0, cas=0.5, text=0.10, img=0.30 |
| violence | 36.7 | 73.3 | 40.0 | 40.0 | **88.3** | sh=20.0, cas=0.4, text=0.30, img=0.10 |
| self-harm | 43.3 | 36.7 | 38.3 | 36.7 | **51.7** | sh=7.0, cas=0.5, text=0.10, img=0.10 |
| shocking | 15.0 | 81.7 | 21.7 | 18.3 | **93.3** | sh=27.5, cas=0.6, text=0.15, img=0.10 |
| illegal_activity | 31.7 | 35.0 | 41.7 | 30.0 | **46.7** | sh=25.0, cas=0.6, text=0.10, img=0.50 |
| harassment | 25.0 | 28.3 | 26.7 | 23.3 | **68.3** | sh=31.25, cas=0.5, text=0.10, img=0.50 |
| hate | 25.0 | 43.3 | 28.3 | 21.7 | **73.3** | sh=28.0, cas=0.6, text=0.25, img=0.0375 |
| **Avg** | **35.0** | **54.5** | **41.2** | **38.1** | **74.2** | — |

**Paper text update**: Ours single-concept I2P avg = **74.2**, SAFREE = **54.5**, delta = **+19.7 pp**.

### 1.2 Ours best per-class breakdown

Format: **SR / Safe / Partial / Full / NR**.

| concept | breakdown |
|---|---:|
| sexual | 98.3 / 91.7 / 6.7 / 1.7 / 0.0 |
| violence | 88.3 / 73.3 / 15.0 / 5.0 / 6.7 |
| self-harm | 51.7 / 8.3 / 43.3 / 23.3 / 25.0 |
| shocking | 93.3 / 88.3 / 5.0 / 3.3 / 3.3 |
| illegal_activity | 46.7 / 31.7 / 15.0 / 15.0 / 38.3 |
| harassment | 68.3 / 56.7 / 11.7 / 13.3 / 18.3 |
| hate | 73.3 / 56.7 / 16.7 / 10.0 / 16.7 |

### 1.3 Ours best config and exact result paths

Common Ours settings: SD1.4, hybrid, probe=both, family_guidance=True, seed=42, CFG=7.5, 512, q16 top-60 prompts.

| concept | sh | CAS | theta_text | theta_img | family pack | prompt | result path inside release |
|---|---:|---:|---:|---:|---|---|---|
| sexual | 20.0 | 0.5 | 0.10 | 0.30 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/sexual_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/sexual/hybrid_best_tau05_cas0.5/results_qwen3_vl_nudity_v5.txt` |
| violence | 20.0 | 0.4 | 0.30 | 0.10 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/violence_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/violence/sh20_tau04_txt030_img010/results_qwen3_vl_violence_v5.txt` |
| self-harm | 7.0 | 0.5 | 0.10 | 0.10 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/self-harm_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/self-harm/hybrid_best_tau05_cas0.5/results_qwen3_vl_self_harm_v5.txt` |
| shocking | 27.5 | 0.6 | 0.15 | 0.10 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/shocking/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/shocking_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/shocking/hybrid_best_ss125_ss27.5/results_qwen3_vl_shocking_v5.txt` |
| illegal_activity | 25.0 | 0.6 | 0.10 | 0.50 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/illegal_activity/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/illegal_activity/hybrid_best_ss125_ss25.0/results_qwen3_vl_illegal_v5.txt` |
| harassment | 31.25 | 0.5 | 0.10 | 0.50 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/harassment/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/harassment_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/harassment/hybrid_best_ss125_ss31.25/results_qwen3_vl_harassment_v5.txt` |
| hate | 28.0 | 0.6 | 0.25 | 0.0375 | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt` | `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_q16_top60/hate_q16_top60.txt` | `outputs/sd14_i2p_single/ours_best/hate/hybrid_best_img075_img0.0375/results_qwen3_vl_hate_v5.txt` |

### 1.4 Baseline / SAFREE / SafeDenoiser / SGF output roots

Inside release:

```text
outputs/sd14_i2p_single/baseline/
outputs/sd14_i2p_single/safree/
outputs/sd14_i2p_single/safedenoiser_cs/
outputs/sd14_i2p_single/sgf_cs/
outputs/sd14_i2p_single/ours_best/
```

Note: `CS` means concept-specific reference/negative setup, not the older sexual-only negative reference.

---

## 2. Probe-channel ablation, SD1.4 I2P q16 top-60, 7 concepts

Use this updated table; it includes sexual and the final violence=88.3 row.

| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 98.3 | 71.7 | 35.0 | 85.0 | 45.0 | 70.0 | 55.0 | 65.7 |
| Image | 96.7 | 86.7 | 50.0 | 88.3 | 31.7 | 40.0 | 66.7 | 65.7 |
| Both (Ours) | **98.3** | **88.3** | **51.7** | **91.7** | 43.3 | **70.0** | **73.3** | **73.8** |

Source summary:

```text
summaries/probe_ablation_i2p_q16_top60_7concept_UPDATED_violence88p3_20260502.md
summaries/probe_ablation_i2p_q16_top60_7concept_UPDATED_violence88p3_20260502.csv
```

Output roots:

```text
outputs/probe_ablation_q16top60_base/
outputs/probe_ablation_q16top60_base/both/violence_FINAL_88p3_sh20_tau04_txt030_img010/
```

Deprecated confusing violence sweeps were moved out of the active output path:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/_deprecated_confusing_violence_20260502
```

---

## 3. Multi-concept SD1.4 I2P q16 top-60

Source summary:

```text
summaries/i2p_multi_sr_full_nr_tables_20260501.md
```

Each compact cell is **SR / Full / NR**.

### 3.1 2-concept: sexual + violence

| Method | sexual | violence | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|
| SAFREE multi | 86.7/1.7/11.6 | 43.3/48.3/8.3 | 65.0 | 25.0 | 9.9 |
| SAFREE + SafeDenoiser multi | 88.3/0.0/11.7 | 66.7/28.3/5.0 | **77.5** | 14.2 | 8.3 |
| SAFREE + SGF multi | 86.7/1.7/11.6 | 58.3/33.3/8.3 | 72.5 | 17.5 | 9.9 |
| Ours multi | 90.0/8.3/1.7 | 63.3/28.3/8.3 | 76.7 | 18.3 | **5.0** |

### 3.2 3-concept: sexual + violence + shocking

| Method | sexual | violence | shocking | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|
| SAFREE multi | 86.7/1.7/11.6 | 46.7/46.7/6.7 | 25.0/71.7/3.3 | 52.8 | 40.0 | 7.2 |
| SAFREE + SafeDenoiser multi | 85.0/1.7/13.3 | 60.0/31.7/8.3 | 61.7/38.3/0.0 | 68.9 | 23.9 | 7.2 |
| SAFREE + SGF multi | 90.0/1.7/8.3 | 53.3/38.3/8.3 | 40.0/60.0/0.0 | 61.1 | 33.3 | 5.5 |
| **Ours multi (C2_ss130)** | 90.0/6.7/3.3 | 76.7/16.7/6.7 | 78.3/21.7/0.0 | **81.7** | **15.0** | **3.3** |

### 3.3 7-concept multi

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE multi | 85.0/0.0/15.0 | 48.3/43.3/8.3 | 36.7/13.3/50.0 | 25.0/68.3/6.7 | 40.0/16.7/43.3 | 33.3/13.3/53.3 | 25.0/41.7/33.3 | 41.9 | 28.1 | 30.0 |
| SAFREE + SafeDenoiser multi | 90.0/0.0/10.0 | 58.3/35.0/6.7 | 40.0/11.7/48.3 | 60.0/38.3/1.7 | 45.0/6.7/48.3 | 28.3/41.7/30.0 | 33.3/46.7/20.0 | 50.7 | 25.7 | 23.6 |
| SAFREE + SGF multi | 86.7/0.0/13.3 | 46.7/38.3/15.0 | 43.3/8.3/48.3 | 50.0/50.0/0.0 | 38.3/11.7/50.0 | 36.7/15.0/48.3 | 36.7/26.7/36.7 | 48.3 | 21.4 | 30.2 |
| **Ours multi (C2_ss130)** | 88.3/1.7/10.0 | 85.0/3.3/11.7 | 66.7/5.0/28.3 | 88.3/1.7/10.0 | 65.0/3.3/31.7 | 60.0/13.3/26.7 | 58.3/26.7/15.0 | **73.1** | **7.9** | **19.1** |

Paper text update for 7c multi: **73.1 vs 41.9 = +31.2 pp**.

Output roots inside release:

```text
outputs/sd14_i2p_multi/safree_phase_safree_multi_q16top60/
outputs/sd14_i2p_multi/safedenoiser_multi_2c/
outputs/sd14_i2p_multi/safedenoiser_multi_3c_svs/
outputs/sd14_i2p_multi/safedenoiser_multi_7c/
outputs/sd14_i2p_multi/sgf_multi_2c/
outputs/sd14_i2p_multi/sgf_multi_3c_svs/
outputs/sd14_i2p_multi/sgf_multi_7c/
outputs/sd14_i2p_multi/ours_multi/
```

---

## 4. Nudity Table 1, SD1.4

Source summaries:

```text
summaries/TABLE1_NUDITY_BREAKDOWN_RELIABLE_HANDOFF_20260501.md
summaries/table1_nudity_breakdown_reliable_handoff_20260501.csv
summaries/WRITER_SGF_P4DN_AND_NFE_CONFIGS_20260502.md
```

### 4.1 Reliable breakdown rows

Format per dataset: **SR / Safe / Partial / Full / NR**.

| Method | UD | RAB | MMA | P4DN |
|---|---:|---:|---:|---:|
| SAFREE | 87.3 / 62.7 / 24.6 / 4.9 / 7.7 | 83.5 / 49.4 / 34.2 / 11.4 / 5.1 | 75.4 / 51.6 / 23.8 / 20.2 / 4.4 | 70.9 / 41.1 / 29.8 / 21.2 / 7.9 |
| SAFREE+SafeDenoiser | 95.1 / 69.0 / 26.1 / 2.1 / 2.8 | 81.0 / 50.6 / 30.4 / 13.9 / 5.1 | 73.4 / 47.2 / 26.2 / 24.1 / 2.5 | 62.9 / 31.8 / 31.1 / 32.5 / 4.6 |
| SAFREE+SGF | 92.3 / TBD / TBD / TBD / TBD | 83.5 / TBD / TBD / TBD / TBD | 76.9 / TBD / TBD / TBD / TBD | **70.2 / 37.7 / 32.5 / 25.8 / 4.0** |
| Ours/EBSG hybrid | 97.2 / 73.9 / 23.2 / 1.4 / 1.4 | 96.2 / 89.9 / 6.3 / 2.5 / 1.3 | 84.2 / 66.9 / 17.3 / 15.4 / 0.4 | 97.4 / 92.1 / 5.3 / 2.6 / 0.0 |

### 4.2 SGF P4DN decision

Use the repaired full n=151 breakdown-backed value if Table 1 reports per-class breakdown.

```text
outputs/sd14_nudity/sgf/p4dn/all/results_qwen3_vl_nudity_v5.txt
```

SGF P4DN repaired n=151:

- Total = 151
- Safe = 57 (37.7)
- Partial = 49 (32.5)
- Full = 39 (25.8)
- NotPeople = 6 (4.0)
- SR = 106/151 = **70.2**

Caveat:

- An older result has SR=72.8 but n=147, incomplete.
- Old handoff/canonical listed 74.2 SR-only, but no corresponding breakdown file was located.
- Recommendation: use **70.2** in breakdown table; if keeping 74.2, mark it SR-only and not breakdown-backed.

### 4.3 Nudity output roots inside release

```text
outputs/sd14_nudity/safree/
outputs/sd14_nudity/safedenoiser/
outputs/sd14_nudity/sgf/
outputs/sd14_nudity/ours_hybrid_family/
```

---

## 5. Cross-backbone: SD3 and FLUX1

Source summary:

```text
summaries/CROSSBACKBONE_I2P_MJA_FULL_BREAKDOWN_BESTMODE_20260501.md
```

Output root:

```text
outputs/crossbackbone_0501/
```

### 5.1 I2P q16 top-60 on SD3

Format: **SR / Safe / Partial / Full / NR**.

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg SR | Avg Safe | Avg Partial | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 93.3/73.3/20.0/5.0/1.7 | 41.7/31.7/10.0/58.3/0.0 | 38.3/8.3/30.0/30.0/31.7 | 36.7/25.0/11.7/63.3/0.0 | 41.7/30.0/11.7/30.0/28.3 | 35.0/26.7/8.3/31.7/33.3 | 48.3/43.3/5.0/43.3/8.3 | 47.9 | 34.0 | 13.8 | 37.4 | 14.8 |
| SAFREE+SafeDenoiser | 91.7/68.3/23.3/3.3/5.0 | 41.7/31.7/10.0/58.3/0.0 | 35.0/11.7/23.3/33.3/31.7 | 31.7/20.0/11.7/68.3/0.0 | 40.0/33.3/6.7/30.0/30.0 | 30.0/20.0/10.0/38.3/31.7 | 48.3/35.0/13.3/46.7/5.0 | 45.5 | 31.4 | 14.0 | 39.7 | 14.8 |
| SAFREE+SGF | 91.7/71.7/20.0/5.0/3.3 | 35.0/28.3/6.7/65.0/0.0 | 38.3/8.3/30.0/33.3/28.3 | 26.7/18.3/8.3/73.3/0.0 | 36.7/28.3/8.3/38.3/25.0 | 31.7/23.3/8.3/41.7/26.7 | 40.0/36.7/3.3/50.0/10.0 | 42.9 | 30.7 | 12.1 | 43.8 | 13.3 |
| **Ours best** | 96.7/TBD/TBD/TBD/TBD | 73.3/TBD/TBD/TBD/TBD | 43.3/TBD/TBD/TBD/TBD | 90.0/TBD/TBD/TBD/TBD | 50.0/TBD/TBD/TBD/TBD | 36.7/TBD/TBD/TBD/TBD | 56.7/TBD/TBD/TBD/TBD | **63.8** | TBD | TBD | TBD | TBD |

### 5.2 I2P q16 top-60 on FLUX1

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg SR | Avg Safe | Avg Partial | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 95.0/56.7/38.3/3.3/1.7 | 56.7/36.7/20.0/36.7/6.7 | 51.7/13.3/38.3/31.7/16.7 | 33.3/25.0/8.3/66.7/0.0 | 43.3/28.3/15.0/20.0/36.7 | 38.3/21.7/16.7/28.3/33.3 | 45.0/40.0/5.0/41.7/13.3 | 51.9 | 31.7 | 20.2 | 32.6 | 15.5 |
| SAFREE+SafeDenoiser | 91.7/58.3/33.3/6.7/1.7 | 45.0/25.0/20.0/53.3/1.7 | 46.7/15.0/31.7/35.0/18.3 | 35.0/18.3/16.7/65.0/0.0 | 45.0/23.3/21.7/18.3/36.7 | 36.7/20.0/16.7/35.0/28.3 | 48.3/36.7/11.7/43.3/8.3 | 49.8 | 28.1 | 21.7 | 36.7 | 13.6 |
| SAFREE+SGF | 93.3/60.0/33.3/3.3/3.3 | 46.7/25.0/21.7/51.7/1.7 | 51.7/10.0/41.7/28.3/20.0 | 31.7/23.3/8.3/68.3/0.0 | 48.3/26.7/21.7/18.3/33.3 | 35.0/20.0/15.0/38.3/26.7 | 46.7/36.7/10.0/46.7/6.7 | 50.5 | 28.8 | 21.7 | 36.4 | 13.1 |
| **Ours best** | 100.0/TBD/TBD/TBD/TBD | 60.0/TBD/TBD/TBD/TBD | 65.0/TBD/TBD/TBD/TBD | 100.0/TBD/TBD/TBD/TBD | 60.0/TBD/TBD/TBD/TBD | 68.3/TBD/TBD/TBD/TBD | 80.0/TBD/TBD/TBD/TBD | **76.2** | TBD | TBD | TBD | TBD |

### 5.3 MJA on SD3 / FLUX1

Compact avg table for paper:

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

Ours MJA best modes/configs from source summary:

| Backbone | concept | SR | mode/config |
|---|---|---:|---|
| SD3 | sexual | 84.0 | hybrid, sh=15, tau=0.6, theta_text=0.10, theta_img=0.30 |
| SD3 | violence | 58.0 | anchor, sa=1.5, tau=0.6, theta_text=0.10, theta_img=0.20 |
| SD3 | illegal | 67.0 | hybrid, sh=20, tau=0.3, theta_text=0.15, theta_img=0.10 |
| SD3 | disturbing | 90.0 | hybrid, sh=20, tau=0.4, theta_text=0.15, theta_img=0.10 |
| FLUX1 | sexual | 97.0 | hybrid, sh=2.5, tau=0.6, theta_text=0.10, theta_img=0.10 |
| FLUX1 | violence | 89.0 | anchor, sa=2.0, tau=0.6, theta_text=0.10, theta_img=0.10 |
| FLUX1 | illegal | 86.0 | anchor, sa=3.0, tau=0.6, theta_text=0.10, theta_img=0.10 |
| FLUX1 | disturbing | 98.0 | anchor, sa=1.5, tau=0.6, theta_text=0.10, theta_img=0.10 |

Cross-backbone output roots inside release:

```text
outputs/crossbackbone_0501/sd3/safree/
outputs/crossbackbone_0501/sd3/safedenoiser/
outputs/crossbackbone_0501/sd3/sgf/
outputs/crossbackbone_0501/flux1/safree/
outputs/crossbackbone_0501/flux1/safedenoiser/
outputs/crossbackbone_0501/flux1/sgf/
```

Note on Ours cross-backbone per-class breakdown: current handoff has verified SR/source configs but does not contain all Ours Safe/Partial/Full/NR raw per-class files for cross-backbone. Keep Ours cross-backbone cells as SR-only unless raw per-class files are provided.

---

## 6. Runtime / NFE result positioning

NFE execution is handled separately by OMC. Do **not** include NFE run commands in the paper handoff. Suggested paper usage:

- If NFE curve completes cleanly, include it as an efficiency/compute appendix figure rather than a main claim.
- If results are noisy or incomplete, mention in limitations/efficiency discussion only: EBSG is training-free and adds extra denoiser/probe overhead; NFE/runtime tradeoff is an implementation-level limitation and an optimization target.
- For fairness, NFE/runtime should compare Baseline, SAFREE, SafeDenoiser, SGF, and Ours under identical SD1.4 seed/CFG/resolution/prompt split and per-concept best Ours configs.

Existing runtime folder in release, if needed:

```text
outputs/runtime/
summaries/runtime_sd14_sexual10_5methods.md
```

---

## 7. Writer action list

1. Replace stale SD1.4 I2P single Ours avg **73.3** with **74.2** if using final violence=88.3.
2. Replace stale violence Ours **81.7** with **88.3**.
3. Update probe ablation to the 7-concept table with Both avg **73.8**.
4. Multi-concept table: use SR/Full/NR cells from `i2p_multi_sr_full_nr_tables_20260501.md`; final 7c Ours **73.1**, SAFREE **41.9**.
5. Nudity Table 1: if showing SGF P4DN breakdown, use repaired n=151 **70.2 / 37.7 / 32.5 / 25.8 / 4.0**. Do not use 74.2 as breakdown-backed.
6. Cross-backbone: use compact avg table above; keep Ours cross-backbone detailed per-class as TBD/SR-only unless raw Ours per-class files are later supplied.
7. Do not use deprecated violence folders/summaries.

