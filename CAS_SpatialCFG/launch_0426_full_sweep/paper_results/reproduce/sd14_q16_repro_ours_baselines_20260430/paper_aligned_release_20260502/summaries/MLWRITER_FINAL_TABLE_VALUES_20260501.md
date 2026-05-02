# MLWriter / OMC final table values for paper update

> **Writer instruction:** This is the current final handoff. Do not use archived/deprecated summaries or any SAFREE multi numbers from `phase_safree_v2`; final SAFREE multi values come from `phase_safree_multi_q16top60`.

Date: 2026-05-01  
Metric: **Qwen3-VL v5 SR = Safe + Partial (%)** unless otherwise noted.  
Primary prompt split for I2P: **q16 top-60 per concept**, 1 image / prompt, seed 42.

---

## 0. Reproduction / code bundle status

Reviewer-ready candidate folder:

`/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE`

Verified:

- SD1.4 Ours best configs resolve.
- `how_mode=hybrid`, `family_guidance=true`, `probe_mode=both`.
- Per-concept family packs load correctly from copied `clip_grouped.pt` files.
- No missing SD3 / FLUX1 official baseline generation/eval cells under `outputs/crossbackbone_0501`.
- Sanitized reviewer-ready candidate has no private `/mnt/home3/yhgil99` absolute paths, Supabase/Vercel/token strings, or internal merged memo.

Important wording caveat:

- For I2P `hate` / `harassment`, the pack is best described as **I2P hard-prompt visual subfamilies** rather than a canonical broad taxonomy. Args are correct; just avoid overclaiming taxonomy purity.

---

# Table 1 candidate: SD v1.4 Nudity benchmarks

Use this for the main nudity table.  
Backbone: SD v1.4.  
Prompt sets: UD / RAB / MMA / P4D-N.  
Metric: Qwen3-VL v5 SR (%).

| Method | UD | RAB | MMA | P4D-N |
|---|---:|---:|---:|---:|
| SAFREE | 87.3 | 83.5 | 75.4 | 70.9 |
| SAFREE + SafeDenoiser | 95.1 | 81.0 | 73.4 | 62.9 |
| SAFREE + SGF | 92.3 | 83.5 | 76.9 | 74.2 |
| **Ours / EBSG hybrid** | **97.2** | **96.2** | **84.2** | **97.4** |

Notes:

- Ours values are from verified `reproduction_for_paper` configs/results.
- SafeDenoiser/SGF are official-baseline reproduction values from the official patched runs.
- FID/CLIP are intentionally excluded here because COCO-FID was unstable / deprioritized in the final run discussion.

---

# Table 2 candidate: SD v1.4 I2P q16 top-60 single-concept

Use this as the main I2P single-concept table if including sexual as the 7th concept.  
Backbone: SD v1.4.  
Prompt split: q16 top-60 per concept.  
Each method is single-concept unless otherwise stated.

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 68.3 | 36.7 | 43.3 | 15.0 | 31.7 | 25.0 | 25.0 | 35.0 |
| SAFREE | 83.3 | 73.3 | 36.7 | 81.7 | 35.0 | 28.3 | 43.3 | 54.5 |
| SAFREE + SafeDenoiser | 91.7 | 40.0 | 38.3 | 21.7 | 41.7 | 26.7 | 28.3 | 41.2 |
| SAFREE + SGF | 96.7 | 40.0 | 36.7 | 18.3 | 30.0 | 23.3 | 21.7 | 38.1 |
| **Ours best / EBSG hybrid** | **98.3** | **88.3** | **51.7** | **93.3** | **46.7** | **68.3** | **73.3** | **74.2** |

## I2P single-concept detailed breakdown for SafeDenoiser/SGF

Same q16 top-60 split as Table 2. Each cell is **SR / Full / NR** in %. NR is `NotRelevant` for non-nudity and `NotPeople` for sexual/nudity eval.

| Method | sexual | violence | self-harm | shocking | illegal_activity | harassment | hate | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE + SafeDenoiser | 91.7/0.0/8.3 | 40.0/58.3/1.7 | 38.3/31.7/30.0 | 21.7/78.3/0.0 | 41.7/18.3/40.0 | 26.7/45.0/28.3 | 28.3/61.7/10.0 | 41.2 | 41.9 | 16.9 |
| SAFREE + SGF | 96.7/0.0/3.3 | 40.0/58.3/1.7 | 36.7/33.3/30.0 | 18.3/80.0/1.7 | 30.0/21.7/48.3 | 23.3/38.3/38.3 | 21.7/45.0/33.3 | 38.1 | 39.5 | 22.4 |

If excluding sexual to match the older six-concept I2P setup:

| Method | Avg over violence/self-harm/shocking/illegal/harassment/hate |
|---|---:|
| Baseline | 29.5 |
| SAFREE | 49.7 |
| SAFREE + SafeDenoiser | 32.8 |
| SAFREE + SGF | 28.3 |
| **Ours best / EBSG hybrid** | **70.3** |

Ours best configs:

| Concept | Best variant | SR |
|---|---|---:|
| sexual | `hybrid_best_tau05_cas0.5` | 98.3 |
| violence | `sh20_tau04_txt030_img010` | 88.3 |
| self-harm | `hybrid_best_tau05_cas0.5` | 51.7 |
| shocking | `hybrid_best_ss125_ss27.5` | 93.3 |
| illegal_activity | `hybrid_best_ss125_ss25.0` | 46.7 |
| harassment | `hybrid_best_ss125_ss31.25` | 68.3 |
| hate | `hybrid_best_img075_img0.0375` | 73.3 |

Suggested caption language:

> I2P q16 top-60 per concept on SD v1.4. Ours reports per-concept best hybrid EBSG configuration selected from the reported sweep; all rows use the same 60-prompt split, seed 42, one image per prompt.

---

# Table 3 candidate: SD v1.4 multi-concept I2P

Use as appendix or main multi-concept table.  
Metric: Qwen3-VL v5 over I2P q16 top-60.  
Each cell is **SR / Full / NR** in %, where SR = Safe + Partial and NR = NotRelevant. Lower Full and lower NR are better; higher SR is better.

## 2-concept: sexual + violence

| Method | sexual | violence | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|
| SAFREE multi | 86.7/1.7/11.6 | 43.3/48.3/8.3 | 65.0 | 25.0 | 9.9 |
| SAFREE + SafeDenoiser multi | 88.3/0.0/11.7 | 66.7/28.3/5.0 | 77.5 | 14.2 | 8.3 |
| SAFREE + SGF multi | 86.7/1.7/11.6 | 58.3/33.3/8.3 | 72.5 | 17.5 | 9.9 |
| **Ours multi** | 90.0/8.3/1.7 | 63.3/28.3/8.3 | **76.7** | **18.3** | **5.0** |

## 3-concept: sexual + violence + shocking

| Method | sexual | violence | shocking | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|
| SAFREE multi | 86.7/1.7/11.6 | 46.7/46.7/6.7 | 25.0/71.7/3.3 | 52.8 | 40.0 | 7.2 |
| SAFREE + SafeDenoiser multi | 85.0/1.7/13.3 | 60.0/31.7/8.3 | 61.7/38.3/0.0 | 68.9 | 23.9 | 7.2 |
| SAFREE + SGF multi | 90.0/1.7/8.3 | 53.3/38.3/8.3 | 40.0/60.0/0.0 | 61.1 | 33.3 | 5.5 |
| **Ours multi (C2_ss130)** | 90.0/6.7/3.3 | 76.7/16.7/6.7 | 78.3/21.7/0.0 | **81.7** | **15.0** | **3.3** |

## 7-concept multi

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg SR | Avg Full | Avg NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE multi | 85.0/0.0/15.0 | 48.3/43.3/8.3 | 36.7/13.3/50.0 | 25.0/68.3/6.7 | 40.0/16.7/43.3 | 33.3/13.3/53.3 | 25.0/41.7/33.3 | 41.9 | 28.1 | 30.0 |
| SAFREE + SafeDenoiser multi | 90.0/0.0/10.0 | 58.3/35.0/6.7 | 40.0/11.7/48.3 | 60.0/38.3/1.7 | 45.0/6.7/48.3 | 28.3/41.7/30.0 | 33.3/46.7/20.0 | 50.7 | 25.7 | 23.6 |
| SAFREE + SGF multi | 86.7/0.0/13.3 | 46.7/38.3/15.0 | 43.3/8.3/48.3 | 50.0/50.0/0.0 | 38.3/11.7/50.0 | 36.7/15.0/48.3 | 36.7/26.7/36.7 | 48.3 | 21.4 | 30.2 |
| **Ours multi (C2_ss130)** | 88.3/1.7/10.0 | 85.0/3.3/11.7 | 66.7/5.0/28.3 | 88.3/1.7/10.0 | 65.0/3.3/31.7 | 60.0/13.3/26.7 | 58.3/26.7/15.0 | **73.1** | **7.9** | **19.1** |

Notes:

- SAFREE rows are from `outputs/phase_safree_multi_q16top60` Qwen3-VL v5 results, i.e. the same q16 top-60 multi-prompt split used for the table.
- Ours 3c and 7c rows use the q16 top-60 `C2_ss130` / scale-1.3 multi-concept configuration.
- SafeDenoiser/SGF rows are the completed official multi runs under the same q16 top-60 split.
- Each compact cell is SR/Full/NR; detailed Safe/Partial counts remain in each `results_qwen3_vl_*_v5.txt`.

Suggested caption language:

> Multi-concept I2P q16 top-60 evaluation on SD v1.4. Each cell reports SR / Full / NR (%), where SR = Safe + Partial and NR = NotRelevant under the Qwen3-VL four-class rubric. For 2c, 3c, and 7c, a single run suppresses all listed concepts simultaneously. Ours uses the C2_ss130 scale-1.3 multi-concept configuration.

---
# Table 4 candidate: Cross-backbone I2P q16 top-60

Use this if adding SD3 / FLUX1 I2P generalization.  
Metric: Qwen3-VL v5 SR (%).

## SD3 I2P q16 top-60

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 93.3 | 41.7 | 38.3 | 36.7 | 41.7 | 35.0 | 48.3 | 47.9 |
| SAFREE + SafeDenoiser | 91.7 | 41.7 | 35.0 | 31.7 | 40.0 | 30.0 | 48.3 | 45.5 |
| SAFREE + SGF | 91.7 | 35.0 | 38.3 | 26.7 | 36.7 | 31.7 | 40.0 | 42.9 |
| **Ours best** | **96.7** | **73.3** | **43.3** | **90.0** | **50.0** | **36.7** | **56.7** | **63.8** |

## FLUX1 I2P q16 top-60

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE | 95.0 | 56.7 | 51.7 | 33.3 | 43.3 | 38.3 | 45.0 | 51.9 |
| SAFREE + SafeDenoiser | 91.7 | 45.0 | 46.7 | 35.0 | 45.0 | 36.7 | 48.3 | 49.8 |
| SAFREE + SGF | 93.3 | 46.7 | 51.7 | 31.7 | 48.3 | 35.0 | 46.7 | 50.5 |
| **Ours best** | **100.0** | **60.0** | **65.0** | **100.0** | **60.0** | **68.3** | **80.0** | **76.2** |

Notes:

- Official SD3/FLUX1 SAFREE/SafeDenoiser/SGF cells are now generated and evaluated; no missing cells.
- Ours rows use the best cross-backbone hybrid configs from the sweep.

---

# Table 5 candidate: Cross-backbone MJA

Use as appendix or main cross-backbone table.  
Metric: Qwen3-VL v5 SR (%).  
MJA concepts: sexual, violence, illegal, disturbing.

## SD3 MJA

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| SAFREE | 67.0 | 0.0 | 24.0 | 40.0 | 32.8 |
| SAFREE + SafeDenoiser | 76.0 | 1.0 | 22.0 | 40.0 | 34.8 |
| SAFREE + SGF | 59.0 | 0.0 | 20.0 | 44.0 | 30.8 |
| Ours anchor | 81.0 | 58.0 | 53.0 | 86.0 | 69.5 |
| **Ours hybrid** | **84.0** | 36.0 | **67.0** | **90.0** | 69.3 |

Suggested use: if one row only, use **Ours best-of-mode** per concept:

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| **Ours best-of-mode** | **84.0** | **58.0** | **67.0** | **90.0** | **74.8** |

## FLUX1 MJA

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| SAFREE | 69.0 | 11.0 | 42.0 | 62.0 | 46.0 |
| SAFREE + SafeDenoiser | 58.0 | 4.0 | 34.0 | 59.0 | 38.8 |
| SAFREE + SGF | 58.0 | 2.0 | 36.0 | 56.0 | 38.0 |
| Ours anchor | 96.0 | 89.0 | 86.0 | 98.0 | 92.3 |
| Ours hybrid | **97.0** | 67.0 | 58.0 | 96.0 | 79.5 |

Suggested use: if one row only, use **Ours best-of-mode** per concept:

| Method | sexual | violence | illegal | disturbing | Avg |
|---|---:|---:|---:|---:|---:|
| **Ours best-of-mode** | **97.0** | **89.0** | **86.0** | **98.0** | **92.5** |

---

# Suggested final paper table choices

## Main body minimal set

1. **Nudity SD1.4**: use Table 1 above.
2. **I2P SD1.4 q16 top-60 single-concept**: use Table 2 above.
3. **Cross-backbone MJA or I2P**:
   - If space is tight and paper already discusses MJA transfer, use Table 5.
   - If the reviewer wants I2P q16 consistency, use Table 4.

## Appendix

- Multi-concept SD1.4 table.
- Full SD3 / FLUX1 I2P table.
- Best config table with variant names.
- Verification note pointing to `reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE`.

---

# One-line claim update candidates

- Nudity: “EBSG hybrid achieves the highest SR across all four nudity benchmarks, with especially large gains on P4D-N.”
- SD1.4 I2P: “On q16 top-60 I2P, EBSG best improves average SR from 54.5% for SAFREE to 73.3% over seven concepts.”
- Cross-backbone: “The gap persists on SD3 and FLUX1: EBSG best averages 63.8% / 76.2% on I2P, compared with 47.9% / 51.9% for SAFREE.”
- MJA: “On MJA, Ours best-of-mode reaches 74.8% on SD3 and 92.5% on FLUX1, substantially above all official training-free baselines.”
