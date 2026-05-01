# MLWriter / OMC final table values for paper update

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
| **Ours best / EBSG hybrid** | **98.3** | **81.7** | **51.7** | **93.3** | **46.7** | **68.3** | **73.3** | **73.3** |

If excluding sexual to match the older six-concept I2P setup:

| Method | Avg over violence/self-harm/shocking/illegal/harassment/hate |
|---|---:|
| Baseline | 29.5 |
| SAFREE | 49.7 |
| SAFREE + SafeDenoiser | 32.8 |
| SAFREE + SGF | 28.3 |
| **Ours best / EBSG hybrid** | **69.2** |

Ours best configs:

| Concept | Best variant | SR |
|---|---|---:|
| sexual | `hybrid_best_tau05_cas0.5` | 98.3 |
| violence | `hybrid_best_img075_img0.225` | 81.7 |
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
Metric: Qwen3-VL v5 SR (%).  
All 3-concept rows below now use the same concept set: **sexual + violence + shocking**.

## 2-concept: sexual + violence

| Method | sexual | violence | Avg |
|---|---:|---:|---:|
| SAFREE multi | 86.7 | 55.0 | 70.9 |
| SAFREE + SafeDenoiser multi | 88.3 | 66.7 | 77.5 |
| SAFREE + SGF multi | 86.7 | 58.3 | 72.5 |
| **Ours multi** | **88.3** | **76.7** | **82.5** |

## 3-concept: sexual + violence + shocking

| Method | sexual | violence | shocking | Avg |
|---|---:|---:|---:|---:|
| SAFREE multi | 85.0 | 55.0 | 36.7 | 58.9 |
| SAFREE + SafeDenoiser multi | 85.0 | 60.0 | 61.7 | 68.9 |
| SAFREE + SGF multi | 90.0 | 53.3 | 40.0 | 61.1 |
| **Ours multi** | **90.0** | **76.7** | **78.3** | **81.7** |

## 7-concept multi

| Method | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFREE multi | 90.0 | 40.0 | 33.3 | 30.0 | 35.0 | 26.7 | 38.3 | 41.9 |
| SAFREE + SafeDenoiser multi | 90.0 | 51.7 | 41.7 | 43.3 | 41.7 | 38.3 | 48.3 | 50.7 |
| SAFREE + SGF multi | 86.7 | 50.0 | 40.0 | 43.3 | 38.3 | 36.7 | 43.3 | 48.3 |
| **Ours multi (C2_ss130)** | **88.3** | **85.0** | **66.7** | **88.3** | **65.0** | **60.0** | **58.3** | **73.1** |

Suggested caption language:

> Multi-concept I2P q16 top-60 evaluation. For 2c, 3c, and 7c, a single model/run suppresses all listed concepts simultaneously. The 3c setting uses sexual + violence + shocking for every method. For 7c, the average is over all seven concepts including sexual. Ours 7c reports the q16 top-60 C2_ss130 / scale-1.3 configuration.

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
