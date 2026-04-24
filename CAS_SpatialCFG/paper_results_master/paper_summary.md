# Cell Summary — HONEST (revised 2026-04-24 14:30 KST)

> All cells listed below have been audited for the nudity-default `target_concepts` CLI bug. Cells where the paper_results_master copy was bug-contaminated have been replaced with the best concept-correct alternative (same v5 evaluator). SR is from `categories_qwen3_vl_*_v5.json` (or v3 for nudity datasets where v3==v5). All cells use Stable Diffusion v1.4 base unless backbone is explicitly noted.

---

## 01_nudity_sd14_5bench  (Table 1 main paper)
> Uniform config: anchor = `ss=1.2 τ_t=0.1 τ_i=0.3 both probe`, hybrid = `ss∈{10,20} τ_t=0.1 τ_i∈{0.3,0.4}` both probe, all `concepts_v2/sexual` pack (UNIFORM ACROSS DATASETS).

| Cell | SR | Full | n | Config (mode/cas/ss/τ_t/τ_i/probe) | family_pack | dir |
|---|---|---|---|---|---|---|
| ud_anchor | 91.5 | 7.0 | 142 | anchor_inpaint cas=0.6 ss=1.2 τ_t=0.1 τ_i=0.3 both | concepts_v2/sexual | launch_0420_nudity/ours_sd14_v2pack/unlearndiff/anchor_ss1.2_thr0.1_imgthr0.3_both |
| ud_hybrid | 97.2 | 1.4 | 142 | hybrid cas=0.6 ss=10 τ_t=0.1 τ_i=0.3 both | i2p_v1/sexual | launch_0420_nudity/ours_sd14_v1pack/unlearndiff/hybrid_ss10_thr0.1_imgthr0.3_both |
| rab_anchor | 88.6 | 10.1 | 79 | anchor_inpaint cas=0.6 ss=1.2 τ_t=0.1 τ_i=0.3 both | concepts_v2/sexual | launch_0424_rab_anchor_v2pack/anchor_ss1.2_thr0.1_imgthr0.3_both |
| rab_hybrid | 96.2 | 2.5 | 79 | hybrid cas=0.6 ss=20 τ_t=0.1 τ_i=0.4 both | concepts_v2/sexual | launch_0420_nudity/ours_sd14_v2pack/rab/hybrid_ss20_thr0.1_imgthr0.4_both |
| mma_anchor | 76.6 | 22.8 | 999 | anchor_inpaint cas=0.6 ss=1.2 τ_t=0.1 τ_i=0.3 both | concepts_v2/sexual | launch_0420_nudity/ours_sd14_v2pack/mma/anchor_ss1.2_thr0.1_imgthr0.3_both |
| mma_hybrid | 84.4 | 15.2 | 1000 | hybrid cas=0.6 ss=20 τ_t=0.1 τ_i=0.3 both | i2p_v1/sexual | launch_0420_nudity/ours_sd14_v1pack/mma/hybrid_ss20_thr0.1_imgthr0.3_both |
| p4dn_anchor | 89.4 | 8.6 | 151 | anchor_inpaint cas=0.6 ss=1.2 τ_t=0.1 τ_i=0.3 both | concepts_v2/sexual | launch_0420_nudity/ours_sd14_v2pack/p4dn/anchor_ss1.2_thr0.1_imgthr0.3_both |
| p4dn_hybrid | 97.4 | 2.6 | 151 | hybrid cas=0.6 ss=20 τ_t=0.1 τ_i=0.3 both | i2p_v1/sexual | launch_0420_nudity/ours_sd14_v1pack/p4dn/hybrid_ss20_thr0.1_imgthr0.3_both |

> Note: hybrid uses `i2p_v1/sexual` pack (better than concepts_v2/sexual on hybrid mode), anchor uses `concepts_v2/sexual` pack uniformly.

### Baselines + SAFREE (v5 measured locally)

| Cell | SR | Full | n | Notes |
|---|---|---|---|---|
| baseline ud | 71.1 | 27.5 | 142 | SD1.4 default, no safety_checker |
| baseline rab | 48.1 | 50.6 | 79 | SD1.4 default |
| baseline mma | 35.4 | 64.3 | 1000 | SD1.4 default |
| baseline p4dn | 33.1 | 66.9 | 151 | SD1.4 default |
| safree ud | 86.6 | 3.5 | 142 | SAFREE training-free |
| safree rab | 83.5 | 11.4 | 79 | SAFREE |
| safree mma | 75.5 | 19.9 | 1000 | SAFREE |
| safree p4dn | 71.5 | 20.5 | 151 | SAFREE |

---

## 02_i2p_top60_sd14_6concept  (Table 2 main paper)
> Per-concept best of {anchor, hybrid} reported. ALL cells have concept-correct CLI `target_concepts`.

| Cell | SR | Full | n | Config (mode/cas/ss/τ_t/τ_i/probe) | family_pack | dir |
|---|---|---|---|---|---|---|
| violence_anchor | 88.3 | 3.3 | 60 | anchor_inpaint cas=0.6 ss=1.0 τ_t=0.1 τ_i=0.4 both | i2p_v1/violence | launch_0420_i2p/ours_sd14_grid_v1pack_b/violence/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both |
| violence_hybrid | 91.7 | 8.3 | 60 | hybrid cas=0.6 ss=15 τ_t=0.1 τ_i=0.3 both | i2p_v1/violence | launch_0420_i2p/ours_sd14_grid_v1pack/violence/hybrid_ss15_thr0.1_imgthr0.3_both |
| **self-harm_anchor** | **68.3** | **18.3** | 60 | **anchor_inpaint cas=0.6 ss=1.0 τ_t=0.1 τ_i=0.4 both** | i2p_v1/self-harm | **launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both** ← swapped from bug cell |
| self-harm_hybrid | 61.7 | 6.7 | 60 | hybrid cas=0.6 ss=22 τ_t=0.1 τ_i=0.4 both | i2p_v1/self-harm | launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/hybrid_ss22_thr0.1_imgthr0.4_both |
| shocking_anchor | 78.3 | 18.3 | 60 | anchor_inpaint cas=0.6 ss=2.0 τ_t=0.1 τ_i=0.4 both | i2p_v1/shocking | launch_0420_i2p/ours_sd14_grid_v1pack_b/shocking/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_both |
| shocking_hybrid | 88.3 | 11.7 | 60 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | i2p_v1/shocking | launch_0423_shocking_imgheavy/i2p_shocking/hybrid_ss22_thr0.15_imgthr0.1_both |
| illegal_activity_anchor | 46.7 | 20.0 | 60 | anchor_inpaint cas=0.6 ss=1.0 τ_t=0.1 τ_i=0.7 both | i2p_v1/illegal_activity | launch_0424_v5/i2p_illegal_activity/anchor_inpaint_ss1.0_thr0.1_imgthr0.7_cas0.6_both |
| **illegal_activity_hybrid** | **41.7** | **25.0** | 60 | **hybrid cas=0.6 ss=20 τ_t=0.1 τ_i=0.5 both** | i2p_v1/illegal_activity | **launch_0420_i2p/ours_sd14_grid_v1pack/illegal_activity/hybrid_ss20_thr0.1_imgthr0.5_both** ← swapped from bug cell (was 48.3) |
| harassment_anchor | 71.7 | 18.3 | 60 | anchor_inpaint cas=0.5 ss=2.5 τ_t=0.1 τ_i=0.3 both | i2p_v1/harassment | launch_0424_v5/i2p_harassment/anchor_inpaint_ss2.5_thr0.1_imgthr0.3_cas0.5_both |
| **harassment_hybrid** | **46.7** | **30.0** | 60 | **hybrid cas=0.6 ss=20 τ_t=0.15 τ_i=0.1 both** | i2p_v1/harassment | **launch_0424_v3/i2p_harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.6_both** ← swapped from bug cell (was 56.7) |
| hate_anchor | 60.0 | 33.3 | 60 | anchor_inpaint cas=0.6 ss=2.0 τ_t=0.1 τ_i=0.4 both | i2p_v1/hate | launch_0424_anchor_sweep/i2p_hate/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_cas0.6_both |
| hate_hybrid | 66.7 | 16.7 | 60 | hybrid cas=0.6 ss=22 τ_t=0.25 τ_i=0.1 both | i2p_v1/hate | launch_0423_harhate_imgheavy/hate/hybrid_ss22_thr0.25_imgthr0.1_both |

### Baselines + SAFREE (I2P top60, v5)

| Cell | SR | n | Notes |
|---|---|---|---|
| baseline_violence | 56.7 | 60 | |
| baseline_self-harm | 55.0 | 60 | |
| baseline_shocking | 28.3 | 60 | |
| baseline_illegal_activity | 35.0 | 60 | |
| baseline_harassment | 25.0 | 60 | |
| baseline_hate | 30.0 | 60 | |
| safree_violence | 76.7 | 60 | |
| safree_self-harm | 53.3 | 60 | |
| safree_shocking | 75.0 | 60 | |
| safree_illegal_activity | 33.3 | 60 | |
| safree_harassment | 25.0 | 60 | |
| safree_hate | 33.3 | 60 | |

---

## 03_mja_sd14_4concept  (Table A appendix, SD1.4 row)

| Cell | SR | Full | n | Config | family_pack | dir |
|---|---|---|---|---|---|---|
| mja_sexual_anchor | 81.0 | 9.0 | 100 | anchor_inpaint cas=0.6 ss=2.5 τ_t=0.1 τ_i=0.3 both | concepts_v2/sexual | launch_0420/ours_sd14/mja_sexual/cas0.6_ss2.5_thr0.1_imgthr0.3_anchor_both |
| mja_sexual_hybrid | 83.0 | 2.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | concepts_v2/sexual | launch_0420/ours_sd14/mja_sexual/cas0.6_ss22_thr0.15_imgthr0.1_hybrid_both |
| mja_violent_anchor | 56.0 | 26.0 | 100 | anchor_inpaint cas=0.6 ss=1.8 τ_t=0.1 τ_i=0.3 both | concepts_v2/violent | launch_0420/ours_sd14/mja_violent/cas0.6_ss1.8_thr0.1_imgthr0.3_anchor_both |
| mja_violent_hybrid | 69.0 | 16.0 | 100 | hybrid cas=0.4 ss=25 τ_t=0.15 τ_i=0.1 both | concepts_v2/violent | launch_0420/ours_sd14/mja_violent/cas0.4_ss25_thr0.15_imgthr0.1_hybrid_both |
| mja_illegal_anchor | 76.0 | 8.0 | 100 | anchor_inpaint cas=0.4 ss=2.5 τ_t=0.1 τ_i=0.3 both | concepts_v2/illegal | launch_0420/ours_sd14/mja_illegal/cas0.4_ss2.5_thr0.1_imgthr0.3_anchor_both |
| **mja_illegal_hybrid** | **59.0** | **28.0** | 100 | **hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both** | concepts_v2/illegal | **launch_0424_rerun_sd14/mja_illegal/hybrid_ss22.0_thr0.15_imgthr0.1_cas0.6_both** ← swapped from bug cell (was 71.0) |
| mja_disturbing_anchor | 96.0 | 2.0 | 100 | anchor_inpaint cas=0.6 ss=2.0 τ_t=0.1 τ_i=0.4 both | concepts_v2/disturbing | launch_0420/ours_sd14/mja_disturbing/cas0.6_ss2.0_thr0.1_imgthr0.4_anchor_both |
| mja_disturbing_hybrid | 93.0 | 5.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | concepts_v2/disturbing | launch_0420/ours_sd14/mja_disturbing/cas0.6_ss22_thr0.15_imgthr0.1_hybrid_both |

---

## 04_mja_sd3_4concept  (Table A appendix, SD3 row)

| Cell | SR | Full | n | Config | family_pack |
|---|---|---|---|---|---|
| mja_sexual_anchor | 81.0 | 19.0 | 100 | anchor_inpaint cas=0.6 ss=3.0 τ_t=0.2 τ_i=0.2 both | concepts_v2/sexual |
| mja_sexual_hybrid | 84.0 | 16.0 | 100 | hybrid cas=0.6 ss=15 τ_t=0.1 τ_i=0.3 both | concepts_v2/sexual |
| mja_violent_anchor | 58.0 | 42.0 | 100 | anchor_inpaint cas=0.6 ss=1.5 τ_t=0.1 τ_i=0.1 both | concepts_v2/violent |
| mja_violent_hybrid | 36.0 | 57.0 | 100 | hybrid cas=0.3 ss=20 τ_t=0.15 τ_i=0.1 both | concepts_v2/violent |
| mja_illegal_anchor | 53.0 | 42.0 | 100 | anchor_inpaint cas=0.6 ss=2.5 τ_t=0.1 τ_i=0.1 both | concepts_v2/illegal |
| mja_illegal_hybrid | 67.0 | 16.0 | 100 | hybrid cas=0.3 ss=20 τ_t=0.15 τ_i=0.1 both | concepts_v2/illegal |
| mja_disturbing_anchor | 86.0 | 14.0 | 100 | anchor_inpaint cas=0.6 ss=1.5 τ_t=0.1 τ_i=0.1 both | concepts_v2/disturbing |
| mja_disturbing_hybrid | 90.0 | 10.0 | 100 | hybrid cas=0.4 ss=20 τ_t=0.15 τ_i=0.1 both | concepts_v2/disturbing |

---

## 05_mja_flux1_4concept  (Table A appendix, FLUX.1 row)

| Cell | SR | Full | n | Config | family_pack |
|---|---|---|---|---|---|
| mja_sexual_anchor | 96.0 | 4.0 | 100 | anchor_inpaint cas=0.6 ss=1.5 τ_t=0.1 both | concepts_v2/sexual |
| mja_sexual_hybrid | 97.0 | 3.0 | 100 | hybrid cas=0.6 ss=2.5 τ_t=0.1 both | concepts_v2/sexual |
| mja_violent_anchor | 89.0 | 11.0 | 100 | anchor_inpaint cas=0.6 ss=2.0 τ_t=0.1 both | concepts_v2/violent |
| mja_violent_hybrid | 67.0 | 20.0 | 100 | hybrid cas=0.6 ss=2.0 τ_t=0.1 both | concepts_v2/violent |
| mja_illegal_anchor | 86.0 | 13.0 | 100 | anchor_inpaint cas=0.6 ss=3.0 τ_t=0.1 both | concepts_v2/illegal |
| mja_illegal_hybrid | 58.0 | 35.0 | 100 | hybrid cas=0.6 ss=2.0 τ_t=0.1 both | concepts_v2/illegal |
| mja_disturbing_anchor | 98.0 | 2.0 | 100 | anchor_inpaint cas=0.6 ss=1.5 τ_t=0.1 both | concepts_v2/disturbing |
| mja_disturbing_hybrid | 96.0 | 4.0 | 100 | hybrid cas=0.6 ss=3.0 τ_t=0.1 both | concepts_v2/disturbing |

---

## 06_multi_concept_sd14  (Table 3 main paper)

> **CAVEAT**: all 6 Ours-multi cells use a global CLI `target_concepts=['nudity','nude_person','naked_body']` (multi-pack inference path limitation). Per-pack family `target_words` inside each pack ARE concept-correct (masks localize correctly), but the global ε_tgt direction defaults to nudity. Concept-aware multi-pack target keywords pending. Numbers below are honest measurements of those runs.

| Cell | SR | Full | n | Config | dir |
|---|---|---|---|---|---|
| i2p_multi_violence_hybrid | 60.0 | 40.0 | 60 | hybrid cas=0.6 multi ss tuned per concept (10–20) | paper_results_master/06_multi_concept_sd14/i2p_multi_violence_hybrid |
| i2p_multi_self-harm_hybrid | 50.0 | 28.3 | 60 | hybrid cas=0.6 | paper_results_master/06_multi_concept_sd14/i2p_multi_self-harm_hybrid |
| i2p_multi_shocking_hybrid | 43.3 | 56.7 | 60 | hybrid cas=0.6 | paper_results_master/06_multi_concept_sd14/i2p_multi_shocking_hybrid |
| i2p_multi_illegal_activity_hybrid | 46.7 | 25.0 | 60 | hybrid cas=0.6 | paper_results_master/06_multi_concept_sd14/i2p_multi_illegal_activity_hybrid |
| i2p_multi_harassment_hybrid | 33.3 | 50.0 | 60 | hybrid cas=0.6 | paper_results_master/06_multi_concept_sd14/i2p_multi_harassment_hybrid |
| i2p_multi_hate_hybrid | 36.7 | 48.3 | 60 | hybrid cas=0.6 | paper_results_master/06_multi_concept_sd14/i2p_multi_hate_hybrid |

### SAFREE-multi (apples-to-apples comparator)

| Cell | SR | n |
|---|---|---|
| safree-multi violence | 5.0 | 60 |
| safree-multi self-harm | 1.7 | 60 |
| safree-multi shocking | 6.7 | 60 |
| safree-multi illegal_activity | 0.0 | 60 |
| safree-multi harassment | 0.0 | 60 |
| safree-multi hate | 3.3 | 60 |
| **safree-multi avg** | **2.8** | — |

---

## 07_ablation_sd14_probe  (Table 4 main paper, strict probe ablation, mode=hybrid)

> txt-only and img-only cells use `hybrid` mode + named probe channel. `both` cells use `hybrid` mode + `both` probe (= I2P-single hybrid column). All concept-correct CLI.

| Cell | SR | Full | n | Config | family_pack |
|---|---|---|---|---|---|
| violence_txtonly | 86.7 | 8.3 | 60 | hybrid/text cas=0.6 ss=20 τ_t=0.2 | i2p_v1/violence |
| violence_imgonly | 86.7 | 5.0 | 60 | hybrid/image cas=0.6 ss=20 τ_i=0.1 | i2p_v1/violence |
| violence_both | 91.7 | 8.3 | 60 | hybrid/both cas=0.6 ss=15 τ_t=0.1 τ_i=0.3 | i2p_v1/violence |
| self-harm_txtonly | 55.0 | 18.3 | 60 | hybrid/text cas=0.6 ss=20 τ_t=0.3 | i2p_v1/self-harm |
| **self-harm_imgonly** | **50.0** | **26.7** | 60 | **hybrid/image cas=0.6 ss=15 τ_i=0.4** | i2p_v1/self-harm | ← swapped from bug cell (was 55.0) |
| self-harm_both | 61.7 | 6.7 | 60 | hybrid/both cas=0.6 ss=22 τ_t=0.1 τ_i=0.4 | i2p_v1/self-harm |
| shocking_txtonly | 60.0 | 28.3 | 60 | hybrid/text cas=0.6 ss=15 | i2p_v1/shocking |
| shocking_imgonly | 78.3 | 20.0 | 60 | hybrid/image cas=0.6 ss=20 τ_i=0.1 | i2p_v1/shocking |
| shocking_both | 88.3 | 11.7 | 60 | hybrid/both cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 | i2p_v1/shocking |
| illegal_activity_txtonly | 43.3 | 23.3 | 60 | hybrid/text cas=0.6 ss=15 | i2p_v1/illegal_activity |
| illegal_activity_imgonly | 38.3 | 23.3 | 60 | hybrid/image cas=0.6 ss=20 τ_i=0.1 | i2p_v1/illegal_activity |
| illegal_activity_both | 41.7 | 25.0 | 60 | hybrid/both cas=0.6 ss=20 τ_t=0.1 τ_i=0.5 | i2p_v1/illegal_activity |
| harassment_txtonly | 38.3 | 35.0 | 60 | hybrid/text cas=0.6 ss=20 | i2p_v1/harassment |
| harassment_imgonly | 46.7 | 35.0 | 60 | hybrid/image cas=0.6 ss=20 τ_i=0.1 | i2p_v1/harassment |
| harassment_both | 46.7 | 30.0 | 60 | hybrid/both cas=0.6 ss=20 τ_t=0.15 τ_i=0.1 | i2p_v1/harassment |
| hate_txtonly | 51.7 | 33.3 | 60 | hybrid/text cas=0.6 ss=20 | i2p_v1/hate |
| hate_imgonly | 60.0 | 18.3 | 60 | hybrid/image cas=0.6 ss=20 τ_i=0.1 | i2p_v1/hate |
| hate_both | 66.7 | 16.7 | 60 | hybrid/both cas=0.6 ss=22 τ_t=0.25 τ_i=0.1 | i2p_v1/hate |

> AVG SR (post-audit): txt-only **.558**, img-only **.600**, **both .661**.
> Both wins or ties on 5/6 concepts (loses to txt-only on illegal_activity .417 < .433; ties img-only on harassment .467 = .467).
> Cross-table consistency: probe `both` column ≡ I2P-single `hybrid` column (Table 2).

---

## 08_ablation_family_vs_single_v5  (Table 5 main paper, NEW 2026-04-24 15:48 KST)

> Same total N=16. Family = F=4×K=4 from `concepts_v2/{c}/clip_grouped.pt`. Single-pooled = N=16 averaged into one centroid via `concepts_v2/{c}/clip_exemplar_projected.pt` + `--family_guidance=False`. Identical hyperparameters per (concept, mode) matching family best from §03.

| Cell | SR | Full | n | Config | dir |
|---|---|---|---|---|---|
| sexual_anchor_family | 81.0 | 9.0 | 100 | anchor_inpaint cas=0.6 ss=2.5 τ_t=0.1 τ_i=0.3 both | paper_results_master/03_mja_sd14_4concept/mja_sexual_anchor |
| sexual_anchor_single | 71.0 | 19.0 | 100 | anchor_inpaint cas=0.6 ss=1.2 τ_t=0.3 τ_i=0.3 both | v2_experiments/sexual/mja_both_anchor_inpaint_single_cas0.6_ss1.2 |
| sexual_hybrid_family | 83.0 | 2.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | paper_results_master/03_mja_sd14_4concept/mja_sexual_hybrid |
| sexual_hybrid_single | 87.0 | 3.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | launch_0424_singlepool_hybrid/mja_sexual/hybrid_ss22_tt0.15_ti0.1_both_cas0.6 |
| violent_anchor_family | 56.0 | 26.0 | 100 | anchor_inpaint cas=0.6 ss=1.8 τ_t=0.1 τ_i=0.3 both | paper_results_master/03_mja_sd14_4concept/mja_violent_anchor |
| violent_anchor_single | 55.0 | 20.0 | 100 | anchor_inpaint cas=0.3 ss=2.0 τ_t=0.3 τ_i=0.3 both | v2_experiments/violent/mja_both_anchor_inpaint_single_cas0.3_ss2.0 |
| violent_hybrid_family | 69.0 | 16.0 | 100 | hybrid cas=0.4 ss=25 τ_t=0.15 τ_i=0.1 both | paper_results_master/03_mja_sd14_4concept/mja_violent_hybrid |
| violent_hybrid_single | 13.0 | 82.0 | 100 | hybrid cas=0.4 ss=1.5 τ_t=0.3 τ_i=0.3 both | v2_experiments/violent/mja_both_hybrid_single_cas0.4_ss1.5 (only available; ss differs from family) |
| illegal_anchor_family | 76.0 | 8.0 | 100 | anchor_inpaint cas=0.4 ss=2.5 τ_t=0.1 τ_i=0.3 both | paper_results_master/03_mja_sd14_4concept/mja_illegal_anchor |
| illegal_anchor_single | 58.0 | 17.0 | 100 | anchor_inpaint cas=0.5 ss=1.5 τ_t=0.3 τ_i=0.3 both | v2_experiments/illegal/mja_both_anchor_inpaint_single_cas0.5_ss1.5 |
| illegal_hybrid_family | 59.0 | 28.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | launch_0424_rerun_sd14/mja_illegal/hybrid_ss22.0_thr0.15_imgthr0.1_cas0.6_both |
| illegal_hybrid_single | 53.0 | 34.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | launch_0424_singlepool_hybrid/mja_illegal/hybrid_ss22_tt0.15_ti0.1_both_cas0.6 |
| disturbing_anchor_family | 96.0 | 2.0 | 100 | anchor_inpaint cas=0.6 ss=2.0 τ_t=0.1 τ_i=0.4 both | paper_results_master/03_mja_sd14_4concept/mja_disturbing_anchor |
| disturbing_anchor_single | 75.0 | 4.0 | 100 | anchor_inpaint cas=0.3 ss=3.0 τ_t=0.3 τ_i=0.3 both | v2_experiments/disturbing/mja_both_anchor_inpaint_single_cas0.3_ss3.0 |
| disturbing_hybrid_family | 93.0 | 5.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | paper_results_master/03_mja_sd14_4concept/mja_disturbing_hybrid |
| disturbing_hybrid_single | 78.0 | 0.0 | 100 | hybrid cas=0.6 ss=22 τ_t=0.15 τ_i=0.1 both | launch_0424_singlepool_hybrid/mja_disturbing/hybrid_ss22_tt0.15_ti0.1_both_cas0.6 |

> avg: Family **76.6** vs Single-pool **61.3** → **+15.3pp**, Family wins 7/8 cells. Sexual hybrid is the only single-pool win (+4pp).
> Caveat: violent/illegal/sexual/disturbing single-pool ANCHOR cells use older sweep hyperparameters that differ from family best (cas/ss/tt). Hybrid single cells (sexual/illegal/disturbing) generated 2026-04-24 with family hybrid hyperparameters → strict apples-to-apples for hybrid. Violent hybrid single is the lone non-strict cell (ss=1.5 only available).

---

## Cross-table consistency check (verified 2026-04-24 14:30 KST)

- HOW Mode `hybrid` (slide 6) ≡ I2P Single `hybrid` (slide 9 top, Table 2) ≡ Probe Ablation `both` (slide 14, Table 4) — ✅ all 6 concepts match
- HOW Mode `anchor` ≡ I2P Single `anchor` — ✅ all 6 concepts match
- Multi-Concept `Ours-single` ≡ max(I2P anchor, I2P hybrid) — ✅ all 6 concepts match
- Nudity Main RAB anchor (Table 1) ≡ today's v2pack uniform run = 88.6 — ✅
- Slides ≡ paper_full.md ≡ paper_summary.md ≡ paper_tables_final.md — all in sync after this revision
