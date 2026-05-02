# Paper Results Master

Best (max SR) generation dirs per (section, concept, mode), copied here for paper.

## 01_nudity_sd14_5bench

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| rab_anchor | — | — | — | NOT FOUND |
| rab_hybrid | 97.4 | 2.6 | 78 | `launch_0420_nudity/ours_sd14_v2pack/rab/hybrid_ss20_thr0.1_imgthr0.4_both` |
| mma_anchor | 77.0 | 23.0 | 993 | `launch_0420_nudity/ours_sd14_v2pack/mma/anchor_ss1.2_thr0.1_imgthr0.3_both` |
| mma_hybrid | 84.7 | 15.3 | 996 | `launch_0420_nudity/ours_sd14_v1pack/mma/hybrid_ss20_thr0.1_imgthr0.3_both` |
| p4dn_anchor | 91.2 | 8.8 | 148 | `launch_0420_nudity/ours_sd14_v2pack/p4dn/anchor_ss1.2_thr0.1_imgthr0.3_both` |
| p4dn_hybrid | 97.4 | 2.6 | 151 | `launch_0420_nudity/ours_sd14_v1pack/p4dn/hybrid_ss20_thr0.1_imgthr0.3_both` |
| unlearndiff_anchor | 92.9 | 7.1 | 140 | `launch_0420_nudity/ours_sd14_v2pack/unlearndiff/anchor_ss1.2_thr0.1_imgthr0.3_both` |
| unlearndiff_hybrid | 98.6 | 1.4 | 140 | `launch_0420_nudity/ours_sd14_v1pack/unlearndiff/hybrid_ss10_thr0.1_imgthr0.3_both` |
| mja_sexual_anchor | 90.0 | 10.0 | 90 | `launch_0420/ours_sd14/mja_sexual/anchor_ss2.5_thr0.1_imgthr0.3_both` |
| mja_sexual_hybrid | 96.7 | 3.3 | 90 | `launch_0420/ours_sd14/mja_sexual/hybrid_ss15_thr0.1_imgthr0.5_both` |

## 02_i2p_top60_sd14_6concept

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| violence_anchor | 88.3 | 3.3 | 60 | `launch_0420_i2p/ours_sd14_grid_v1pack_b/violence/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both` |
| violence_hybrid | 91.7 | 8.3 | 60 | `launch_0420_i2p/ours_sd14_grid_v1pack/violence/hybrid_ss15_thr0.1_imgthr0.3_both` |
| self-harm_anchor | 68.3 | 25.0 | 60 | `launch_0420_i2p/ours_sd14/self-harm/cas0.6_ss1.5_thr0.1_imgthr0.3_anchor_imgonly` |
| self-harm_hybrid | 61.7 | 6.7 | 60 | `launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/hybrid_ss22_thr0.1_imgthr0.4_both` |
| shocking_anchor | 78.3 | 18.3 | 60 | `launch_0424_rerun_sd14/i2p_shocking/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_cas0.6_both` |
| shocking_hybrid | 88.3 | 11.7 | 60 | `launch_0423_shocking_imgheavy/hybrid_ss22_thr0.15_imgthr0.1_both` |
| illegal_activity_anchor | 46.7 | 20.0 | 60 | `launch_0424_v5/i2p_illegal_activity/anchor_inpaint_ss1.0_thr0.1_imgthr0.7_cas0.6_both` |
| illegal_activity_hybrid | 48.3 | 20.0 | 60 | `launch_0420_i2p/ours_sd14_v1pack_repatched/illegal_activity/hybrid_ss20_thr0.1_imgthr0.5_both` |
| harassment_anchor | 71.7 | 18.3 | 60 | `launch_0424_v5/i2p_harassment/anchor_inpaint_ss2.5_thr0.1_imgthr0.3_cas0.5_both` |
| harassment_hybrid | 56.7 | 26.7 | 60 | `launch_0423_retune/i2p/harassment/hybrid_cas0.5_ss25_thr0.1_imgthr0.5_both` |
| hate_anchor | 60.0 | 33.3 | 60 | `launch_0424_rerun_sd14/i2p_hate/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_cas0.6_both` |
| hate_hybrid | 66.7 | 16.7 | 60 | `launch_0423_harhate_imgheavy/hate/hybrid_ss22_thr0.25_imgthr0.1_both` |

## 03_mja_sd14_4concept

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| mja_sexual_anchor | 90.0 | 10.0 | 90 | `launch_0420/ours_sd14/mja_sexual/anchor_ss2.5_thr0.1_imgthr0.3_both` |
| mja_sexual_hybrid | 97.6 | 2.4 | 85 | `launch_0424_rerun_sd14/mja_sexual/hybrid_ss22.0_thr0.15_imgthr0.1_cas0.6_both` |
| mja_violent_anchor | 56.0 | 26.0 | 100 | `launch_0420/ours_sd14/mja_violent/anchor_ss1.8_thr0.1_imgthr0.3_both` |
| mja_violent_hybrid | 69.0 | 16.0 | 100 | `launch_0424_v6/mja_violent/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.4_both` |
| mja_illegal_anchor | 76.0 | 8.0 | 100 | `launch_0424_v5/mja_illegal/anchor_inpaint_ss2.5_thr0.1_imgthr0.3_cas0.4_both` |
| mja_illegal_hybrid | 71.0 | 16.0 | 100 | `launch_0423_illegal_aggro/hybrid_cas0.1_ss30_thr0.1_imgthr0.6_both` |
| mja_disturbing_anchor | 96.0 | 2.0 | 100 | `launch_0424_rerun_sd14/mja_disturbing/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_cas0.6_both` |
| mja_disturbing_hybrid | 93.0 | 5.0 | 100 | `launch_0424_rerun_sd14/mja_disturbing/hybrid_ss22.0_thr0.15_imgthr0.1_cas0.6_both` |

## 04_mja_sd3_4concept

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| mja_sexual_anchor | 81.0 | 19.0 | 100 | `launch_0420/ours_sd3/mja_sexual/cas0.6_ss3.0_thr0.2_anchor_inpaint_both` |
| mja_sexual_hybrid | 84.0 | 16.0 | 100 | `launch_0420/ours_sd3/mja_sexual/hybrid_ss15_thr0.1_imgthr0.3_both` |
| mja_violent_anchor | 58.0 | 42.0 | 100 | `launch_0420/ours_sd3/mja_violent/cas0.6_ss1.5_thr0.1_anchor_inpaint_both` |
| mja_violent_hybrid | 36.0 | 57.0 | 100 | `launch_0424_v4_sd3/mja_violent/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.3_both` |
| mja_illegal_anchor | 53.0 | 42.0 | 100 | `launch_0420/ours_sd3/mja_illegal/cas0.6_ss2.5_thr0.1_anchor_inpaint_both` |
| mja_illegal_hybrid | 67.0 | 16.0 | 100 | `launch_0424_v4_sd3/mja_illegal/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.3_both` |
| mja_disturbing_anchor | 86.0 | 14.0 | 100 | `launch_0420/ours_sd3/mja_disturbing/cas0.6_ss1.5_thr0.1_anchor_inpaint_both` |
| mja_disturbing_hybrid | 90.0 | 10.0 | 100 | `launch_0424_v4_sd3/mja_disturbing/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both` |

## 05_mja_flux1_4concept

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| mja_sexual_anchor | 96.0 | 4.0 | 100 | `launch_0420/ours_flux1/mja_sexual/cas0.6_ss1.5_thr0.1_anchor_inpaint_both` |
| mja_sexual_hybrid | 97.0 | 3.0 | 100 | `launch_0420/ours_flux1/mja_sexual/cas0.6_ss2.5_thr0.1_hybrid_both` |
| mja_violent_anchor | 89.0 | 11.0 | 100 | `launch_0420/ours_flux1/mja_violent/cas0.6_ss2.0_thr0.1_anchor_inpaint_both` |
| mja_violent_hybrid | 67.0 | 20.0 | 100 | `launch_0420/ours_flux1/mja_violent/cas0.6_ss2.0_thr0.1_hybrid_both` |
| mja_illegal_anchor | 86.0 | 13.0 | 100 | `launch_0420/ours_flux1/mja_illegal/cas0.6_ss3.0_thr0.1_anchor_inpaint_both` |
| mja_illegal_hybrid | 58.0 | 35.0 | 100 | `launch_0420/ours_flux1/mja_illegal/cas0.6_ss2.0_thr0.1_hybrid_both` |
| mja_disturbing_anchor | 98.0 | 2.0 | 100 | `launch_0420/ours_flux1/mja_disturbing/cas0.6_ss1.5_thr0.1_anchor_inpaint_both` |
| mja_disturbing_hybrid | 96.0 | 4.0 | 100 | `launch_0420/ours_flux1/mja_disturbing/cas0.6_ss3.0_thr0.1_hybrid_both` |

## 06_multi_concept_sd14

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| mja_multi_mja_sexual_anchor | 73.0 | 27.0 | 89 | `launch_0420/ours_sd14_multiconcept/mja_sexual/cas0.6_ss3.0_thr0.1_anchor_inpaint_both` |
| mja_multi_mja_sexual_hybrid | 43.8 | 56.2 | 89 | `launch_0420/ours_sd14_multiconcept/mja_sexual/cas0.6_ss0.5_thr0.2_hybrid_both` |
| mja_multi_mja_violent_anchor | 71.0 | 23.0 | 100 | `launch_0420/ours_sd14_multiconcept/mja_violent/cas0.6_ss2.5_thr0.2_anchor_inpaint_both` |
| mja_multi_mja_violent_hybrid | 20.0 | 76.0 | 100 | `launch_0420/ours_sd14_multiconcept/mja_violent/cas0.6_ss1.0_thr0.2_hybrid_both` |
| i2p_multi_violence_anchor | — | — | — | NOT FOUND |
| i2p_multi_violence_hybrid | 60.0 | 40.0 | 60 | `launch_0420_i2p/ours_sd14_multi/violence/hybrid_ss20_thr0.1_imgthr0.3_both` |
| i2p_multi_self-harm_anchor | — | — | — | NOT FOUND |
| i2p_multi_self-harm_hybrid | 50.0 | 28.3 | 60 | `launch_0420_i2p/ours_sd14_multi/self-harm/hybrid_ss10_thr0.1_imgthr0.3_both` |
| i2p_multi_shocking_anchor | — | — | — | NOT FOUND |
| i2p_multi_shocking_hybrid | 43.3 | 56.7 | 60 | `launch_0420_i2p/ours_sd14_multi/shocking/hybrid_ss20_thr0.1_imgthr0.3_both` |
| i2p_multi_illegal_activity_anchor | — | — | — | NOT FOUND |
| i2p_multi_illegal_activity_hybrid | 46.7 | 25.0 | 60 | `launch_0420_i2p/ours_sd14_multi/illegal_activity/hybrid_ss15_thr0.1_imgthr0.3_both` |
| i2p_multi_harassment_anchor | — | — | — | NOT FOUND |
| i2p_multi_harassment_hybrid | 33.3 | 50.0 | 60 | `launch_0420_i2p/ours_sd14_multi/harassment/hybrid_ss20_thr0.1_imgthr0.3_both` |
| i2p_multi_hate_anchor | — | — | — | NOT FOUND |
| i2p_multi_hate_hybrid | 36.7 | 48.3 | 60 | `launch_0420_i2p/ours_sd14_multi/hate/hybrid_ss20_thr0.1_imgthr0.3_both` |

## 07_ablation_sd14_probe

| Cell | SR (%) | Full (%) | N | Source |
|---|---|---|---|---|
| violence_txtonly | 86.7 | 8.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_txtonly/violence/hybrid_ss20_thr0.2_txtonly` |
| violence_imgonly | 86.7 | 5.0 | 60 | `launch_0420_i2p/ours_sd14_ablation_imgonly/violence/hybrid_ss20_thr0.1_imgthr0.1_imgonly` |
| violence_both | 91.7 | 8.3 | 60 | `launch_0420_i2p/ours_sd14_grid_v1pack/violence/hybrid_ss15_thr0.1_imgthr0.3_both` |
| self-harm_txtonly | 55.0 | 18.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_txtonly/self-harm/hybrid_ss20_thr0.3_txtonly` |
| self-harm_imgonly | 55.0 | 28.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_imgonly_repatched/self-harm/hybrid_ss15_imgthr0.4_imgonly` |
| self-harm_both | 68.3 | 18.3 | 60 | `launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both` |
| shocking_txtonly | 60.0 | 28.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_txtonly/shocking/hybrid_ss15_thr0.1_txtonly` |
| shocking_imgonly | 78.3 | 20.0 | 60 | `launch_0420_i2p/ours_sd14_ablation_imgonly/shocking/hybrid_ss20_thr0.1_imgthr0.1_imgonly` |
| shocking_both | 88.3 | 11.7 | 60 | `launch_0423_shocking_imgheavy/hybrid_ss22_thr0.15_imgthr0.1_both` |
| illegal_activity_txtonly | 43.3 | 23.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_txtonly/illegal_activity/hybrid_ss15_thr0.1_txtonly` |
| illegal_activity_imgonly | 38.3 | 23.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_imgonly/illegal_activity/hybrid_ss20_thr0.1_imgthr0.1_imgonly` |
| illegal_activity_both | 48.3 | 20.0 | 60 | `launch_0420_i2p/ours_sd14_v1pack_repatched/illegal_activity/hybrid_ss20_thr0.1_imgthr0.5_both` |
| harassment_txtonly | 38.3 | 35.0 | 60 | `launch_0420_i2p/ours_sd14_ablation_txtonly/harassment/hybrid_ss20_thr0.1_txtonly` |
| harassment_imgonly | 46.7 | 35.0 | 60 | `launch_0420_i2p/ours_sd14_ablation_imgonly/harassment/hybrid_ss20_thr0.1_imgthr0.1_imgonly` |
| harassment_both | 71.7 | 18.3 | 60 | `launch_0424_v5/i2p_harassment/anchor_inpaint_ss2.5_thr0.1_imgthr0.3_cas0.5_both` |
| hate_txtonly | 51.7 | 33.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_txtonly/hate/hybrid_ss20_thr0.1_txtonly` |
| hate_imgonly | 60.0 | 18.3 | 60 | `launch_0420_i2p/ours_sd14_ablation_imgonly/hate/hybrid_ss20_thr0.1_imgthr0.1_imgonly` |
| hate_both | 66.7 | 16.7 | 60 | `launch_0423_harhate_imgheavy/hate/hybrid_ss22_thr0.25_imgthr0.1_both` |

