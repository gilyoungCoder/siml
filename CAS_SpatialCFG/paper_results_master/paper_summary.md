## 01_nudity_sd14_5bench
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| rab_hybrid | 97.4 | 2.6 | 78 | hybrid/both cas=0.6 ss=20.0 txt=0.1 img=0.4 | concepts_v2/sexual/clip_grouped.pt | ringabell.txt |
| mma_anchor | 77.0 | 23.0 | 993 | anchor_inpaint/both cas=0.6 ss=1.2 txt=0.1 img=0.3 | concepts_v2/sexual/clip_grouped.pt | mma.txt |
| mma_hybrid | 84.7 | 15.3 | 996 | hybrid/both cas=0.6 ss=20.0 txt=0.1 img=0.3 | i2p_v1/sexual/clip_grouped.pt | mma.txt |
| p4dn_anchor | 91.2 | 8.8 | 148 | anchor_inpaint/both cas=0.6 ss=1.2 txt=0.1 img=0.3 | concepts_v2/sexual/clip_grouped.pt | p4dn.txt |
| p4dn_hybrid | 97.4 | 2.6 | 151 | hybrid/both cas=0.6 ss=20.0 txt=0.1 img=0.3 | i2p_v1/sexual/clip_grouped.pt | p4dn.txt |
| unlearndiff_anchor | 92.9 | 7.1 | 140 | anchor_inpaint/both cas=0.6 ss=1.2 txt=0.1 img=0.3 | concepts_v2/sexual/clip_grouped.pt | unlearndiff.txt |
| unlearndiff_hybrid | 98.6 | 1.4 | 140 | hybrid/both cas=0.6 ss=10.0 txt=0.1 img=0.3 | i2p_v1/sexual/clip_grouped.pt | unlearndiff.txt |
| mja_sexual_anchor | 90.0 | 10.0 | 90 | anchor_inpaint/both cas=0.6 ss=2.5 txt=0.1 img=0.3 | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_sexual_hybrid | 96.7 | 3.3 | 90 | hybrid/both cas=0.6 ss=15.0 txt=0.1 img=0.5 | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |

## 02_i2p_top60_sd14_6concept
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| violence_anchor | 88.3 | 3.3 | 60 | anchor_inpaint/both cas=0.6 ss=1.0 txt=0.1 img=0.4 | i2p_v1/violence/clip_grouped.pt | i2p_sweep60/violence_sweep.txt |
| violence_hybrid | 91.7 | 8.3 | 60 | hybrid/both cas=0.6 ss=15.0 txt=0.1 img=0.3 | i2p_v1/violence/clip_grouped.pt | i2p_sweep60/violence_sweep.txt |
| self-harm_anchor | 68.3 | 25.0 | 60 | anchor_inpaint/image cas=0.6 ss=1.5 txt=0.1 img=0.4 | i2p_v1/self-harm/clip_grouped.pt | i2p_sweep60/self-harm_sweep.txt |
| self-harm_hybrid | 61.7 | 6.7 | 60 | hybrid/both cas=0.6 ss=22.0 txt=0.1 img=0.4 | i2p_v1/self-harm/clip_grouped.pt | i2p_sweep60/self-harm_sweep.txt |
| shocking_anchor | 78.3 | 18.3 | 60 | anchor_inpaint/both cas=0.6 ss=2.0 txt=0.1 img=0.4 | i2p_v1/shocking/clip_grouped.pt | i2p_sweep60/shocking_sweep.txt |
| shocking_hybrid | 88.3 | 11.7 | 60 | hybrid/both cas=0.6 ss=22.0 txt=0.15 img=0.1 | i2p_v1/shocking/clip_grouped.pt | i2p_sweep60/shocking_sweep.txt |
| illegal_activity_anchor | 46.7 | 20.0 | 60 | anchor_inpaint/both cas=0.6 ss=1.0 txt=0.1 img=0.7 | i2p_v1/illegal_activity/clip_grouped.pt | i2p_sweep60/illegal_activity_sweep.txt |
| illegal_activity_hybrid | 48.3 | 20.0 | 60 | hybrid/both cas=0.6 ss=20.0 txt=0.1 img=0.5 | i2p_v1/illegal_activity/clip_grouped.pt | i2p_sweep60/illegal_activity_sweep.txt |
| harassment_anchor | 71.7 | 18.3 | 60 | anchor_inpaint/both cas=0.5 ss=2.5 txt=0.1 img=0.3 | i2p_v1/harassment/clip_grouped.pt | i2p_sweep60/harassment_sweep.txt |
| harassment_hybrid | 56.7 | 26.7 | 60 | hybrid/both cas=0.5 ss=25.0 txt=0.1 img=0.5 | i2p_v1/harassment/clip_grouped.pt | i2p_sweep60/harassment_sweep.txt |
| hate_anchor | 60.0 | 33.3 | 60 | anchor_inpaint/both cas=0.6 ss=2.0 txt=0.1 img=0.4 | i2p_v1/hate/clip_grouped.pt | i2p_sweep60/hate_sweep.txt |
| hate_hybrid | 66.7 | 16.7 | 60 | hybrid/both cas=0.6 ss=22.0 txt=0.25 img=0.1 | i2p_v1/hate/clip_grouped.pt | i2p_sweep60/hate_sweep.txt |

## 03_mja_sd14_4concept
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| mja_sexual_anchor | 90.0 | 10.0 | 90 | anchor_inpaint/both cas=0.6 ss=2.5 txt=0.1 img=0.3 | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_sexual_hybrid | 97.6 | 2.4 | 85 | hybrid/both cas=0.6 ss=22.0 txt=0.15 img=0.1 | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_violent_anchor | 56.0 | 26.0 | 100 | anchor_inpaint/both cas=0.6 ss=1.8 txt=0.1 img=0.3 | concepts_v2/violent/clip_grouped.pt | mja_violent.txt |
| mja_violent_hybrid | 69.0 | 16.0 | 100 | hybrid/both cas=0.4 ss=25.0 txt=0.15 img=0.1 | concepts_v2/violent/clip_grouped.pt | mja_violent.txt |
| mja_illegal_anchor | 76.0 | 8.0 | 100 | anchor_inpaint/both cas=0.4 ss=2.5 txt=0.1 img=0.3 | concepts_v2/illegal/clip_grouped.pt | mja_illegal.txt |
| mja_illegal_hybrid | 71.0 | 16.0 | 100 | hybrid/both cas=0.1 ss=30.0 txt=0.1 img=0.6 | concepts_v2/illegal/clip_grouped.pt | mja_illegal.txt |
| mja_disturbing_anchor | 96.0 | 2.0 | 100 | anchor_inpaint/both cas=0.6 ss=2.0 txt=0.1 img=0.4 | concepts_v2/disturbing/clip_grouped.pt | mja_disturbing.txt |
| mja_disturbing_hybrid | 93.0 | 5.0 | 100 | hybrid/both cas=0.6 ss=22.0 txt=0.15 img=0.1 | concepts_v2/disturbing/clip_grouped.pt | mja_disturbing.txt |

## 04_mja_sd3_4concept
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| mja_sexual_anchor | 81.0 | 19.0 | 100 | anchor_inpaint/both cas=0.6 ss=3.0 txt=0.2 img=0.2 | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_sexual_hybrid | 84.0 | 16.0 | 100 | hybrid/both cas=0.6 ss=15.0 txt=0.1 img=0.3 | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_violent_anchor | 58.0 | 42.0 | 100 | anchor_inpaint/both cas=0.6 ss=1.5 txt=0.1 img=0.1 | concepts_v2/violent/clip_grouped.pt | mja_violent.txt |
| mja_violent_hybrid | 36.0 | 57.0 | 100 | hybrid/both cas=0.3 ss=20.0 txt=0.15 img=0.1 | concepts_v2/violent/clip_grouped.pt | mja_violent.txt |
| mja_illegal_anchor | 53.0 | 42.0 | 100 | anchor_inpaint/both cas=0.6 ss=2.5 txt=0.1 img=0.1 | concepts_v2/illegal/clip_grouped.pt | mja_illegal.txt |
| mja_illegal_hybrid | 67.0 | 16.0 | 100 | hybrid/both cas=0.3 ss=20.0 txt=0.15 img=0.1 | concepts_v2/illegal/clip_grouped.pt | mja_illegal.txt |
| mja_disturbing_anchor | 86.0 | 14.0 | 100 | anchor_inpaint/both cas=0.6 ss=1.5 txt=0.1 img=0.1 | concepts_v2/disturbing/clip_grouped.pt | mja_disturbing.txt |
| mja_disturbing_hybrid | 90.0 | 10.0 | 100 | hybrid/both cas=0.4 ss=20.0 txt=0.15 img=0.1 | concepts_v2/disturbing/clip_grouped.pt | mja_disturbing.txt |

## 05_mja_flux1_4concept
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| mja_sexual_anchor | 96.0 | 4.0 | 100 | anchor_inpaint/both cas=0.6 ss=1.5 txt=0.1 img=? | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_sexual_hybrid | 97.0 | 3.0 | 100 | hybrid/both cas=0.6 ss=2.5 txt=0.1 img=? | concepts_v2/sexual/clip_grouped.pt | mja_sexual.txt |
| mja_violent_anchor | 89.0 | 11.0 | 100 | anchor_inpaint/both cas=0.6 ss=2.0 txt=0.1 img=? | concepts_v2/violent/clip_grouped.pt | mja_violent.txt |
| mja_violent_hybrid | 67.0 | 20.0 | 100 | hybrid/both cas=0.6 ss=2.0 txt=0.1 img=? | concepts_v2/violent/clip_grouped.pt | mja_violent.txt |
| mja_illegal_anchor | 86.0 | 13.0 | 100 | anchor_inpaint/both cas=0.6 ss=3.0 txt=0.1 img=? | concepts_v2/illegal/clip_grouped.pt | mja_illegal.txt |
| mja_illegal_hybrid | 58.0 | 35.0 | 100 | hybrid/both cas=0.6 ss=2.0 txt=0.1 img=? | concepts_v2/illegal/clip_grouped.pt | mja_illegal.txt |
| mja_disturbing_anchor | 98.0 | 2.0 | 100 | anchor_inpaint/both cas=0.6 ss=1.5 txt=0.1 img=? | concepts_v2/disturbing/clip_grouped.pt | mja_disturbing.txt |
| mja_disturbing_hybrid | 96.0 | 4.0 | 100 | hybrid/both cas=0.6 ss=3.0 txt=0.1 img=? | concepts_v2/disturbing/clip_grouped.pt | mja_disturbing.txt |

## 06_multi_concept_sd14
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| mja_multi_mja_sexual_anchor | 73.0 | 27.0 | 89 | anchor_inpaint/both cas=0.6 ss=3.0 txt=0.1 img=0.1 | multiconcept_v1/sexual+violent/clip_grouped.pt | mja_sexual.txt |
| mja_multi_mja_sexual_hybrid | 43.8 | 56.2 | 89 | hybrid/both cas=0.6 ss=0.5 txt=0.2 img=0.2 | multiconcept_v1/sexual+violent/clip_grouped.pt | mja_sexual.txt |
| mja_multi_mja_violent_anchor | 71.0 | 23.0 | 100 | anchor_inpaint/both cas=0.6 ss=2.5 txt=0.2 img=0.2 | multiconcept_v1/sexual+violent/clip_grouped.pt | mja_violent.txt |
| mja_multi_mja_violent_hybrid | 20.0 | 76.0 | 100 | hybrid/both cas=0.6 ss=1.0 txt=0.2 img=0.2 | multiconcept_v1/sexual+violent/clip_grouped.pt | mja_violent.txt |
| i2p_multi_violence_hybrid | 60.0 | 40.0 | 60 | ? | ? | ? |
| i2p_multi_self-harm_hybrid | 50.0 | 28.3 | 60 | ? | ? | ? |
| i2p_multi_shocking_hybrid | 43.3 | 56.7 | 60 | ? | ? | ? |
| i2p_multi_illegal_activity_hybrid | 46.7 | 25.0 | 60 | ? | ? | ? |
| i2p_multi_harassment_hybrid | 33.3 | 50.0 | 60 | ? | ? | ? |
| i2p_multi_hate_hybrid | 36.7 | 48.3 | 60 | ? | ? | ? |

## 07_ablation_sd14_probe
| Cell | SR | Full | n | Config (mode/cas/ss/txt/img) | family_pack | prompts |
|---|---|---|---|---|---|---|
| violence_txtonly | 86.7 | 8.3 | 60 | hybrid/text cas=0.6 ss=20.0 txt=0.2 img=0.3 | i2p_v1/violence/clip_grouped.pt | i2p_sweep60/violence_sweep.txt |
| violence_imgonly | 86.7 | 5.0 | 60 | hybrid/image cas=0.6 ss=20.0 txt=0.1 img=0.1 | i2p_v1/violence/clip_grouped.pt | i2p_sweep60/violence_sweep.txt |
| violence_both | 91.7 | 8.3 | 60 | hybrid/both cas=0.6 ss=15.0 txt=0.1 img=0.3 | i2p_v1/violence/clip_grouped.pt | i2p_sweep60/violence_sweep.txt |
| self-harm_txtonly | 55.0 | 18.3 | 60 | hybrid/text cas=0.6 ss=20.0 txt=0.3 img=0.3 | i2p_v1/self-harm/clip_grouped.pt | i2p_sweep60/self-harm_sweep.txt |
| self-harm_imgonly | 55.0 | 28.3 | 60 | hybrid/image cas=0.6 ss=15.0 txt=0.1 img=0.4 | i2p_v1/self-harm/clip_grouped.pt | i2p_sweep60/self-harm_sweep.txt |
| self-harm_both | 68.3 | 18.3 | 60 | anchor_inpaint/both cas=0.6 ss=1.0 txt=0.1 img=0.4 | i2p_v1/self-harm/clip_grouped.pt | i2p_sweep60/self-harm_sweep.txt |
| shocking_txtonly | 60.0 | 28.3 | 60 | hybrid/text cas=0.6 ss=15.0 txt=0.1 img=0.3 | i2p_v1/shocking/clip_grouped.pt | i2p_sweep60/shocking_sweep.txt |
| shocking_imgonly | 78.3 | 20.0 | 60 | hybrid/image cas=0.6 ss=20.0 txt=0.1 img=0.1 | i2p_v1/shocking/clip_grouped.pt | i2p_sweep60/shocking_sweep.txt |
| shocking_both | 88.3 | 11.7 | 60 | hybrid/both cas=0.6 ss=22.0 txt=0.15 img=0.1 | i2p_v1/shocking/clip_grouped.pt | i2p_sweep60/shocking_sweep.txt |
| illegal_activity_txtonly | 43.3 | 23.3 | 60 | hybrid/text cas=0.6 ss=15.0 txt=0.1 img=0.3 | i2p_v1/illegal_activity/clip_grouped.pt | i2p_sweep60/illegal_activity_sweep.txt |
| illegal_activity_imgonly | 38.3 | 23.3 | 60 | hybrid/image cas=0.6 ss=20.0 txt=0.1 img=0.1 | i2p_v1/illegal_activity/clip_grouped.pt | i2p_sweep60/illegal_activity_sweep.txt |
| illegal_activity_both | 48.3 | 20.0 | 60 | hybrid/both cas=0.6 ss=20.0 txt=0.1 img=0.5 | i2p_v1/illegal_activity/clip_grouped.pt | i2p_sweep60/illegal_activity_sweep.txt |
| harassment_txtonly | 38.3 | 35.0 | 60 | hybrid/text cas=0.6 ss=20.0 txt=0.1 img=0.3 | i2p_v1/harassment/clip_grouped.pt | i2p_sweep60/harassment_sweep.txt |
| harassment_imgonly | 46.7 | 35.0 | 60 | hybrid/image cas=0.6 ss=20.0 txt=0.1 img=0.1 | i2p_v1/harassment/clip_grouped.pt | i2p_sweep60/harassment_sweep.txt |
| harassment_both | 71.7 | 18.3 | 60 | anchor_inpaint/both cas=0.5 ss=2.5 txt=0.1 img=0.3 | i2p_v1/harassment/clip_grouped.pt | i2p_sweep60/harassment_sweep.txt |
| hate_txtonly | 51.7 | 33.3 | 60 | hybrid/text cas=0.6 ss=20.0 txt=0.1 img=0.3 | i2p_v1/hate/clip_grouped.pt | i2p_sweep60/hate_sweep.txt |
| hate_imgonly | 60.0 | 18.3 | 60 | hybrid/image cas=0.6 ss=20.0 txt=0.1 img=0.1 | i2p_v1/hate/clip_grouped.pt | i2p_sweep60/hate_sweep.txt |
| hate_both | 66.7 | 16.7 | 60 | hybrid/both cas=0.6 ss=22.0 txt=0.25 img=0.1 | i2p_v1/hate/clip_grouped.pt | i2p_sweep60/hate_sweep.txt |

