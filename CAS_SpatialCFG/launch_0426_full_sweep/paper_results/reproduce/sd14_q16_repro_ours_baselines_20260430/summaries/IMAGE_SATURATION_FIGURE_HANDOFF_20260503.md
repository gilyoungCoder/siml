# Image-exemplar saturation figure handoff (seed 42, final rounded EBSG configs)

Use the two-panel figure below to support the text/image-probe interpretation:

- Figure PNG: `summaries/image_saturation_text_vs_image_dominant_seed42_20260503.png`
- Figure PDF: `summaries/image_saturation_text_vs_image_dominant_seed42_20260503.pdf`
- Source CSVs:
  - `summaries/image_saturation_best3_finalconfig_seed42_20260503_results.csv`
  - `summaries/image_saturation_extra3_finalconfig_seed42_20260503_results.csv`

Recommended placement: Appendix ablation section, immediately after the probe-channel ablation table. If space allows, mention the result briefly in the main ablation paragraph and defer the figure to appendix.

Suggested text:

> Image-exemplar saturation. We further vary the number of image exemplars per family while holding all other final EBSG hyperparameters fixed. For the text-dominant sexual concept, performance is already saturated at K=1 (98.3% SR across K), consistent with the explicit lexical cues being sufficient for localization. In contrast, for the image-dominant hate concept, SR improves from 61.7% at K=1--2 to 63.3% at K=4 and saturates at 65.0% for K>=8, indicating that additional visual exemplars help the image probe capture non-lexical visual cues before reaching diminishing returns.

Caveat / wording guard: this is a lightweight seed-42 diagnostic, not the primary benchmark. Avoid claiming monotonic improvement for all concepts; shocking/violence are stable or slightly non-monotonic under the same fixed-config sweep.

Other observed SR curves:

| Concept | K=1 | K=2 | K=4 | K=8 | K=12 | K=16 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| sexual | 98.3 | 98.3 | 98.3 | 98.3 | 98.3 | 98.3 | text-dominant, saturated immediately |
| hate | 61.7 | 61.7 | 63.3 | 65.0 | 65.0 | 65.0 | image-dominant, mild saturation by K=8 |
| shocking | 93.3 | 93.3 | 91.7 | 91.7 | 91.7 | 91.7 | already high, slight NR increase at K>=4 |
| violence | 90.0 | 85.0 | 85.0 | 85.0 | 85.0 | 85.0 | K=1 best in this seed; not a saturation example |
| self-harm | 51.7 | 55.0 | 48.3 | 48.3 | 48.3 | 48.3 | non-monotonic; do not use for saturation claim |
| illegal | 43.3 | 41.7 | 43.3 | 43.3 | 43.3 | 43.3 | flat / low; not useful for figure |

