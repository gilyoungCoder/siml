# SD3.0 / FLUX.1-dev Ours I2P-7 Sweep Plan (prepared 2026-04-29 KST)

Purpose: tomorrow's run should find best **Ours** configs for **SD3.0** and **FLUX.1-dev** on the **I2P Top-60 7-concept split**: sexual, violence, self-harm, shocking, illegal_activity, harassment, hate.

This folder is intentionally self-contained for OMC handoff:

- `concepts.tsv` — prompt/pack/eval mapping and SD1.4 reference configs.
- `matrix_sd3.tsv` — SD3 candidate cells.
- `matrix_flux1.tsv` — FLUX.1-dev candidate cells.
- `run_matrix.sh` — generation + VLM eval worker. Not launched yet.
- `launch_nohup_template.sh` — tomorrow's nohup launch examples. Not launched yet.
- `collect_results.py` — summarize `results_qwen3_vl_*_v5.txt` into CSV/Markdown.

Recommended output root:

```text
/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_sd3_flux_i2p7_sweep
```

## Fixed settings

- Base seed: `42`
- 1 image per prompt
- Prompt split: `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/*_sweep.txt` (all 7 files have 60 prompts)
- Eval: `/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py`, Qwen, concept mapping in `concepts.tsv`
- SD3: `steps=28`, `cfg_scale=7.0`, `resolution=1024`, generator `scripts/sd3/generate_sd3_safegen.py`
- FLUX.1-dev: `steps=28`, `guidance_scale=3.5`, `height=width=512`, generator `CAS_SpatialCFG/generate_flux1_v1.py`

## Design rationale

### SD3

Use SD1.4 best configs as the center, because SD3 prior MJA runs still needed large hybrid correction scales (`ss≈15–25`) while anchor cells stayed small (`ss≈1.5–3.0`). The SD3 matrix therefore includes:

1. direct SD1.4 hybrid port,
2. lower/higher hybrid strength variants,
3. image-tight mask variant,
4. CAS sensitivity variant,
5. two anchor sanity cells.

Total SD3 cells: **49** = 7 per concept × 7 concepts.

### FLUX.1-dev

Do **not** port SD1.4 `safety_scale=20+` to FLUX. FLUX uses embedded guidance, and previous FLUX MJA bests live around `safety_scale=1.5–3.0`. The FLUX matrix therefore uses low-scale anchor/hybrid cells plus one concept-specific sensitivity cell.

Total FLUX cells: **56** = 8 per concept × 7 concepts.

## Recommended tomorrow execution order

1. **Pilot first**: run `MODE=pilot` (`end_idx=12`) for both models. This costs 588 SD3 images + 672 FLUX images before eval.
2. Collect pilot summaries with `collect_results.py`.
3. Pick top 2 configs per concept/model, then run full 60 prompts for those. If time is abundant, run full matrix directly.
4. Final selection rule: maximize SR first, then use Full-rate and image quality as tie-breakers. Watch NotRelevant/NotPeople inflation; high SR with excessive NR should be flagged, not blindly chosen.

## Important CAS/family note

CAS is concept-level, not family-level. For a prompt/timestep, the code compares prompt prediction direction against a single concept descriptor target. Once CAS triggers, the WHERE/HOW stage applies per-family masks and per-family target/anchor predictions. So do not describe CAS as averaged over subfamilies; only the concept descriptor strings are averaged into the global target embedding.

## Path note for Junhyung/baseline message

If referring to `i2p_q16_top60`, filenames are `*_q16_top60.txt`, not `*_sweep.txt`. For the current Ours sweep, use `i2p_sweep60/*_sweep.txt` because it includes all 7 concepts including `sexual_sweep.txt`.
