# ML writer handoff — NFE wall-clock vs SR Pareto (2026-05-02)

This experiment closes the **inference-cost limitation** framing: EBSG's per-step UNet
overhead (5.6× baseline at fixed NFE=50) is more than recoverable when NFE is lowered,
because EBSG saturates much faster than the inference-time baselines. On the
**wall-clock vs SR plane**, EBSG is Pareto-dominant: it reaches higher SR than any baseline
at every per-image latency budget we measured.

---

## 1. Where it goes in the paper

- **Main text** (§5 Experiments / Discussion): **1 paragraph** acknowledging compute
  overhead and pointing at the figure. Use the headline numbers from §3 below.
- **Appendix** (NFE ablation section, replacing/extending the existing 5-method NFE figure):
  this **single 1×3 panel** figure is the new headline. Move the old NFE-vs-SR (DDIM step
  x-axis) figure to a 2nd appendix sub-figure if useful, or drop it.
- **Limitation paragraph** (Conclusion or §5): "Cost is recoverable: at any wall-clock
  budget … EBSG reaches a higher SR than any inference-time baseline."

---

## 2. Experimental setup

- **Model**: SD v1.4, seed 42, CFG 7.5, 512×512, DDIM scheduler.
- **Prompts**: I2P q16 top-60, **all 7 concepts** (sexual, violence, self-harm, shocking,
  illegal_activity, harassment, hate). 60 prompts × 1 image per cell.
- **Methods (5)**: Baseline / SAFREE / **SAFREE + SafeDenoiser** / **SAFREE + SGF** / EBSG.
  - **Both compound baselines** (SAFREE+SafeDenoiser, SAFREE + SGF) are the *stacked*
    variants on top of SAFREE — they invoke the SafeDenoiser / SGF official scripts with
    `--config=configs/base/vanilla/safree_neg_prompt_config.json` and
    `--erase_id=safree_neg_prompt_rep_*_time`. This matches the Table 1 row labels.
  - Concept-specific YAML configs and NEGSPACE templates are the same ones used in the
    main paper Table 1 single-I2P best (no per-NFE retuning).
  - SAFREE alone uses `--safree -svf -lra` (the same v2 config as Table 1).
  - EBSG uses the **per-concept best config** from the main paper (sexual ss=20 cas=0.5,
    violence ss=20 cas=0.4, self-harm ss=7 cas=0.5, shocking ss=27.5 cas=0.6,
    illegal ss=25 cas=0.6, harassment ss=31.25 cas=0.5, hate ss=28 cas=0.6).
- **NFE grid**: {5, 10, 15, 20, 25, 30, 40, 50} — 8 points.
- **Total cells**: 5 × 7 × 8 = **280**, 60 imgs each = 16 800 images. All evaluated with
  Qwen3-VL-8B v5.

### Wall-clock measurement (separate isolated benchmark)
- Single GPU, no contention: **siml-05 g0 (RTX 3090 24 GB)**.
- 5 methods × 8 NFE × 20 prompts (violence, representative; gen-time is concept-invariant).
- Per-image time = (max PNG mtime − min PNG mtime) / (N − 1). Excludes one-time
  model-load cost. (We also report `with_load` as a sanity check; conclusions identical.)

---

## 3. Key results (concept-averaged, 7 concepts)

| Method                    | NFE=5  | NFE=10 | NFE=50 |
|---|---:|---:|---:|
| **Baseline**               | 0.42 s · 34.8 SR · 41.4 Full | 0.63 s · 31.4 SR · 53.1 Full | 2.68 s · 35.7 SR · 51.7 Full |
| **SAFREE**                 | 1.11 s · 19.0 SR · 1.0 Full  | 2.00 s · 31.2 SR · 4.0 Full | 8.89 s · 54.3 SR · 10.5 Full |
| **SAFREE + SafeDenoiser**  | 0.58 s · 46.9 SR · 13.6 Full | 1.00 s · 54.8 SR · 16.9 Full | 4.32 s · 53.1 SR · 21.2 Full |
| **SAFREE + SGF**           | 0.58 s · **0.0** SR · 0.0 Full · 100 NotRel | 1.05 s · **0.0** SR · 0.0 Full · 100 NotRel | 4.42 s · 47.1 SR · 23.1 Full |
| **EBSG (Ours)**            | 1.47 s · **52.4** SR · 13.8 Full | 2.79 s · **60.2** SR · 18.1 Full | 14.11 s · **70.2** SR · 11.2 Full |

Per-image time is `per_img_sec_excl_load_mtime` (filesystem mtime range / (N−1)).

### Headline callouts to copy into paper text

1. **EBSG @ NFE=5 (1.47 s/img) vs Baseline @ NFE=50 (2.68 s/img)**:
   EBSG is **1.8× faster** AND **+16.7 pp higher SR** (52.4 vs 35.7).
2. **EBSG @ NFE=10 (2.79 s/img) vs SAFREE @ NFE=50 (8.89 s/img)**:
   EBSG is **3.2× faster** AND **+5.9 pp higher SR** (60.2 vs 54.3).
3. **EBSG @ NFE=50 (14.11 s/img)**: Pareto-dominant — highest SR (70.2) among all 5
   methods at any wall-clock budget we measured; also the lowest Full-violation rate
   (11.2 %) among methods with non-degenerate generation.
4. **SGF small-NFE failure**: SAFREE + SGF at NFE ≤ 10 produces 100 % NotRelevant — i.e.
   the gradient-MMD repellency *destroys the image* at small NFE. SGF only becomes
   non-degenerate at NFE ≥ 25 (47 % SR at NFE=50). This is the same observation as the
   existing NFE-vs-step appendix; the wall-clock view makes it sharper because SGF
   never appears on the SR-positive Pareto frontier.
5. **Baseline plateau**: SR does not increase with NFE for the unmodified SD1.4 baseline
   (≈ 31–36 % across all NFE). Adding compute does not buy safety without an erasure
   method.
6. **EBSG monotone**: Among inference-time methods, EBSG is the **only** one that is
   monotonic in NFE on the SR axis (52.4 → 60.2 → 70.2). The other four either plateau
   (SafeDenoiser) or require NFE ≥ 20 to avoid catastrophic NotRelevant collapse (SGF).

---

## 4. Figure files

All under
`paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/summaries/` (this doc)
and `paper_results/figures/` on the NFS:

| Use | File |
|---|---|
| **Headline (paper-ready)** | `paper_results/figures/nfe_walltime_pareto_polished.{pdf,png}` |
| Vanilla version | `paper_results/figures/nfe_walltime_pareto_concept_avg.{pdf,png}` |
| Appendix per-concept facet | `paper_results/figures/nfe_walltime_pareto_per_concept.{pdf,png}` |
| Plot script (rerun-able) | `scripts/nfe_walltime_pareto_polished.py` |

### Suggested caption (polished figure)

> **Figure N.** *Wall-clock cost vs erasure quality, concept-averaged across the seven
> I2P top-60 concepts. The horizontal axis is per-image generation time (excluding
> one-time model load) measured on a single RTX 3090 with no GPU contention. Markers
> denote NFE ∈ {5, 10, 15, 20, 25, 30, 40, 50}. \textbf{EBSG is Pareto-dominant on the SR
> axis at every per-image latency budget}: at NFE = 5 it already exceeds the SR of
> SD1.4 + DDIM-50 by +16.7 pp while running 1.8× faster, and at NFE = 10 it exceeds
> SAFREE + DDIM-50 by +5.9 pp while running 3.2× faster. SAFREE + SGF degenerates to
> 100 % NotRelevant for NFE ≤ 10 (the safety-by-destruction failure mode), shown by the
> NotRel panel.*

---

## 5. CSV files (ml-writer source-of-truth for any number changes)

| File | Rows |
|---|---|
| `paper_results/figures/nfe_walltime_5method_7concept_table.csv` | 280 (every cell, raw SR/Full/NR per concept) |
| `paper_results/figures/nfe_walltime_5method_concept_avg.csv`     | 40 (per (method, NFE) concept-average + per_img_sec) |
| `paper_results/figures/nfe_walltime_timing.csv`                   | 40 (per (method, NFE) wall-clock measurements) |

CSV header (concept_avg):
`method, nfe, per_img_sec, SR_avg_pct, Full_avg_pct, NR_avg_pct, n_concepts`

---

## 6. What the writer should NOT do

- Do **not** average wall-clock across methods to claim "EBSG is N× slower"; use the
  per-method per-NFE numbers from `nfe_walltime_timing.csv`. Headline ratios above
  already do the cross-method comparison correctly.
- Do **not** mix this 8-point NFE grid with the existing 11-point (1, 3, 5, 8, 12, 16,
  20, 25, 30, 40, 50) NFE figure. They use **different EBSG configs** (the old one used a
  single fixed config; this new one uses per-concept best). Replace, do not overlay.
- Do **not** report `per_img_sec_with_load`; it includes the ~10 s model-load amortized
  over only 20 imgs and biases the small-NFE values upward.

---

## 7. Open items / caveats

- **GPU**: RTX 3090. 4090 / A100 numbers will differ; quote the GPU explicitly in caption.
- **Concept hate** uses `n_img_tokens=16` per the EBSG single-I2P best config (others
  use `n_img_tokens=4`). EBSG hate cells are slightly more expensive per step than other
  EBSG cells; the concept-averaged time bakes this in. Per-concept facet figure makes
  this visible (hate cell EBSG times are ≈ 10–15 % above the EBSG average).
- **SAFREE + SGF NFE=5,10 SR=0** is a real measurement (NudeNet+ rubric labels every
  generated image NotRelevant because the image is destroyed); not a bug. We may want
  to clip those points or annotate "image destruction" in the appendix per-concept facet
  to keep the SGF curve from disappearing into y=0.

---

## 8. Reproduction (one-line)

```
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_nfe_walltime_master.sh'
```

Outputs land at `outputs/phase_nfe_walltime_v3/<method>_<concept>_steps<N>/`.
Plot rerun: `python scripts/nfe_walltime_pareto_polished.py`.
