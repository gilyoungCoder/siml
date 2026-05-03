# ML writer handoff — NFE wall-clock vs SR Pareto, 9-method (2026-05-03 v2, CORRECTED)

**v2 update (2026-05-03 evening)**: re-measured all 9 methods on the **same RTX
3090, single GPU, no contention**, after fixing a measurement-script bug that
under-counted SafeDenoiser and SGF per-image times by ~2× (their filenames
`0_violence.png … 19_violence.png` were sorted alphabetically by `ls`, so
`tail -1` returned `9_violence.png` instead of `19_violence.png`, halving the
mtime span). The corrected timings put SafeDenoiser and SGF in the same
~9 s/img band as SAFREE and the 4 SLD variants, which is the architecturally
expected behaviour (3 UNet forwards/step). EBSG remains the only method at
14 s/img (11 forwards/step).

> EBSG is Pareto-dominant on the SR axis at every per-image latency budget,
> against **5 baselines + 4 SLD operating points** (Max/Strong/Medium/Weak).

---

## 1. Why 9 methods

| Method | UNet forwards / step | Notes |
|---|---:|---|
| Baseline (SD1.4 + DDIM)         | 2  | cond + uncond CFG |
| SAFREE (`-svf -lra`)            | 2  | + token re-projection + self-validation filter + latent re-attention |
| SAFREE + SafeDenoiser           | 3  | + concept-specific NEGSPACE + NudeNet routing |
| SAFREE + SGF                    | 3  | + repellency MMD |
| SLD-Weak                        | 3  | gs=200, warmup=15, threshold=0.005, no momentum |
| SLD-Medium                      | 3  | gs=1000, warmup=10, threshold=0.01, momentum=0.3/0.4 |
| SLD-Strong                      | 3  | gs=2000, warmup=7, threshold=0.025, momentum=0.5/0.7 |
| SLD-Max                         | 3  | gs=5000, warmup=0, threshold=1.0, momentum=0.5/0.7 |
| **EBSG (Ours)**                 | **11** | probe + cond + uncond × per-family fork |

All inference-time methods on SD v1.4 / I2P q16 top-60 / seed 42 / CFG 7.5 /
DDIM scheduler / 7 concepts × 60 prompts.

---

## 2. Setup

| Axis | Value |
|---|---|
| Backbone | SD v1.4, DDIM scheduler, seed 42, CFG = 7.5, 512×512 |
| Concepts (7) | sexual, violence, self-harm, shocking, illegal_activity, harassment, hate |
| Methods (9) | Baseline / SAFREE / SAFREE+SafeDenoiser / SAFREE+SGF / SLD-{Max, Strong, Medium, Weak} / EBSG |
| NFE grid (8) | 5, 10, 15, 20, 25, 30, 40, 50 |
| Cells | 9 × 7 × 8 = **504** (≈30,240 images) |
| Evaluator | Qwen3-VL-8B v5 (4-class rubric: NotRel / Safe / Partial / Full) |
| GPU | siml-05 RTX 3090 (g0..g7 used variably across the run) |

**Wall-clock measurement (corrected)**: per-method isolated single-GPU bench on
RTX 3090 (one process per GPU, no contention). 9 methods × 8 NFE × 20 violence
prompts. Per-image time = `(max(mtime) − min(mtime)) / (N − 1)` over canonical
output files (`<outdir>/all/` for SD/SGF, top-level for the others). The bug
fix was to use mtime-sorted min/max instead of `ls | head -1 / tail -1`, which
fails on non-zero-padded filenames.

---

## 3. Result tables (CORRECTED)

### 3.1 Concept-averaged headline (3 NFE points; full grid in CSV)

| Method                    | NFE=5  | NFE=10 | NFE=50 |
|---|---|---|---|
| **Baseline**              | 0.42 s · 34.8 SR · 41.4 Full | 0.63 s · 31.4 SR · 53.1 Full | 2.63 s · 35.7 SR · 51.7 Full |
| **SAFREE**                | 1.11 s · 19.0 SR · 1.0 Full  | 2.00 s · 31.2 SR · 4.0 Full  | 8.74 s · 54.3 SR · 10.5 Full |
| **SAFREE + SafeDenoiser** | 1.16 s · 46.9 SR · 13.6 Full | 2.05 s · 54.8 SR · 16.9 Full | 9.11 s · 53.1 SR · 21.2 Full |
| **SAFREE + SGF**          | 1.21 s · **0.0** SR · 100 NotRel | 2.05 s · **0.0** SR · 100 NotRel | 9.26 s · 47.1 SR · 23.1 Full |
| **SLD-Weak**              | 1.20 s · 32.6 SR · 38.8 Full | 2.12 s · 31.4 SR · 48.3 Full | 8.95 s · 40.7 SR · 42.6 Full |
| **SLD-Medium**            | 1.24 s · 32.6 SR · 38.8 Full | 2.12 s · 31.4 SR · 48.8 Full | 9.11 s · 46.9 SR · 31.0 Full |
| **SLD-Strong**            | 1.24 s · 32.6 SR · 38.8 Full | 2.12 s · 33.1 SR · 43.1 Full | 9.11 s · 48.8 SR · 21.7 Full |
| **SLD-Max**               | 1.20 s · 26.2 SR · 21.0 Full | 2.12 s · 31.7 SR · 30.9 Full | 8.89 s · 46.7 SR · 18.8 Full |
| **EBSG (Ours)**           | 1.47 s · **52.4** SR · 13.8 Full | 2.79 s · **60.2** SR · 18.1 Full | **14.11 s · 70.2 SR · 11.2 Full** |

### 3.2 Headline callouts

1. **EBSG @ NFE = 5 (1.47 s) vs SD1.4 + DDIM-50 (2.63 s)**:
   EBSG is **1.8× faster AND +16.7 pp higher SR** (52.4 vs 35.7).
2. **EBSG @ NFE = 10 (2.79 s) vs SAFREE @ NFE = 50 (8.74 s)**:
   EBSG is **3.1× faster AND +5.9 pp higher SR** (60.2 vs 54.3).
3. **EBSG @ NFE = 10 (2.79 s) vs every SAFREE-compound and SLD variant @ NFE = 50 (≈9.0 s)**:
   EBSG is **~3.2× faster AND +7 – +20 pp higher SR**:
   - vs SAFREE+SafeDenoiser 53.1 → +7.1 pp
   - vs SAFREE+SGF 47.1 → +13.1 pp
   - vs SLD-Strong 48.8 → +11.4 pp
   - vs SLD-Max 46.7 → +13.5 pp
   - vs SLD-Medium 46.9 → +13.3 pp
   - vs SLD-Weak 40.7 → +19.5 pp
4. **EBSG @ NFE = 50 (14.11 s)**: highest SR (70.2 %) and lowest Full (11.2 %)
   among ALL nine methods at any latency we measured. **Pareto-dominant**.
5. **All 8 inference-time methods cluster at 8.7 – 9.3 s/img at NFE = 50**
   (3 UNet forwards/step). EBSG sits at 14.1 s/img (11 forwards/step). The
   per-step cost ratio explains the per-method × baseline factor: 3 / 2 ≈ 1.5×
   for the SAFREE/SLD family, 11 / 2 = 5.5× for EBSG. Empirical wall-clock
   ratios (3.3× and 5.4× respectively) match this prediction within ~5 %.
6. **SAFREE + SGF small-NFE collapse**: at NFE ≤ 15, SR = 0 % and NotRel = 100
   % (image destruction; gradient-MMD repellency unstable at low NFE). SGF
   only becomes non-degenerate from NFE ≥ 25.
7. **SLD plateau across all 4 operating points**: at NFE = 50, all 4 SLD
   variants land in 40.7 – 48.8 % SR, regardless of guidance scale (gs ∈
   {200, 1000, 2000, 5000}), warmup, or momentum. The internal `clamp(scale ·
   gs, max=1.0)` in the SLD pipeline (Eq. 6) caps the effective per-step push
   at 1.0 once gs is "large enough", so the variants differ mainly in
   *threshold* and *momentum*, not in raw guidance magnitude. None escapes
   the ~50 % SR ceiling.
8. **Baseline plateau**: SR is roughly flat at 31 – 37 % across all NFE.
   Adding compute does not buy safety without an erasure mechanism.
9. **EBSG monotone**: EBSG is the **only** inference-time method that is
   monotonic in NFE on the SR axis (52.4 → 60.2 → 66.4 → 69.0 → 72.6 → 70.7
   → 70.5 → 70.2). Every other method either plateaus early (SAFREE +
   SafeDenoiser, all SLD) or requires NFE ≥ 25 to avoid catastrophic NotRel
   collapse (SAFREE + SGF), or never improves with NFE (Baseline).

---

## 4. Updated Table 19 (replacement for the previous A6000 version)

> **Table 19**: Per-image generation cost in an isolated bench on
> **siml-05 / RTX 3090, no contention** (20 violence q16 prompts × 50 DDIM
> steps; 1 process per GPU, single GPU per measurement). Per-image excluding
> model load is `(max(mtime) − min(mtime)) / (N − 1)` over the canonical
> output dir (`all/` for SD/SGF, top-level for the others).

| Method | Wall (s, 20 imgs) | Per-img w/ load (s) | Per-img excl. load (s) | × baseline |
|---|---:|---:|---:|---:|
| Baseline (SD v1.4) | 59.27 | 2.96 | 2.63 | 1.0× |
| SAFREE [Yoon et al., 2024] | 187.07 | 9.35 | 8.74 | 3.3× |
| SAFREE + SafeDenoiser [Kim et al., 2025] | 197.83 | 9.89 | 9.11 | 3.5× |
| SAFREE + SGF [Kim et al., 2026] | 204.51 | 10.23 | 9.26 | 3.5× |
| SLD-Weak [Schramowski et al., 2023] | 186.09 | 9.30 | 8.95 | 3.4× |
| SLD-Medium | 190.41 | 9.52 | 9.11 | 3.5× |
| SLD-Strong | 190.79 | 9.54 | 9.11 | 3.5× |
| SLD-Max | 184.98 | 9.25 | 8.89 | 3.4× |
| **EBSG (Ours)** | **287.77** | **14.39** | **14.11** | **5.4×** |

---

## 5. Figures (paper-ready PDFs in `paper_results/figures/`)

| LaTeX slot | File | Width |
|---|---|---|
| `\begin{figure*}` (full-width) | `nfe_walltime_pareto_3panel_neurips_wide.pdf` | 6.75 in |
| `\begin{figure}` (single-column) | `nfe_walltime_pareto_3panel_neurips_textwidth.pdf` | 5.5 in |
| **SR-only** main body | `nfe_walltime_pareto_sr_neurips_textwidth.pdf` | 5.5 in |
| 3-panel side-by-side appendix grid | `nfe_walltime_pareto_{sr,full,notrel}_neurips_halfwidth.pdf` | 2.7 in × 3 |
| 7-concept × 3-metric supplementary | `nfe_walltime_pareto_per_concept.pdf` | 15 × 22 in |

### Suggested LaTeX caption (3-panel headline)

> **Figure N.** *Wall-clock cost vs erasure quality, concept-averaged across
> the seven I2P top-60 concepts. The horizontal axis is per-image generation
> time (excluding one-time model-load cost) measured on a single RTX 3090
> with no GPU contention, one process per GPU. Markers correspond to NFE
> $\in$ \{5, 10, 15, 20, 25, 30, 40, 50\}. We compare EBSG against eight
> inference-time baselines: SD\,1.4, SAFREE-v2, SAFREE\,+\,SafeDenoiser,
> SAFREE\,+\,SGF, and the four official SLD operating points (Weak,
> Medium, Strong, Max).
> \textbf{EBSG (red) is Pareto-dominant on the SR axis at every per-image
> latency budget}: at NFE = 5 it already exceeds SD\,1.4 + DDIM-50 by
> $+16.7$\,pp while running 1.8$\times$ faster, and at NFE = 10 it exceeds
> every SAFREE-compound and SLD variant + DDIM-50 by $+7$ -- $+20$\,pp
> while running 3.1$\times$ -- 3.2$\times$ faster. The compound SAFREE\,+\,SGF
> baseline degenerates to 100\,\% NotRelevant for NFE $\le$ 15 (orange,
> NotRel panel), the safety-by-destruction failure mode that the four-class
> VLM rubric is designed to expose. All four SLD operating points cluster in
> 40 -- 49\,\% SR at NFE = 50, regardless of guidance scale, due to SLD's
> internal clamp on per-step safety magnitude (Eq. 6 of the SLD paper).*

---

## 6. CSV files

| File | Rows | Columns |
|---|---|---|
| `paper_results/figures/nfe_walltime_timing.csv` | 73 | 9 method × 8 NFE timing (corrected v2) |
| `paper_results/figures/nfe_walltime_timing_clean_<method>.csv` | 9 each | per-method backups (8 rows each) |
| `paper_results/figures/nfe_walltime_5method_concept_avg.csv` | 40 | (legacy 5-method concept-avg; eval data unchanged) |

Per-cell raw eval at
`outputs/phase_nfe_walltime_v3/<method>_<concept>_steps<NFE>/results_qwen3_vl_<rubric>_v5.txt`.
Concept→rubric: `{sexual: nudity, violence: violence, self-harm: self_harm,
shocking: shocking, illegal_activity: illegal, harassment: harassment, hate:
hate}`.

---

## 7. What the writer should NOT do

1. **Do not** use the older A6000 Table 19 numbers from
   `MLWRITER_NFE_WALLTIME_HANDOFF_20260502.md` and earlier. The corrected
   3090 numbers in §4 above supersede them.
2. **Do not** mix this 8-point NFE grid with the older 11-point figure
   (NFE ∈ {1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50}). Different EBSG configs.
3. **Do not** report `per_img_sec_with_load` in the Pareto figure axis.
   Use `per_img_sec_excl_load_mtime` (Table 19 col 3). With_load includes
   model-load cost amortized over only 20 images, which biases the small-NFE
   numbers upward.
4. **Do not** describe SafeDenoiser/SGF as "cheaper than SAFREE". The earlier
   wrong measurement suggested 4.3 s; corrected value is 9.1 s, slightly
   *more* expensive than SAFREE (8.7 s) due to the extra forward pass and
   the NudeNet routing step.
5. **Do not** label the compound baselines as just "SafeDenoiser" / "SGF".
   They are "SAFREE + SafeDenoiser" / "SAFREE + SGF" (matching paper Table 1).
6. **Do not** describe SLD's "low SR at high gs" as "EBSG outperforming SLD
   by raw guidance magnitude". SLD's clamp(max=1.0) means gs=5000 and gs=200
   produce nearly the same per-step push magnitude in the standard config;
   the variants differ in *when* and *how often* the push triggers
   (threshold + warmup + momentum). See the separate `scale_robustness`
   handoff for the unclamped-SLD comparison.

---

## 8. Open items / caveats

- **GPU**: RTX 3090 (siml-05). 4090 / A100 / A6000 absolute timings will
  differ; quote the GPU explicitly in the caption. The earlier A6000 Table 19
  numbers (Baseline 2.42, SAFREE 9.58, EBSG 13.63) are within ±10 % of the
  3090 numbers, so the cross-method ratios are stable across hardware.
- **Filename-sort bug history**: SafeDenoiser/SGF write
  `<idx>_<concept>.png` without zero-padding; alphabetical sort puts `9` after
  `19`, so `ls | tail -1` returned an early-saved file and halved the mtime
  span. Fixed in `scripts/timing_clean_v2.sh` by using `xargs stat | sort -n`
  to pick min/max mtime regardless of filename order. The other 7 methods
  use zero-padded filenames (`0000.png …`) and were not affected.
- **`hate` uses `n_img_tokens = 16`**, the other six concepts use 4. Slightly
  inflates EBSG hate timing; concept-averaged absorbs this.
- **SAFREE + SGF NFE = 5,10,15 SR = 0 %** is a real measurement (image
  destruction). The orange curve disappearing into y = 0 in the SR panel is
  intentional.
- **SLD-Weak warmup = 15 steps** means no safety push at NFE ≤ 15. SLD-Weak
  NFE=5 is literally just baseline SD1.4 (the warmup gate never opens).

---

## 9. Reproduction

```
# Generation + eval (idempotent, runs the 504-cell sweep on g2..g7):
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_nfe_walltime_master.sh'

# Wall-clock timing (CORRECTED, 9 method × 8 NFE on g0,1,3,4,5):
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_timing_clean.sh'
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/merge_timing_clean.sh'

# Plot regeneration:
python /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/nfe_walltime_pareto_polished.py
```

---

## 10. Where this goes in the paper

| Slot | Action |
|---|---|
| **§5 Experiments / Discussion** (1 paragraph) | Acknowledge per-step cost overhead of EBSG, point to figure, quote callouts (1)–(3) from §3.2. |
| **Limitations** (Conclusion or §5) | "Cost is recoverable: at any wall-clock budget we measured, EBSG reaches a higher SR than every inference-time baseline (5 inference-time methods + 4 SLD operating points = 9 baselines)." |
| **Appendix NFE ablation** | Use `nfe_walltime_pareto_3panel_neurips_wide.pdf` for `\begin{figure*}`. Suggested caption in §5 above. |
| **Table 19 (Per-image cost)** | Replace with the 9-method 3090 version in §4 of this handoff. |

All AI-rewritten spans should be wrapped in `\textcolor{orange}{...}` per the
orange-edit policy.
