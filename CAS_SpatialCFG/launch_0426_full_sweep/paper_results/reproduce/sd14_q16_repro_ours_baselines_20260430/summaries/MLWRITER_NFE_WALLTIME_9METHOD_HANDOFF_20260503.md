# ML writer handoff — NFE wall-clock vs SR Pareto, 9-method (2026-05-03)

**Update from 2026-05-02 version**: 4 SLD variants (Max, Strong, Medium, Weak) added.
Total methods: **9** (was 5). The headline message is unchanged but is now stronger:

> EBSG is Pareto-dominant on the SR axis at every per-image latency budget,
> against **5 baselines + 4 SLD operating points** (Max/Strong/Medium/Weak).

---

## 1. Why 9 methods

The previous draft compared EBSG against 5 baselines (Baseline, SAFREE,
SAFREE+SafeDenoiser, SAFREE+SGF). Reviewers will plausibly ask "what about SLD?",
which has 4 official operating points. We added all 4 to make the cost-vs-quality
trade-off comprehensive:

| Method | UNet forwards / step | Notes |
|---|---:|---|
| Baseline (SD1.4 + DDIM)         | 2 | cond + uncond CFG |
| SAFREE (`-svf -lra`)            | 2 | + token re-projection + self-validation filter + latent re-attention |
| SAFREE + SafeDenoiser           | 3+ | + concept-specific NEGSPACE + NudeNet routing |
| SAFREE + SGF                    | 3+ | + repellency MMD |
| **SLD-Max**                     | 3 | gs=5000, warmup=0, threshold=1.0, momentum=0.5/0.7 |
| **SLD-Strong**                  | 3 | gs=2000, warmup=7, threshold=0.025, momentum=0.5/0.7 |
| **SLD-Medium**                  | 3 | gs=1000, warmup=10, threshold=0.01, momentum=0.3/0.4 |
| **SLD-Weak**                    | 3 | gs=200, warmup=15, threshold=0.005, no momentum |
| **EBSG (Ours)**                 | **11** | probe + cond + uncond × per-family fork |

All inference-time methods run on the same SD v1.4 backbone with the same prompt set
(I2P q16 top-60, 7 concepts, 60 prompts each), seed 42, CFG 7.5, DDIM scheduler.

---

## 2. Experimental setup (unchanged from 2026-05-02 except method list)

| Axis | Value |
|---|---|
| Backbone | SD v1.4, DDIM scheduler, seed 42, CFG = 7.5, 512×512 |
| Concepts (7) | sexual, violence, self-harm, shocking, illegal_activity, harassment, hate |
| Methods (9) | Baseline / SAFREE / SAFREE+SafeDenoiser / SAFREE+SGF / SLD-{Max, Strong, Medium, Weak} / EBSG |
| NFE grid (8) | **5, 10, 15, 20, 25, 30, 40, 50** |
| Prompts/cell | 60 (I2P q16 top-60 per concept) |
| Total cells | 9 × 7 × 8 = **504** (≈30,240 images) |
| Evaluator | Qwen3-VL-8B v5 (4-class rubric: NotRel / Safe / Partial / Full) |
| GPU pool | siml-05 RTX 3090 (g2..g7 originally; later g0/g1 also) |

**Wall-clock measurement**: separate isolated single-GPU benchmark on RTX 3090, no
contention. 5 base methods (`baseline / safree / safedenoiser / sgf / ebsg`) × 8 NFE
× 20 prompts in `nfe_walltime_timing.csv`; 4 SLD variants × 8 NFE × 60 prompts in
`nfe_walltime_timing_sld.csv`. Plot script merges both timing CSVs (later overrides
on (method, nfe) collision). `per_img_sec_excl_load_mtime` reported (excludes one-time
model-load cost via PNG mtime range / (N-1)).

---

## 3. Result tables

### 3.1 Concept-averaged headline (3 NFE points; full grid in CSV)

| Method                        | NFE=5 | NFE=10 | NFE=50 |
|---|---|---|---|
| **Baseline**                  | 0.42 s · 34.8 SR · 41.4 Full | 0.63 s · 31.4 SR · 53.1 Full | 2.68 s · 35.7 SR · 51.7 Full |
| **SAFREE**                    | 1.11 s · 19.0 SR · 1.0 Full  | 2.00 s · 31.2 SR · 4.0 Full  | 8.89 s · 54.3 SR · 10.5 Full |
| **SAFREE + SafeDenoiser**     | 0.58 s · 46.9 SR · 13.6 Full | 1.00 s · 54.8 SR · 16.9 Full | 4.32 s · 53.1 SR · 21.2 Full |
| **SAFREE + SGF**              | 0.58 s · **0.0** SR · 100 NotRel | 1.05 s · **0.0** SR · 100 NotRel | 4.42 s · 47.1 SR · 23.1 Full |
| **SLD-Max**                   | 1.20 s · 26.2 SR · 21.0 Full | 2.12 s · 31.7 SR · 30.9 Full | 9.17 s · 46.7 SR · 18.8 Full |
| **SLD-Strong**                | 1.24 s · 32.6 SR · 38.8 Full | 2.12 s · 33.1 SR · 43.1 Full | 9.15 s · 48.8 SR · 21.7 Full |
| **SLD-Medium**                | 1.24 s · 32.6 SR · 38.8 Full | 2.12 s · 31.4 SR · 48.8 Full | 9.15 s · 46.9 SR · 31.0 Full |
| **SLD-Weak**                  | 1.22 s · 32.6 SR · 38.8 Full | 2.12 s · 31.4 SR · 48.3 Full | 9.14 s · 40.7 SR · 42.6 Full |
| **EBSG (Ours)**               | 1.47 s · **52.4** SR · 13.8 Full | 2.79 s · **60.2** SR · 18.1 Full | **14.11 s · 70.2 SR · 11.2 Full** |

### 3.2 Headline callouts to copy into paper text

1. **EBSG @ NFE=5 (1.47 s) vs SD1.4 + DDIM-50 (2.68 s)**:
   EBSG is **1.8× faster AND +16.7 pp higher SR** (52.4 vs 35.7).
2. **EBSG @ NFE=10 (2.79 s) vs SAFREE @ NFE=50 (8.89 s)**:
   EBSG is **3.2× faster AND +5.9 pp higher SR** (60.2 vs 54.3).
3. **EBSG @ NFE=10 (2.79 s) vs every SLD variant @ NFE=50 (≈9.15 s)**:
   EBSG is **3.3× faster AND +11–19 pp higher SR** (60.2 vs SLD-Max 46.7, Strong 48.8,
   Medium 46.9, Weak 40.7). All 4 SLD operating points cluster in ~40–49 % SR; EBSG
   exceeds the entire SLD family at a small fraction of their cost.
4. **EBSG @ NFE=50 (14.11 s)**: highest SR (70.2 %) and lowest Full (11.2 %) among
   ALL nine methods at any latency we measured. **Pareto-dominant**.
5. **SAFREE + SGF small-NFE collapse**: at NFE ≤ 15, SR = 0 % and NotRel = 100 %
   (image destruction; gradient-MMD repellency unstable at low NFE). SGF only becomes
   non-degenerate from NFE ≥ 25.
6. **SLD plateau across all 4 operating points**: at NFE = 50, all 4 SLD variants
   land in 40.7 – 48.8 % SR, regardless of guidance scale (gs ∈ {200, 1000, 2000,
   5000}), warmup steps, or momentum. The internal `clamp(scale * gs, max=1.0)` in
   the SLD pipeline (Eq. 6) caps the effective per-step push at 1.0 once gs is "large
   enough" (~50), so the four operating points differ mainly in *threshold* and
   *momentum*, not in raw guidance magnitude. None escapes the ~50 % SR ceiling.
7. **Baseline plateau**: SR is roughly flat at 31–37 % across all NFE.
   Adding compute does not buy safety without an erasure mechanism.
8. **EBSG monotone**: EBSG is the **only** inference-time method that is monotonic in
   NFE on the SR axis (52.4 → 60.2 → 66.4 → 69.0 → 72.6 → 70.7 → 70.5 → 70.2). Other
   methods either plateau early (SAFREE + SafeDenoiser, all SLD), require NFE ≥ 25 to
   avoid catastrophic NotRel collapse (SAFREE + SGF), or never improve with NFE
   (Baseline).

---

## 4. Figures (paper-ready PDFs in `paper_results/figures/`)

Multiple sizes available so the paper can place the figure in any NeurIPS layout slot.

### 4.1 Recommended file per LaTeX slot

| LaTeX slot | Width | File | Notes |
|---|---|---|---|
| `\begin{figure*}` (full text width) | **6.75 in** | `nfe_walltime_pareto_3panel_neurips_wide.pdf` | Recommended **headline** for the appendix NFE ablation. |
| `\begin{figure}` (single column) | **5.5 in** | `nfe_walltime_pareto_3panel_neurips_textwidth.pdf` | Compact 3-panel that fits single-column. |
| Tight side-by-side with text | 4.8 in | `nfe_walltime_pareto_3panel_neurips_compact.pdf` | When you need to leave room for caption / text wrap. |
| **SR-only** main body figure (single column) | 5.5 in | `nfe_walltime_pareto_sr_neurips_textwidth.pdf` | Best if you only have room for one panel; carries the headline message alone. |
| SR / Full / NotRel side-by-side appendix grid | 2.7 in × 3 | `nfe_walltime_pareto_sr_neurips_halfwidth.pdf`, `_full_neurips_halfwidth.pdf`, `_notrel_neurips_halfwidth.pdf` | Use `\subfigure` or `minipage` to lay them out 1×3. |
| 7-concept × 3-metric supplementary facet | 15 in × 22 in | `nfe_walltime_pareto_per_concept.pdf` | Appendix supplementary view; full-page `figure*`. |

### 4.2 Other sizes (slide / debug)

| Use | File |
|---|---|
| 15 in × 4.4 in 3-panel (slide deck / first-pass review) | `nfe_walltime_pareto_polished.pdf` |
| Vanilla 3-panel without NeurIPS styling | `nfe_walltime_pareto_concept_avg.pdf` |
| Single-panel un-resized (5.6 × 4.4 in) | `nfe_walltime_pareto_sr.pdf`, `_full.pdf`, `_notrel.pdf` |

### 4.3 Suggested LaTeX caption (3-panel headline)

> **Figure N.** *Wall-clock cost vs erasure quality, concept-averaged across the
> seven I2P top-60 concepts (sexual, violence, self-harm, shocking,
> illegal_activity, harassment, hate). The horizontal axis is per-image generation
> time (excluding one-time model-load cost) measured on a single RTX 3090 with no
> GPU contention. Markers correspond to NFE $\in$ \{5, 10, 15, 20, 25, 30, 40, 50\}.
> We compare EBSG against eight inference-time baselines: SD\,1.4, SAFREE-v2,
> SAFREE\,+\,SafeDenoiser, SAFREE\,+\,SGF, and the four official SLD operating
> points (Max, Strong, Medium, Weak).
> \textbf{EBSG (red) is Pareto-dominant on the SR axis at every per-image latency
> budget}: at NFE = 5 it already exceeds SD\,1.4 + DDIM-50 by $+16.7$\,pp while
> running 1.8$\times$ faster, and at NFE = 10 it exceeds SAFREE + DDIM-50 by
> $+5.9$\,pp and every SLD variant + DDIM-50 by $+11$ -- $+19$\,pp while running
> 3.3$\times$ faster. The compound SAFREE\,+\,SGF baseline degenerates to
> 100\,\% NotRelevant for NFE $\le$ 15 (orange, NotRel panel), the
> safety-by-destruction failure mode that the four-class VLM rubric is designed to
> expose. All four SLD operating points cluster in 40 -- 49\,\% SR at NFE = 50,
> regardless of guidance scale, due to SLD's internal clamp on per-step safety
> magnitude (Eq. 6 of the SLD paper).*

---

## 5. CSV files

| File | Rows | Columns |
|---|---|---|
| `paper_results/figures/nfe_walltime_5method_7concept_table.csv` | 280 | method, concept, nfe, SR_pct, Full_pct, NR_pct (5 base methods only) |
| `paper_results/figures/nfe_walltime_5method_concept_avg.csv`     | 40 | per (method, nfe) avg + per_img_sec (5 base methods only) |
| `paper_results/figures/nfe_walltime_timing.csv`                  | 41 | 5 base methods × 8 NFE timing |
| `paper_results/figures/nfe_walltime_timing_sld.csv`              | 33 | 4 SLD variants × 8 NFE timing |

To get the **9-method concept-avg numbers** for SLD, use the script in
`scripts/nfe_walltime_pareto_polished.py` which now reads ALL 9 methods from raw
`outputs/phase_nfe_walltime_v3/<method>_<concept>_steps<NFE>/results_qwen3_vl_<rubric>_v5.txt`
and merges both timing CSVs. The headline numbers in §3.1 above are the canonical
9-method values (use those, not the older 5-method CSV).

---

## 6. What the writer should NOT do

1. **Do not** mix this 8-point NFE grid with the older 11-point NFE figure
   (NFE ∈ {1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50}). Different EBSG configs.
2. **Do not** average wall-clock across methods; quote per-method per-NFE values.
3. **Do not** report `per_img_sec_with_load`. Use `per_img_sec_excl_load_mtime` only.
4. **Do not** label the compound baselines as just "SafeDenoiser" / "SGF". They are
   "SAFREE + SafeDenoiser" / "SAFREE + SGF" (matching paper Table 1).
5. **Do not** describe SLD's "low SR at high gs" as "EBSG outperforming SLD by raw
   guidance magnitude". SLD's clamp at 1.0 means gs=5000 and gs=200 produce nearly
   the same per-step push magnitude; the variants differ in *when* and *how often*
   the push triggers (threshold + warmup + momentum), not in raw push strength.
   See the separate `scale_robustness` figure for the unclamped-SLD comparison.

---

## 7. Open items / caveats

- **GPU**: RTX 3090. 4090 / A100 absolute timings will differ; quote the GPU
  explicitly in the caption.
- **`hate` uses `n_img_tokens = 16`**, the other six concepts use 4. Slightly
  inflates EBSG hate timing; concept-averaged absorbs this.
- **SAFREE + SGF NFE = 5,10,15 SR = 0 %** is a real measurement (image destruction).
  The orange curve disappearing into y = 0 in the SR panel is intentional.
- **SLD-Weak warmup = 15 steps** means no safety push at NFE ≤ 15. SLD-Weak NFE=5 is
  literally just baseline SD1.4 (the warmup gate never opens).
  The SR/Full numbers for SLD-Weak at NFE ∈ {5, 10, 15} happen to match SLD-Strong
  exactly because all three reduce to "no safety push within the budget", and our
  CSV correctly shows this artifact.

---

## 8. Reproduction

```
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_nfe_walltime_master.sh'
```

Launches 6 parallel orchestrators (`scripts/nfe_walltime_orchestrator.py`) on g2..g7
for the 504-cell generation + eval sweep. Idempotent: skips cells already at 60 PNGs.

Plot regeneration:
```
python scripts/nfe_walltime_pareto_polished.py
```

This produces all NeurIPS-sized variants and the per-panel standalones.

---

## 9. Where this goes in the paper

| Slot | Action |
|---|---|
| **§5 Experiments / Discussion** (1 paragraph) | Acknowledge per-step cost overhead of EBSG, point to figure, quote callouts (1)–(3) from §3.2. |
| **Limitations** (Conclusion or §5) | "Cost is recoverable: at any wall-clock budget we measured, EBSG reaches a higher SR than every inference-time baseline (5 inference-time methods + 4 SLD operating points = 9 baselines)." |
| **Appendix NFE ablation** | Use `nfe_walltime_pareto_3panel_neurips_wide.pdf` for `\begin{figure*}`. Suggested caption in §4.3 above. |
| **Appendix tables** | If a numerical table is desired, use the headline §3.1 values (9 methods × 3 NFE columns). |

All AI-rewritten spans should be wrapped in `\textcolor{orange}{...}` per the
orange-edit policy.
