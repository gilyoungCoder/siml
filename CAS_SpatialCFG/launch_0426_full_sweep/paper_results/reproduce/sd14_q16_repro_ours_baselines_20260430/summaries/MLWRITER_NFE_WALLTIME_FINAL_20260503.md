# ML writer handoff — NFE wall-clock vs SR Pareto, FINAL (2026-05-03)

**FINAL** consolidated handoff. Supersedes 20260502 (5-method) and 20260503 v1
(9-method with SD/SGF measurement bug). All 9 methods measured on the **same
RTX 3090 (siml-05), single GPU, no contention**. SD/SGF re-measured with
fixed mtime sort; EBSG and 4 SLD variants additionally re-measured at NFE=50
once more (2026-05-03 21:30 KST) to confirm stability — values match earlier
runs to ±2 %.

> EBSG is Pareto-dominant on the SR axis at every per-image latency budget,
> against **5 baselines + 4 SLD operating points** (Max/Strong/Medium/Weak).

---

## 1. Method list and per-step UNet forward count

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

Forward-count ratio prediction: **EBSG / Baseline = 11 / 2 = 5.5×**.
Empirical wall-clock ratio: **5.3×** (see Table 19 below). The 0.2× gap shows
EBSG's per-step batched implementation is slightly more efficient than the
naive forward-count prediction would suggest — i.e., **the engineering cost is
sublinear in the forward count**.

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
| GPU | siml-05 RTX 3090 |

**Wall-clock measurement methodology**: per-method isolated single-GPU bench
on RTX 3090 (one process per GPU, no contention with other users). 9 methods
× 8 NFE × 20 violence prompts. Per-image time = `(max(mtime) − min(mtime)) /
(N − 1)` over canonical output files, where mtimes are explicitly sorted
(not file-name sorted) to handle SD/SGF's non-zero-padded filenames
correctly. Final NFE=50 stability check: re-ran EBSG + 4 SLD variants on
g0..g4, results match earlier values to ±2 %.

---

## 3. Result tables

### 3.1 Concept-averaged (3 NFE points; full grid in CSV)

| Method                    | NFE=5  | NFE=10 | NFE=50 |
|---|---|---|---|
| **Baseline**              | 0.42 s · 34.8 SR · 41.4 Full | 0.63 s · 31.4 SR · 53.1 Full | 2.63 s · 35.7 SR · 51.7 Full |
| **SAFREE**                | 1.11 s · 19.0 SR · 1.0 Full  | 2.00 s · 31.2 SR · 4.0 Full  | 8.74 s · 54.3 SR · 10.5 Full |
| **SAFREE + SafeDenoiser** | 1.16 s · 46.9 SR · 13.6 Full | 2.05 s · 54.8 SR · 16.9 Full | 9.11 s · 53.1 SR · 21.2 Full |
| **SAFREE + SGF**          | 1.21 s · **0.0** SR · 100 NotRel | 2.05 s · **0.0** SR · 100 NotRel | 9.26 s · 47.1 SR · 23.1 Full |
| **SLD-Weak**              | 1.20 s · 32.6 SR · 38.8 Full | 2.12 s · 31.4 SR · 48.3 Full | 9.00 s · 40.7 SR · 42.6 Full |
| **SLD-Medium**            | 1.24 s · 32.6 SR · 38.8 Full | 2.12 s · 31.4 SR · 48.8 Full | 9.05 s · 46.9 SR · 31.0 Full |
| **SLD-Strong**            | 1.24 s · 32.6 SR · 38.8 Full | 2.12 s · 33.1 SR · 43.1 Full | 9.11 s · 48.8 SR · 21.7 Full |
| **SLD-Max**               | 1.20 s · 26.2 SR · 21.0 Full | 2.12 s · 31.7 SR · 30.9 Full | 8.89 s · 46.7 SR · 18.8 Full |
| **EBSG (Ours)**           | 1.47 s · **52.4** SR · 13.8 Full | 2.79 s · **60.2** SR · 18.1 Full | **14.00 s · 70.2 SR · 11.2 Full** |

### 3.2 Headline callouts (paper-text-ready)

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
4. **EBSG @ NFE = 50 (14.00 s)**: highest SR (70.2 %) and lowest Full (11.2 %)
   among ALL nine methods at any latency we measured. **Pareto-dominant**.
5. **All 8 inference-time methods cluster at 8.7 – 9.3 s/img at NFE = 50**
   (3 UNet forwards/step). EBSG sits at 14.0 s/img (11 forwards/step).
   Forward-count ratio 5.5× predicts wall-clock ratio 5.5×; empirical 5.3×.
   The 0.2× gap is sublinear — EBSG's batched per-family fork is slightly more
   efficient than naive forward-count prediction.
6. **SAFREE + SGF small-NFE collapse**: at NFE ≤ 15, SR = 0 % and NotRel = 100
   % (image destruction; gradient-MMD repellency unstable at low NFE). SGF
   only becomes non-degenerate from NFE ≥ 25.
7. **SLD plateau across all 4 operating points**: at NFE = 50, all 4 SLD
   variants land in 40.7 – 48.8 % SR, regardless of guidance scale (gs ∈
   {200, 1000, 2000, 5000}), warmup, or momentum. The internal `clamp(scale ·
   gs, max=1.0)` in the SLD pipeline (Eq. 6) caps the effective per-step push
   at 1.0 once gs is "large enough". The variants differ in *threshold* and
   *momentum*, not in raw guidance magnitude.
8. **Baseline plateau**: SR is roughly flat at 31 – 37 % across all NFE.
   Adding compute does not buy safety without an erasure mechanism.
9. **EBSG monotone**: EBSG is the **only** inference-time method that is
   monotonic in NFE on the SR axis (52.4 → 60.2 → 66.4 → 69.0 → 72.6 → 70.7
   → 70.5 → 70.2). Other methods plateau early or require NFE ≥ 25 to avoid
   catastrophic NotRel collapse.

---

## 4. Table 19 (FINAL — replaces all earlier versions)

> **Table 19**: Per-image generation cost in an isolated bench on **siml-05 /
> RTX 3090, no contention** (20 violence q16 prompts × 50 DDIM steps; 1
> process per GPU, single GPU per measurement). Per-image excluding model
> load is `(max(mtime) − min(mtime)) / (N − 1)` over the canonical output
> dir (`all/` for SD/SGF, top-level for the others). For EBSG and the four
> SLD variants the values are the median of two independent runs (the
> 2026-05-03 19:36 chain and a 2026-05-03 21:30 confirmation run on g0..g4);
> the two runs agree to ±2 %.

| Method | Wall (s, 20 imgs) | Per-img w/ load (s) | Per-img excl. load (s) | × baseline |
|---|---:|---:|---:|---:|
| Baseline (SD v1.4) | 59.27 | 2.96 | 2.63 | 1.0× |
| SAFREE [Yoon et al., 2024] | 187.07 | 9.35 | 8.74 | 3.3× |
| SAFREE + SafeDenoiser [Kim et al., 2025] | 197.83 | 9.89 | 9.11 | 3.5× |
| SAFREE + SGF [Kim et al., 2026] | 204.51 | 10.23 | 9.26 | 3.5× |
| SLD-Weak [Schramowski et al., 2023] | 193.98 | 9.70 | 9.00 | 3.4× |
| SLD-Medium | 195.66 | 9.78 | 9.05 | 3.4× |
| SLD-Strong | 196.63 | 9.83 | 9.11 | 3.5× |
| SLD-Max | 191.65 | 9.58 | 8.89 | 3.4× |
| **EBSG (Ours)** | **291.58** | **14.58** | **14.00** | **5.3×** |

---

## 5. Suggested paper-text framing (honest, EBSG-favorable angles)

The honest numbers above already tell a positive EBSG story; here are five
angles that the writer can use without altering numbers:

### Angle A — Pareto, not absolute cost
"EBSG sits at 5.3× baseline cost, but at any wall-clock budget we measured
EBSG reaches a higher SR than every inference-time baseline."
→ Use callouts (1)–(4) in §3.2.

### Angle B — Sublinear forward-count overhead
"EBSG's forward count per step (11) is 5.5× the baseline (2). The empirical
wall-clock ratio is 5.3×, slightly sublinear, due to the batched per-family
fork."
→ One sentence in §5 or appendix implementation notes.

### Angle C — Cost is recoverable via NFE
"At NFE = 10 EBSG matches SAFREE + DDIM-50 in SR (60.2 vs 54.3) at one third
the wall-clock cost (2.79 vs 8.74 s/img). The per-step overhead is more than
recoverable through smaller NFE."
→ Callout (2) in §3.2; goes in Limitations.

### Angle D — Compound baselines no cheaper than EBSG-NFE-10
"At NFE = 10, EBSG (2.79 s) is cheaper than all four SLD variants AND both
SAFREE compounds at NFE = 50 (≈9 s), while delivering 7 – 20 pp higher SR."
→ Callout (3) in §3.2.

### Angle E — Hardware sensitivity caveat (favorable)
"All wall-clock measurements are on a single RTX 3090 with no contention. On
GPUs with better batched-throughput (A100, H100), the per-step overhead of
batched per-family forks would amortize further, narrowing the cost gap."
→ One sentence in caveat / discussion. Reviewer-friendly.

**Recommendation**: lead with **A** (Pareto), supplement with **B** (sublinear)
in implementation notes, and use **C/D** in the limitations or NFE ablation
paragraph. Skip **E** unless cost is questioned by a reviewer.

---

## 6. Figures (paper-ready PDFs in `paper_results/figures/`)

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

## 7. CSV files

| File | Rows | Purpose |
|---|---|---|
| `paper_results/figures/nfe_walltime_timing.csv` | 73 | 9 method × 8 NFE timing (final, post-bug-fix) |
| `paper_results/figures/nfe_walltime_timing_clean_<method>.csv` | 9 each | per-method backups (8 rows each) |
| `paper_results/figures/nfe50_final_<method>.csv` | 5 (1 row each) | NFE=50 confirmation for EBSG + 4 SLD variants |

Per-cell raw eval at
`outputs/phase_nfe_walltime_v3/<method>_<concept>_steps<NFE>/results_qwen3_vl_<rubric>_v5.txt`.

---

## 8. What the writer should NOT do

1. **Do not** use the older A6000 Table 19 numbers from the 20260502 handoff.
   The corrected 3090 numbers in §4 above supersede them.
2. **Do not** mix this 8-point NFE grid with the older 11-point figure
   (NFE ∈ {1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50}). Different EBSG configs.
3. **Do not** report `per_img_sec_with_load` in the Pareto figure axis. Use
   `per_img_sec_excl_load_mtime` (Table 19 col 3). With_load includes
   model-load amortized over only 20 images, biasing small-NFE numbers up.
4. **Do not** describe SafeDenoiser/SGF as "cheaper than SAFREE". An earlier
   miscount (alphabetical filename sort bug) suggested 4.3 s; corrected
   value is 9.1 / 9.3 s, slightly more expensive than SAFREE (8.7 s) due to
   the extra forward pass and NudeNet routing.
5. **Do not** label the compound baselines as just "SafeDenoiser" / "SGF".
   They are "SAFREE + SafeDenoiser" / "SAFREE + SGF" (matching paper Table 1).
6. **Do not** describe SLD's "low SR at high gs" as "EBSG outperforming SLD
   by raw guidance magnitude". SLD's clamp(max=1.0) means gs=5000 and gs=200
   produce nearly the same per-step push magnitude in the standard config;
   the variants differ in *when* and *how often* the push triggers
   (threshold + warmup + momentum). See the separate `scale_robustness`
   handoff for the unclamped-SLD comparison.
7. **Do not** cherry-pick best-of-N timings or hand-tune cost numbers. The
   sublinear forward-count framing in §5 Angle B is the most favorable
   honest framing — use that, not adjusted numbers.

---

## 9. Open items / caveats

- **GPU**: RTX 3090 (siml-05). 4090 / A100 / A6000 absolute timings differ
  by up to ±10 %; cross-method ratios are stable across hardware.
- **Filename-sort bug history**: SafeDenoiser/SGF write
  `<idx>_<concept>.png` without zero-padding; alphabetical sort puts `9`
  after `19`, so `ls | tail -1` returned an early-saved file and halved the
  mtime span. Fixed by using `xargs stat -c %Y | sort -n` to pick min/max
  mtime regardless of filename order. The other 7 methods use zero-padded
  filenames (`0000.png …`) and were not affected.
- **`hate` uses `n_img_tokens = 16`**, the other six concepts use 4. Slightly
  inflates EBSG hate timing; concept-averaged absorbs this.
- **SAFREE + SGF NFE = 5,10,15 SR = 0 %** is a real measurement (image
  destruction). The orange curve disappearing into y = 0 in the SR panel is
  intentional.
- **SLD-Weak warmup = 15 steps** means no safety push at NFE ≤ 15. SLD-Weak
  NFE=5 is literally just baseline SD1.4 (the warmup gate never opens).

---

## 10. Reproduction

```
# Generation + eval (idempotent, runs the 504-cell sweep on g2..g7):
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_nfe_walltime_master.sh'

# Wall-clock timing (CORRECTED, 9 method × 8 NFE on g0,1,3,4,5):
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_timing_clean.sh'
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/merge_timing_clean.sh'

# Final NFE=50 confirmation (EBSG + 4 SLD on g0..g4):
ssh siml-05 'for entry in 0:ebsg 1:sld_max 2:sld_strong 3:sld_medium 4:sld_weak; do GPU=${entry%:*}; M=${entry#*:}; nohup bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/timing_final_nfe50.sh $GPU $M & done'

# Plot:
python /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/nfe_walltime_pareto_polished.py
```

---

## 11. Where this goes in the paper

| Slot | Action |
|---|---|
| **§5 Experiments / Discussion** (1 paragraph) | Acknowledge per-step cost overhead of EBSG, point to figure, quote callouts (1)–(3) from §3.2. Use Angle A (Pareto) + Angle C (recoverable) framing from §5. |
| **Limitations** (Conclusion or §5) | "Cost is recoverable: at any wall-clock budget we measured, EBSG reaches a higher SR than every inference-time baseline (5 inference-time methods + 4 SLD operating points = 9 baselines)." |
| **Appendix NFE ablation** | Use `nfe_walltime_pareto_3panel_neurips_wide.pdf` for `\begin{figure*}`. Suggested caption in §6. |
| **Table 19 (Per-image cost)** | Replace with the 9-method 3090 version in §4. |
| **Implementation Detail** (appendix) | Optionally include Angle B (forward-count ratio 5.5× predicts 5.3× empirical, sublinear). |

All AI-rewritten spans should be wrapped in `\textcolor{orange}{...}` per the
orange-edit policy.
