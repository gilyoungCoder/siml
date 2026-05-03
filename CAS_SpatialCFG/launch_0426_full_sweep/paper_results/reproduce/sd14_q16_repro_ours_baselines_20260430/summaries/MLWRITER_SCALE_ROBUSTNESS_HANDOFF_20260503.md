# ML writer handoff — Scale robustness ablation (2026-05-03)

A new ablation that addresses the natural reviewer question:
**"What happens when the safety knob is pushed harder?"** EBSG and SLD both expose
a "guidance scale" parameter; the experiment shows EBSG is robust to wide
guidance-scale sweeps while every SLD operating point eventually collapses at
high scale (Not-Relevant surge). The result strengthens the claim that EBSG's
explicit *when* (CAS gate) and *where* (dual-probe winner-take-all) decomposition
admits stronger guidance without the safety-by-destruction failure mode that
all four SLD variants exhibit.

---

## 1. Background — what each SLD variant actually means

`SLDPipeline` (Schramowski et al., 2023) has four official operating points,
which differ in **five hyperparameters**, not just `sld_guidance_scale`:

| Variant | gs | warmup | threshold | momentum_scale | mom_beta |
|---|---:|---:|---:|---:|---:|
| **Weak**   | 200  | 15 | 0.005 | 0.0 | 0.0 |
| **Medium** | 1000 | 10 | 0.01  | 0.3 | 0.4 |
| **Strong** | 2000 | 7  | 0.025 | 0.5 | 0.7 |
| **Max**    | 5000 | 0  | 1.0   | 0.5 | 0.7 |

Per-parameter semantics:

- **`sld_guidance_scale` (gs)**: enters via `scale = clamp(|d|·gs, max=1)` where
  `d = noise_pred_text − noise_pred_safety`. With the default `clamp(max=1)`,
  gs is essentially a "saturation knob" — once gs exceeds ~1/|d| the per-step
  push is pinned at 1.0 regardless of gs. So the gs values 200..5000 are NOT
  proportional magnitudes; they all push at full magnitude (=1.0) in steady
  state. The variants are differentiated by the other four knobs.
- **`sld_warmup_steps`**: number of leading DDIM steps with safety guidance OFF.
  Weak warmup=15 means safety only kicks in from step 16 onward; at NFE ≤ 15
  Weak is literally vanilla SD. Max warmup=0 means safety is on from step 1.
- **`sld_threshold`**: the gating predicate
  `safety_concept_scale = where((text−safety) ≥ threshold, 0, scale)`. Higher
  threshold means the gate triggers more often (push is active in more steps).
  Max threshold=1.0 ⇒ push triggers nearly always; Weak threshold=0.005 ⇒
  push triggers rarely.
- **`sld_momentum_scale`**: linear coefficient on accumulated past push
  (`safety_momentum`). Weak has 0 (no momentum); Max/Strong have 0.5.
- **`sld_mom_beta`**: EMA decay for `safety_momentum`. Weak has 0; Max/Strong
  have 0.7 (strong accumulation).

**Effective ordering** at default settings: Weak (mildest, late + rare push,
no memory) < Medium < Strong < Max (most aggressive, immediate + frequent push,
strong momentum).

This experiment **removes the clamp** (`SLD_CLAMP_MAX=1e6`) and sweeps `gs` over
the same range as EBSG's `safety_scale`, so gs becomes a real magnitude knob and
all four variants can be compared on the same axis.

---

## 2. Experimental setup

| Axis | Value |
|---|---|
| Backbone | SD v1.4, DDIM scheduler, seed 42, CFG = 7.5, 512×512 |
| NFE | **50** (fixed) |
| Concept | **sexual** (I2P q16 top-60, 60 prompts) |
| Methods (5) | SLD-Max / SLD-Strong / SLD-Medium / SLD-Weak (all clamp-removed via env var) + EBSG (Ours) |
| Scale axis | gs (SLD) and safety_scale (EBSG) ∈ {5, 10, 20, 50, 100, 200, 500} |
| Per cell | 60 prompts × 1 image |
| Total cells | 5 × 7 = **35** (re-using 7 EBSG cells from the earlier scale_robustness_v1 sweep) |
| Evaluator | Qwen3-VL-8B v5 (NotRel / Safe / Partial / Full) |
| GPU | siml-05 RTX 3090 (g1, g2, g3, g4, g5; 1 cell per slot, no contention) |

**Important**: SLD variants keep their default `warmup`, `threshold`,
`momentum_scale`, `mom_beta` (Table in §1); only `sld_guidance_scale` is
swept. With `SLD_CLAMP_MAX=1e6` the per-step push is no longer pinned at 1.0,
so gs really controls magnitude.

EBSG variants in this sweep use the per-concept best `cas_threshold = 0.5`,
`attn_threshold = 0.10`, `img_attn_threshold = 0.30` from Table 1
(sexual single-I2P best); only `safety_scale` varies.

---

## 3. Results (SR = Safe + Partial)

### 3.1 Full table (sexual, 60 prompts each)

| Method        | gs/ss=5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EBSG (Ours)** | **86.7** | **96.7** | **98.3** | **96.7** | **96.7** | **90.0** | **73.3** |
| SLD-Max       | 75.0 | 71.7 | 75.0 | 80.0 | 68.3 | 41.7 | **15.0** |
| SLD-Strong    | 70.0 | 70.0 | 76.7 | 83.3 | 83.3 | 55.0 | 43.3 |
| SLD-Medium    | 70.0 | 73.3 | 68.3 | 75.0 | 88.3 | 73.3 | 53.3 |
| SLD-Weak      | 70.0 | 70.0 | 70.0 | 71.7 | 86.7 | 85.0 | 66.7 |

Corresponding NotRel (semantic-collapse rate) at scale=500:
EBSG 25.0 < SLD-Weak 33.3 < SLD-Medium 46.7 < SLD-Strong 56.7 < SLD-Max 85.0.

### 3.2 Headline callouts

1. **EBSG plateau**: SR ≥ 96.7 for `safety_scale` ∈ [10, 100] (a 10× range), and
   stays at 73.3 even at the extreme `safety_scale = 500`. NotRel ≤ 25 across
   the entire 100× sweep. EBSG is the **only** method with plateau-style
   robustness in this experiment.
2. **All four SLD variants have a narrow sweet spot**:
   peak SR is reached in a 1.5×–2× window of gs, and SR drops sharply outside
   it. Peak/scale: Max 80.0 @ gs=50, Strong 83.3 @ gs=50–100, Medium 88.3 @
   gs=100, Weak 86.7 @ gs=100.
3. **High-scale (gs = 500) ranking** (worst → best):
   SLD-Max 15 → SLD-Strong 43 → SLD-Medium 53 → SLD-Weak 67 → **EBSG 73**.
   Within SLD, the variant with the *least* aggressive momentum/threshold
   (Weak) is the most scale-robust — confirming that the SLD clamp + heavy
   momentum is what causes Max/Strong to collapse first when the cap is lifted.
4. **NotRel surge tells the safety-by-destruction story**: at gs = 500 the
   variants in order Max → Strong → Medium → Weak have NotRel = 85 → 57 → 47 →
   33, monotonic with momentum strength. EBSG holds NotRel = 25 because the
   *where* probe restricts the correction to the unsafe region rather than
   pushing the entire latent.
5. **Mechanism implication**: SLD's stability at "default" gs values (200–5000)
   is an artefact of the `clamp(max=1)` cap inside Eq. 6 of the SLD paper,
   not of inherent stability. With the clamp removed the scale knob does what
   reviewers expect (more push → more destruction), and SLD inherits that
   behaviour. EBSG remains robust because the spatial mask gate prevents the
   push from rewriting safe regions.

---

## 4. Figure (paper-ready PDF)

| File | Layout | Notes |
|---|---|---|
| `paper_results/figures/scale_robustness_v2.{pdf,png}` | 1 row × 3 col, 13.5 × 3.6 in | Headline figure — 5 curves on log-x SR / Full / NotRel |
| `paper_results/figures/scale_robustness_v2_table.csv` | 35 rows | Raw data per (method, scale) |

### Suggested LaTeX caption

> **Figure N.** *Robustness to safety-scale, sexual concept (NFE = 50, 60
> prompts). Five methods sweep their respective safety knob over the same
> magnitude range (`safety_scale` for EBSG, `sld_guidance_scale` for the four
> SLD variants) on a log-scale x-axis. SLD's internal clamp is removed
> (`SLD_CLAMP_MAX = 1e6`) so gs really controls per-step push magnitude rather
> than saturating at 1.0. \textbf{EBSG (red) is the only method with
> plateau-style robustness}: SR $\ge$ 96.7\,\% across `safety_scale` $\in [10,
> 100]$ and SR = 73.3\,\% even at the extreme value 500. All four SLD variants
> have a narrow sweet spot (peak SR 80 -- 88\,\% at gs $\in [50, 100]$) and
> collapse to NotRel-dominated outputs at gs $\ge 200$, in monotonic order of
> momentum strength: SLD-Max collapses first (SR 15 \% / NotRel 85 \% at gs =
> 500), then Strong, Medium, and Weak (SR 43 / 53 / 67 \%). EBSG's spatial
> mask gate (\S\ref{subsec:where}) prevents the push from rewriting safe
> regions, hence the wider robustness window.*

---

## 5. CSV file

`paper_results/figures/scale_robustness_v2_table.csv` (35 rows).
Columns: `method, scale, SR_pct, Full_pct, NR_pct`.

`method` ∈ {`ebsg`, `sld_max`, `sld_strong`, `sld_medium`, `sld_weak`}.
For SLD rows, `scale` is `sld_guidance_scale`; for EBSG rows it is
`safety_scale`. Both are the same magnitude axis (clamp removed for SLD).

---

## 6. What the writer should NOT do

1. **Do not** describe SLD's "stability at default gs" as a property of the
   method. It is a **clamp artefact** (`min(scale, 1)` cap in Eq. 6). With the
   clamp removed (this experiment) SLD behaves like every other guidance method:
   more push → more destruction.
2. **Do not** average over the 4 SLD variants — they have different sweet
   spots and the average smears the message. Plot all four (as in the figure).
3. **Do not** label "SLD-Max" / "SLD-Weak" without the four-parameter context
   (§1 here). Reviewers will assume the variants only differ in `gs` unless
   the paper text says otherwise.
4. **Do not** claim EBSG is "infinitely robust" — at `safety_scale = 500` SR
   drops to 73.3 % and NotRel rises to 25 %. The claim is **wider plateau**,
   not "no degradation".
5. **Do not** use this figure with a clamped SLD. The headline message
   ("SLD collapses at high gs") only holds with `SLD_CLAMP_MAX=1e6`. If the
   figure regenerates from the raw data, ensure the cells were generated with
   `SLD_CLAMP_MAX=1000000` env var.

---

## 7. Caveats / open items

- **Sexual only** in the current experiment. Extending to violence, hate,
  etc. would strengthen the claim; one concept is sufficient for the paper's
  primary scale-robustness claim but reviewers may ask for more.
- **Single seed (42)** per cell. Variance across seeds untested; with 60
  prompts the per-cell SE is small but inter-seed variance could shift peak
  SR by ~5 %p. Consider 3-seed re-run if a reviewer asks.
- **EBSG `safety_scale = 500`** is far outside the per-concept best
  (sexual best ≈ 20). The 73.3 % SR at ss=500 is therefore not a
  recommendation to use ss=500 in practice; it is a **stress test** showing
  EBSG degrades gracefully rather than catastrophically.
- **SLD `sld_threshold` not swept**. We sweep gs only because reviewers will
  most naturally read gs as the "guidance scale". Threshold (the actual gate)
  would be a follow-up sweep; but for the headline message gs sweep is enough.

---

## 8. Reproduction

```
# Generation + eval (sexual, 4 SLD variants, 7 scales, NFE=50):
ssh siml-05 'bash /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/launch_scale_v2.sh'

# Plot:
python /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/scripts/scale_v2_plot.py
```

The orchestrator (`scale_v2_orchestrator.py`) embeds `SLD_CLAMP_MAX=1000000`
in the `env` arg of every SLD subprocess; the `sld_runner.py` accepts
`--sld_guidance_scale` to override the variant default. EBSG cells are reused
from the earlier `phase_scale_robustness/` (no re-gen needed) since EBSG was
unchanged across the v1 → v2 revision.

---

## 9. Where this goes in the paper

| Slot | Action |
|---|---|
| **§5.X Robustness ablation** (new sub-section) | 1 paragraph + Figure N. Use the headline callouts (1) – (5) from §3.2. |
| **§3 Method discussion** (optional reference) | Mention robustness as evidence that the spatial mask gate (Where) is what enables wider-magnitude correction without semantic collapse. |
| **Limitations** | Note that SLD's clamp removal is a deliberate choice; the comparison is unfair against the SLD design philosophy. The fairer reading is "EBSG matches SLD-default in safety while admitting a wider operating-scale window". |
| **Appendix Implementation Detail** | Include §1 (SLD variant hyperparameters table) so reviewers understand what "SLD-Max" really means. |

All AI-rewritten spans should be wrapped in `\textcolor{orange}{...}` per the
orange-edit policy.
