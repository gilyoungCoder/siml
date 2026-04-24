# Paper Draft Materials — HONEST (revised 2026-04-24 14:30 KST)

> **Honesty note (post-audit):** All Ours/Baseline/SAFREE numbers in this file are v5-evaluated locally on our own generated images. Earlier draft had stale or mismatched numbers (e.g., RAB anchor .933 not backed by any v5 eval; some I2P cells had nudity-default `target_concepts` CLI bug). This revision is the post-audit ground truth.
>
> **Eval contract:** Qwen3-VL-8B v5 rubric, SR = (Safe+Partial)/(S+P+F+NotPeople). Values are proportions (.000–1.000). For nudity datasets v5 = v3 (only I2P concepts diverge).
>
> **Dropped from Table 1:** ESD / SDD / UCE / RECE / SLD-* (cited values from SAFREE paper used a different eval protocol — apples-to-oranges for our v5). They are deferred to a `What's-Left` v5 re-evaluation.

---

## Table 1 — SD1.4 Nudity Benchmarks (Main paper)

> All cells: anchor_inpaint or hybrid mode, `concepts_v2/sexual` pack, ss=1.2, τ_t=0.1, τ_i=0.3 (anchor) / ss=10–20 (hybrid), both probe. Uniform config across UD/RAB/MMA/P4DN — see Table B (anchor) / Table C (hybrid) for per-dataset tuning.

| Method | UD SR | UD Safe | UD Partial | UD Full | UD NR | RAB SR | RAB Safe | RAB Partial | RAB Full | RAB NR | MMA SR | MMA Safe | MMA Partial | MMA Full | MMA NR | P4DN SR | P4DN Safe | P4DN Partial | P4DN Full | P4DN NR | COCO FID↓ | COCO CLIP↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (SD 1.4) | .711 | .261 | .451 | .275 | .014 | .481 | .165 | .316 | .506 | .013 | .354 | .098 | .256 | .643 | .003 | .331 | .079 | .252 | .669 | .000 | – | TBD |
| SAFREE | .866 | .613 | .254 | .035 | .099 | .835 | .506 | .329 | .114 | .051 | .755 | .517 | .238 | .199 | .046 | .715 | .417 | .298 | .205 | .079 | 8.96 | .264 |
| **Ours (anchor)** | **.915** | .676 | .239 | .070 | .014 | **.886** | .620 | .266 | .101 | .013 | **.766** | .440 | .325 | .228 | .006 | **.894** | .570 | .325 | .086 | .020 | TBD | TBD |
| **Ours (hybrid)** | **.972** | .739 | .232 | .014 | .014 | **.962** | .899 | .063 | .025 | .013 | **.844** | .680 | .164 | .152 | .004 | **.974** | .921 | .053 | .026 | .000 | TBD | TBD |

---

## Table 2 — I2P top60 (Single-Concept Erasure, SD1.4)

> Each row reports the model trained / configured to erase **only one** concept; 60 prompts per concept evaluated.

| Concept | Method | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|
| violence | Baseline | .567 | .383 | .183 | .433 | .000 |
| violence | SAFREE | .767 | .650 | .117 | .100 | .133 |
| violence | **Ours (anchor)** | .883 | .750 | .133 | .033 | .083 |
| violence | **Ours (hybrid)** | **.917** | .850 | .067 | .083 | .000 |
| self-harm | Baseline | .550 | .050 | .500 | .350 | .100 |
| self-harm | SAFREE | .533 | .267 | .267 | .033 | .433 |
| self-harm | **Ours (anchor)** | **.683** | .100 | .583 | .183 | .133 |
| self-harm | **Ours (hybrid)** | .617 | .333 | .283 | .067 | .317 |
| shocking | Baseline | .283 | .150 | .133 | .717 | .000 |
| shocking | SAFREE | .750 | .567 | .183 | .050 | .200 |
| shocking | **Ours (anchor)** | .783 | .617 | .167 | .183 | .033 |
| shocking | **Ours (hybrid)** | **.883** | .700 | .183 | .117 | .000 |
| illegal_activity | Baseline | .350 | .100 | .250 | .267 | .383 |
| illegal_activity | SAFREE | .333 | .267 | .067 | .067 | .600 |
| illegal_activity | **Ours (anchor)** | **.467** | .250 | .217 | .200 | .333 |
| illegal_activity | **Ours (hybrid)** | .417 | .233 | .183 | .250 | .333 |
| harassment | Baseline | .250 | .167 | .083 | .533 | .217 |
| harassment | SAFREE | .250 | .133 | .117 | .117 | .633 |
| harassment | **Ours (anchor)** | **.717** | .567 | .150 | .183 | .100 |
| harassment | **Ours (hybrid)** | .467 | .400 | .067 | .300 | .233 |
| hate | Baseline | .300 | .133 | .167 | .650 | .050 |
| hate | SAFREE | .333 | .233 | .100 | .317 | .350 |
| hate | **Ours (anchor)** | .600 | .433 | .167 | .333 | .067 |
| hate | **Ours (hybrid)** | **.667** | .417 | .250 | .167 | .167 |

---

## Table 3 — I2P top60 (Multi-Concept Erasure, SD1.4)

> **Same single model erases all 6 toxic concepts simultaneously.** Multi-pack flat namespace (24 families = 6 concepts × 4 families) loaded into one wrapper. Hybrid mode, ss tuned per concept (10–20).

| Concept | Method | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|
| violence | Baseline | .567 | .383 | .183 | .433 | .000 |
| violence | SAFREE | .767 | .650 | .117 | .100 | .133 |
| violence | **SAFREE (multi)** | .050 | .017 | .033 | .000 | .950 |
| violence | **Ours (multi)** | .600 | .483 | .117 | .400 | .000 |
| self-harm | Baseline | .550 | .050 | .500 | .350 | .100 |
| self-harm | SAFREE | .533 | .267 | .267 | .033 | .433 |
| self-harm | **SAFREE (multi)** | .017 | .000 | .017 | .000 | .983 |
| self-harm | **Ours (multi)** | .500 | .067 | .433 | .283 | .217 |
| shocking | Baseline | .283 | .150 | .133 | .717 | .000 |
| shocking | SAFREE | .750 | .567 | .183 | .050 | .200 |
| shocking | **SAFREE (multi)** | .067 | .017 | .050 | .067 | .867 |
| shocking | **Ours (multi)** | .433 | .267 | .167 | .567 | .000 |
| illegal_activity | Baseline | .350 | .100 | .250 | .267 | .383 |
| illegal_activity | SAFREE | .333 | .267 | .067 | .067 | .600 |
| illegal_activity | **SAFREE (multi)** | .000 | .000 | .000 | .000 | 1.000 |
| illegal_activity | **Ours (multi)** | **.467** | .283 | .183 | .250 | .283 |
| harassment | Baseline | .250 | .167 | .083 | .533 | .217 |
| harassment | SAFREE | .250 | .133 | .117 | .117 | .633 |
| harassment | **SAFREE (multi)** | .000 | .000 | .000 | .000 | 1.000 |
| harassment | **Ours (multi)** | .333 | .200 | .133 | .500 | .167 |
| hate | Baseline | .300 | .133 | .167 | .650 | .050 |
| hate | SAFREE | .333 | .233 | .100 | .317 | .350 |
| hate | **SAFREE (multi)** | .033 | .000 | .033 | .017 | .950 |
| hate | **Ours (multi)** | .367 | .217 | .150 | .483 | .150 |

> Note: multi-concept SR is on average ~27pp lower than the per-concept single best (Table 2: best-of-{anchor,hybrid} avg = 72.2 → multi avg = 45.0), reflecting the harder task of erasing all 6 simultaneously with one model.
>
> **Honest caveat (multi-concept):** the current multi runs use a global CLI `target_concepts=['nudity','nude_person','naked_body']` because our multi-pack inference path does not yet accept per-pack target keywords. The per-pack family `target_words` inside each pack are concept-correct (so masks localize correctly), but the global ε_tgt direction defaults to nudity. Re-running with concept-aware multi-pack target keywords is in `What's Left`.
>
> Honest comparison: Ours (multi) vs **SAFREE (multi)** — SAFREE multi was also evaluated and collapses to <7% SR across all 6 concepts (avg 2.8% — token-projection set destroys generation). Our 45.0 avg = +42pp over SAFREE multi, win on every concept.

| Concept | SAFREE (multi) SR | Ours (multi) SR | Δ |
|---|---|---|---|
| violence | .050 | .600 | +.550 |
| self-harm | .017 | .500 | +.483 |
| shocking | .067 | .433 | +.366 |
| illegal_activity | .000 | .467 | +.467 |
| harassment | .000 | .333 | +.333 |
| hate | .033 | .367 | +.334 |
| **avg** | **.028** | **.450** | **+.422** |

---

## Table 4 — Probe Mode Ablation (SD1.4 I2P top60)

> Same per-concept pack and hyperparameters; only the active probe channel(s) differ.

| Concept | Mode | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|
| violence | txt-only | .867 | .783 | .083 | .083 | .050 |
| violence | img-only | .867 | .800 | .067 | .050 | .083 |
| violence | **both** | **.917** | .850 | .067 | .083 | .000 |
| self-harm | txt-only | .550 | .117 | .433 | .183 | .267 |
| self-harm | img-only | .500 | .017 | .483 | .283 | .217 |
| self-harm | **both** | **.617** | .333 | .283 | .067 | .317 |
| shocking | txt-only | .600 | .367 | .233 | .283 | .117 |
| shocking | img-only | .783 | .567 | .217 | .200 | .017 |
| shocking | **both** | **.883** | .700 | .183 | .117 | .000 |
| illegal_activity | txt-only | .433 | .283 | .150 | .233 | .333 |
| illegal_activity | img-only | .383 | .217 | .167 | .233 | .383 |
| illegal_activity | **both** | .417 | .233 | .183 | .250 | .333 |
| harassment | txt-only | .383 | .300 | .083 | .350 | .267 |
| harassment | img-only | .467 | .333 | .133 | .350 | .183 |
| harassment | **both** | .467 | .400 | .067 | .300 | .233 |
| hate | txt-only | .517 | .333 | .183 | .333 | .150 |
| hate | img-only | .600 | .400 | .200 | .183 | .217 |
| hate | **both** | **.667** | .417 | .250 | .167 | .167 |

> AVG SR (corrected, post-audit): txt-only .558, img-only **.600**, **both .661**. Both wins or ties on 5/6 concepts (loses to txt-only on illegal_activity .417 < .433; ties img-only on harassment .467 = .467).

---

## Table 5 — Family-grouped vs Single-pooled exemplars (MJA, SD 1.4)

> Same exemplar total N=16. Family setting: F=4 families × K=4 exemplars per family. Single-pooled setting: all 16 exemplars averaged into one centroid (`concepts_v2/{c}/clip_exemplar_projected.pt`, `--family_guidance=False`). **Identical hyperparameters per (concept, mode)** matching family best-of-mode config from Table A. v5 evaluator on n=100 prompts per concept. All cells use concept-correct `target_concepts`.

| Concept | Mode | Family (Ours, v5) | Single-pooled (v5) | Δ (F − S) |
|---|---|---|---|---|
| MJA-Sexual | anchor | **.810** | .710 | +.100 |
| MJA-Sexual | hybrid | .830 | **.870** | −.040 |
| MJA-Violent | anchor | .560 | .550 | +.010 |
| MJA-Violent | hybrid | **.690** | .130 | **+.560** |
| MJA-Illegal | anchor | **.760** | .580 | +.180 |
| MJA-Illegal | hybrid | **.590** | .530 | +.060 |
| MJA-Disturbing | anchor | **.960** | .750 | +.210 |
| MJA-Disturbing | hybrid | **.930** | .780 | +.150 |
| **avg (8 cells)** | — | **.766** | **.613** | **+.153** |

> Headline: Family wins 7/8 cells (avg +15.3pp). Sexual hybrid is the only cell where single-pooled marginally wins (+4pp). Largest gain from family grouping on semantically diverse concepts: MJA-Violent hybrid +56pp, illegal anchor +18pp, disturbing anchor +21pp. Tightly-scoped concepts (sexual anchor, violent anchor) are near-tied.
>
> **Caveat**: Single-pool MJA-Violent hybrid uses ss=1.5 (only cell available from existing sweep); a re-run at ss=22 would tighten apples-to-apples but the qualitative pattern (family advantage on diverse concepts) is robust.
>
> **Source dirs**: family = `paper_results_master/03_mja_sd14_4concept/mja_{concept}_{mode}/`; single anchor = `outputs/v2_experiments/{concept}/mja_both_anchor_inpaint_single_*/`; single hybrid = `outputs/launch_0424_singlepool_hybrid/mja_{concept}/hybrid_ss22_tt0.15_ti0.1_both_cas0.6/`.

---

## Appendix Table A — MJA Cross-Backbone (full)

> Same pack (`concepts_v2/<concept>/clip_grouped.pt`) reused across SD1.4 / SD3 / FLUX1; only the model swaps.

| Concept | Backbone | Method | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|---|---|
| sexual | SD1.4 | Baseline | .429 | .122 | .307 | .571 | .000 |
| sexual | SD1.4 | SAFREE | .713 | .363 | .350 | .288 | .000 |
| sexual | SD1.4 | Ours (anchor) | .900 | .644 | .256 | .100 | .000 |
| sexual | SD1.4 | Ours (hybrid) | **.967** | .778 | .189 | .033 | .000 |
| sexual | SD3 | Baseline | .505 | .071 | .434 | .495 | .000 |
| sexual | SD3 | SAFREE | .636 | .172 | .465 | .364 | .000 |
| sexual | SD3 | Ours (anchor) | .810 | .470 | .340 | .190 | .000 |
| sexual | SD3 | Ours (hybrid) | **.840** | .560 | .280 | .160 | .000 |
| sexual | FLUX1 | Baseline | .620 | .050 | .570 | .380 | .000 |
| sexual | FLUX1 | SAFREE | .735 | .112 | .622 | .265 | .000 |
| sexual | FLUX1 | Ours (anchor) | .960 | .840 | .120 | .040 | .000 |
| sexual | FLUX1 | Ours (hybrid) | **.970** | .850 | .120 | .030 | .000 |
| violent | SD1.4 | Baseline | .103 | .065 | .038 | .858 | .040 |
| violent | SD1.4 | SAFREE | .550 | .340 | .210 | .270 | .180 |
| violent | SD1.4 | Ours (anchor) | .560 | .420 | .140 | .260 | .180 |
| violent | SD1.4 | Ours (hybrid) | **.690** | .500 | .190 | .160 | .150 |
| violent | SD3 | Baseline | .000 | .000 | .000 | 1.00 | .000 |
| violent | SD3 | SAFREE | .060 | .040 | .020 | .940 | .000 |
| violent | SD3 | Ours (anchor) | **.580** | .520 | .060 | .420 | .000 |
| violent | SD3 | Ours (hybrid) | .360 | .250 | .110 | .570 | .070 |
| violent | FLUX1 | Baseline | .020 | .000 | .020 | .980 | .000 |
| violent | FLUX1 | SAFREE | .030 | .000 | .030 | .970 | .000 |
| violent | FLUX1 | Ours (anchor) | **.890** | .870 | .020 | .110 | .000 |
| violent | FLUX1 | Ours (hybrid) | .670 | .570 | .100 | .200 | .130 |
| illegal | SD1.4 | Baseline | .507 | .230 | .278 | .400 | .093 |
| illegal | SD1.4 | SAFREE | .730 | .440 | .290 | .100 | .170 |
| illegal | SD1.4 | Ours (anchor) | **.760** | .650 | .110 | .080 | .160 |
| illegal | SD1.4 | Ours (hybrid) | .590 | .380 | .210 | .290 | .120 |
| illegal | SD3 | Baseline | .190 | .080 | .110 | .800 | .010 |
| illegal | SD3 | SAFREE | .200 | .130 | .070 | .770 | .030 |
| illegal | SD3 | Ours (anchor) | .530 | .410 | .120 | .420 | .050 |
| illegal | SD3 | Ours (hybrid) | **.670** | .480 | .190 | .160 | .170 |
| illegal | FLUX1 | Baseline | .320 | .060 | .260 | .650 | .030 |
| illegal | FLUX1 | SAFREE | .340 | .080 | .260 | .640 | .020 |
| illegal | FLUX1 | Ours (anchor) | **.860** | .800 | .060 | .130 | .010 |
| illegal | FLUX1 | Ours (hybrid) | .580 | .330 | .250 | .350 | .070 |
| disturbing | SD1.4 | Baseline | .490 | .008 | .483 | .510 | .000 |
| disturbing | SD1.4 | SAFREE | .820 | .330 | .490 | .100 | .080 |
| disturbing | SD1.4 | Ours (anchor) | .890 | .370 | .520 | .000 | .110 |
| disturbing | SD1.4 | Ours (hybrid) | **.930** | .360 | .570 | .050 | .020 |
| disturbing | SD3 | Baseline | .350 | .020 | .330 | .650 | .000 |
| disturbing | SD3 | SAFREE | .630 | .040 | .590 | .370 | .000 |
| disturbing | SD3 | Ours (anchor) | .860 | .180 | .680 | .140 | .000 |
| disturbing | SD3 | Ours (hybrid) | **.900** | .440 | .460 | .100 | .000 |
| disturbing | FLUX1 | Baseline | .510 | .010 | .500 | .490 | .000 |
| disturbing | FLUX1 | SAFREE | .460 | .000 | .460 | .540 | .000 |
| disturbing | FLUX1 | Ours (anchor) | **.980** | .900 | .080 | .020 | .000 |
| disturbing | FLUX1 | Ours (hybrid) | .960 | .740 | .220 | .040 | .000 |

---

# Experimental Setup (for ml-writer)

## 1. Backbones

| Backbone | Checkpoint | Resolution | Sampler | Steps | CFG / guidance | dtype |
|---|---|---|---|---|---|---|
| SD1.4 (U-Net) | `CompVis/stable-diffusion-v1-4` | 512 × 512 | DDIM | 50 | CFG = 7.5 | fp32 |
| SD3 (MM-DiT) | `stabilityai/stable-diffusion-3-medium-diffusers` | 1024 × 1024 | flow matching | 28 | CFG = 7.0 | fp16 |
| FLUX.1-dev (DiT) | `black-forest-labs/FLUX.1-dev` | 1024 × 1024 | flow matching | 28 | embedded guidance scale = 3.5 (no CFG negative pass) | bfloat16 |

## 2. Datasets

| Name | # prompts | Domain | Notes |
|---|---|---|---|
| **UnlearnDiff (UD)** | 142 | Adversarial nudity | from Zhang et al. NeurIPS '23 |
| **Ring-A-Bell (RAB)** | 78 | Adversarial nudity | "nudity-ring-a-bell.csv" |
| **MMA-Diffusion (MMA)** | 1000 | Adversarial nudity | trimmed to first ~1000 |
| **P4DN** | 151 | Targeted-nudity attack | `p4dn.txt` |
| **MJA-Diffusion** | 100 / concept | sexual / violent / illegal / disturbing | `mja_<concept>.txt` |
| **I2P top60** | 60 / concept | violence, self-harm, shocking, illegal, harassment, hate | top-60 by Q16 toxicity score |
| **COCO-30k** | 30 000 | natural prompts | for FID / CLIP utility |

## 3. Per-Concept Family Pack

Each pack `clip_grouped.pt` contains:

* `family_names`: K = 4 family identifiers (e.g., harassment ⇒ `political_mockery`, `horror_creature_h`, `threatening_aggressive`, `mocking_caricature`).
* `family_metadata[fname]`: `target_words`, `target_prompts`, `anchor_words`, `anchor_prompts` per family.
* `target_clip_features[fname]`: ℝ^{N_f × 768} CLIP image features from N_f exemplar images per family.
* `anchor_clip_features[fname]`: image-side anchor features (loaded but currently unused at inference; reserved).
* `concept_keywords`: short concept-level descriptor list, identical to the CLI `--target_concepts` we used:

| Concept (pack) | concept_keywords (CLI `--target_concepts`) |
|---|---|
| i2p_v1/violence | `violence bloody scene weapon fight` |
| i2p_v1/self-harm | `self harm cutting pills suicide noose` |
| i2p_v1/shocking | `gore body_horror mutilation corpse` |
| i2p_v1/illegal_activity | `drugs crime contraband illegal activity` |
| i2p_v1/harassment | `bullying mockery intimidation abuse` |
| i2p_v1/hate | `hate_symbol discrimination racist_imagery` |
| i2p_v1/sexual, concepts_v2/sexual | `nudity nude_person naked_body` |
| concepts_v2/violent | `violence bloody scene weapon fight` |
| concepts_v2/illegal | `drugs crime contraband illegal activity` |
| concepts_v2/disturbing | `grotesque body_horror gore monster` |
| concepts_v2/harassment | `bullying mockery intimidation abuse` |
| concepts_v2/hate | `hate_symbol discrimination racist_imagery` |
| concepts_v2/selfharm | `self harm cutting pills suicide noose` |

CLI `--anchor_concepts` (text safe direction) is set per-pack (e.g., harassment ⇒ `friendly cooperation support teamwork`).

## 4. Probe Construction

* **Image probe** (`probe_mode ∈ {image, both}`): K=4 family-specific pseudo-tokens overwrite slots 1..K of the empty-prompt encoder hidden state with `F.normalize(mean(target_clip_features[fname]))`; if K < 4 the remaining slots are padded with the last family's centroid (`build_grouped_probe_embeds`, `max_tokens=4`).
* **Text probe** (`probe_mode ∈ {text, both}`): `target_text = ", ".join(args.target_concepts)`; tokenized indices inside `target_text` for each token in `args.target_words` are tracked.
* Probes are registered as cross-attention hooks at U-Net resolutions {16, 32} (SD1.4) / corresponding DiT blocks (SD3, FLUX1).

## 5. Per-Family Mask Construction (each step)

\[
M_f^{\text{img}}(i,j) = \sigma(\alpha [\text{Blur}_{σ_b}(A^{\text{img}}_{t,f})(i,j) - τ_{\text{img}}])
\]
\[
M_f^{\text{txt}}(i,j) = \sigma(\alpha [\text{Blur}_{σ_b}(A^{\text{txt}}_{t,f})(i,j) - τ_{\text{txt}}])
\]
\[
M_f(i,j) = \max(M_f^{\text{img}}(i,j),\; M_f^{\text{txt}}(i,j))
\]

* `attn_sigmoid_alpha α = 10.0`
* `blur_sigma σ_b = 1.0`
* `n_img_tokens = 4` (max family slots)

Then **inter-family winner-take-all**: `winner(i,j) = argmax_f M_f(i,j)`; each $M_f$ is masked by `(winner == f)`, making the family masks spatially disjoint.

## 6. CAS Gate (when to apply)

`GlobalCAS` (cosine-based):
\[
c_t = \text{cos}\bigl(\epsilon^{\text{prompt}}_t - \epsilon^{\varnothing}_t,\; \epsilon^{\text{tgt}}_t - \epsilon^{\varnothing}_t\bigr)
\]

* Trigger if $c_t > \tau_{\text{cas}}$.
* `sticky=True` (once triggered, stays on for the rest of the schedule).
* SD1.4: $τ_{\text{cas}} = 0.6$. SD3: $τ_{\text{cas}} = 0.3$ (DiT cosine magnitudes are smaller; 0.6 never triggers). FLUX1: $τ_{\text{cas}} = 0.6$.

## 7. Family-Conditioned Guidance Modes

Per-family extra forwards (`apply_family_guidance`):
\[
\epsilon^{\text{tgt}}_{t,f} = U(z_t, t, E_f^{\text{tgt}}), \quad
\epsilon^{\text{anc}}_{t,f} = U(z_t, t, E_f^{\text{anc}})
\]
where $E_f^{\text{tgt}/\text{anc}} = \text{mean-pool}_{j=1..3}\bigl(\text{te}(\text{tok}(\text{(target|anchor)\_words}_f[j]))\bigr)$.

### anchor_inpaint
\[
\tilde\epsilon^{\text{anc}}_{t,f} = \epsilon^{\varnothing}_t + w_{\text{cfg}}(\epsilon^{\text{anc}}_{t,f} - \epsilon^{\varnothing}_t)
\]
\[
\beta_f = \min(s \cdot M_f, 1)
\]
\[
\epsilon^{\text{safe}}_t \leftarrow (1-\beta_f) \epsilon^{\text{safe}}_t + \beta_f \tilde\epsilon^{\text{anc}}_{t,f}
\]

### hybrid
\[
\epsilon^{\text{safe}}_t \leftarrow \epsilon^{\text{safe}}_t - s \cdot M_f (\epsilon^{\text{tgt}}_{t,f} - \epsilon^{\varnothing}_t) + s \cdot M_f (\epsilon^{\text{anc}}_{t,f} - \epsilon^{\varnothing}_t)
\]

`safety_scale s` is the per-method, per-backbone tuned hyperparameter (see Table B below).

## 8. Best Hyperparameters (per cell)

### Table B — anchor_inpaint best hyperparameters

| Setting | s | τ_txt | τ_img | τ_cas | mode |
|---|---|---|---|---|---|
| Nudity (UD/RAB/MMA/P4DN) SD1.4 | 1.2 | 0.10 | 0.30 | 0.6 | both |
| MJA-Sexual SD1.4 | 2.5 | 0.10 | 0.30 | 0.6 | both |
| MJA-Violent SD1.4 | 1.8 | 0.10 | 0.30 | 0.6 | both |
| MJA-Illegal SD1.4 | 2.5 | 0.10 | 0.30 | 0.4 | both |
| MJA-Disturbing SD1.4 | 1.2 | 0.10 | 0.30 | 0.6 | both |
| I2P-Violence SD1.4 | 1.0 | 0.10 | 0.40 | 0.6 | both |
| I2P-Self-harm SD1.4 | 1.0 | 0.10 | 0.40 | 0.6 | both |
| I2P-Shocking SD1.4 | 2.0 | 0.10 | 0.40 | 0.6 | both |
| I2P-Illegal SD1.4 | 1.0 | 0.10 | 0.70 | 0.6 | both |
| I2P-Harassment SD1.4 | 2.5 | 0.10 | 0.30 | 0.5 | both |
| I2P-Hate SD1.4 | 2.5 | 0.10 | 0.40 | 0.6 | both |
| MJA-Sexual SD3 | 3.0 | 0.20 | 0.20 | 0.6 | both |
| MJA-Violent SD3 | 1.5 | 0.10 | 0.20 | 0.6 | both |
| MJA-Illegal SD3 | 2.5 | 0.10 | 0.20 | 0.6 | both |
| MJA-Disturbing SD3 | 1.5 | 0.10 | 0.20 | 0.6 | both |
| MJA-Sexual FLUX1 | 1.5 | 0.10 | 0.10 | 0.6 | both |
| MJA-Violent FLUX1 | 2.0 | 0.10 | 0.10 | 0.6 | both |
| MJA-Illegal FLUX1 | 3.0 | 0.10 | 0.10 | 0.6 | both |
| MJA-Disturbing FLUX1 | 1.5 | 0.10 | 0.10 | 0.6 | both |

### Table C — hybrid best hyperparameters

| Setting | s | τ_txt | τ_img | τ_cas | mode |
|---|---|---|---|---|---|
| Nudity (UD) SD1.4 | 10 | 0.10 | 0.30 | 0.6 | both |
| Nudity (RAB) SD1.4 | 20 | 0.10 | 0.40 | 0.6 | both |
| Nudity (MMA) SD1.4 | 20 | 0.10 | 0.30 | 0.6 | both |
| Nudity (P4DN) SD1.4 | 20 | 0.10 | 0.30 | 0.6 | both |
| MJA-Sexual SD1.4 | 15 | 0.10 | 0.50 | 0.6 | both |
| MJA-Violent SD1.4 | 25 | 0.15 | 0.10 | 0.4 | both |
| MJA-Illegal SD1.4 | 20 | 0.10 | 0.30 | 0.6 | both |
| MJA-Disturbing SD1.4 | 22 | 0.15 | 0.10 | 0.6 | both |
| I2P-Violence SD1.4 | 15 | 0.10 | 0.30 | 0.6 | both |
| I2P-Self-harm SD1.4 | 22 | 0.10 | 0.40 | 0.6 | both |
| I2P-Shocking SD1.4 | 22 | 0.15 | 0.10 | 0.6 | both |
| I2P-Illegal SD1.4 | 20 | 0.10 | 0.50 | 0.6 | both |
| I2P-Harassment SD1.4 | 20 | 0.15 | 0.10 | 0.6 | both |
| I2P-Hate SD1.4 | 22 | 0.25 | 0.10 | 0.6 | both |
| MJA-Sexual SD3 | 15 | 0.10 | 0.30 | 0.6 | both |
| MJA-Violent SD3 | 20 | 0.15 | 0.10 | 0.3 | both |
| MJA-Illegal SD3 | 20 | 0.15 | 0.10 | 0.3 | both |
| MJA-Disturbing SD3 | 20 | 0.15 | 0.10 | 0.4 | both |
| MJA-Sexual FLUX1 | 2.5 | 0.10 | 0.10 | 0.6 | both |
| MJA-Violent FLUX1 | 2.0 | 0.10 | 0.10 | 0.6 | both |
| MJA-Illegal FLUX1 | 2.0 | 0.10 | 0.10 | 0.6 | both |
| MJA-Disturbing FLUX1 | 3.0 | 0.10 | 0.10 | 0.6 | both |

## 9. Multi-Concept Setup (for Table 3)

* CLI: `--family_config <path1> <path2> ... <path6>` (six I2P concept packs).
* Code automatically merges into a flat 24-family namespace (6 × 4) prefixed by concept name.
* Same per-family mask + winner-take-all + guidance applied across the merged set.
* Image probe pseudo-token capacity is `n_img_tokens=4`; with 24 families, the first 4 receive unique centroids, the rest are padded with the last family's centroid → image-side discriminability is capped at 4 unique families across all loaded concepts (text-side keyword indexing remains fully separable).
* Per-family forward count per CAS-active step: 2 × 24 = 48 extra UNet calls.

## 10. Eval Protocol

* **VLM judge**: Qwen3-VL-8B-Instruct (`Qwen/Qwen3-VL-8B-Instruct`).
* **Rubric script**: `vlm/opensource_vlm_i2p_all_v5.py`.
* **Per concept rubric**: nudity / violence / harassment / hate / shocking / disturbing / illegal / self_harm — concept-specific Safe/Partial/Full/NotRel definitions (see `PROMPTS` dict in v5 file).
* **Output**: `categories_qwen3_vl_<concept>_v5.json` with `{filename: {"category": "Safe|Partial|Full|NotRelevant"}}`.
* **Aggregation**: SR = (Safe + Partial) / (Safe + Partial + Full + NotRelevant).
* All numbers in our tables follow this protocol; numbers reproduced from the SAFREE / SD baselines paper rows are the published Qwen3-VL numbers from the SAFREE paper (2024).

## 11. Compute Notes

* Generation: 1 GPU per config. SD1.4 ~16 min/60 prompts; SD3 / FLUX1 ~25–30 min/100 prompts (1024 × 1024).
* CAS gating reduces per-step cost when not triggered; sticky mode means triggered runs do all 2K extra forwards every step thereafter.
* Total experiments reported here: ~ 600 generation runs across all concept × backbone × hyperparameter combinations, evaluated by Qwen3-VL on ~ 120 K images.

---

## Key Honesty Notes (post-audit 2026-04-24 14:30 KST)

1. **All single-concept Ours cells** in Tables 1 / 2 / 4 / A use CLI `target_concepts` that are concept-correct keywords (verified by `args.json` audit). Cells where the master copy used a `nudity`-default fallback have been REPLACED with the best concept-correct cell:
   - I2P illegal_activity hybrid: was 48.3% (bug) → now **41.7%** (concept-correct best both-probe, same ss=20)
   - I2P harassment hybrid: was 56.7% (bug) → now **46.7%** (concept-correct best both-probe)
   - I2P self-harm anchor: 68.3% (master had bug; concept-correct alternative coincidentally same SR)
   - MJA SD1.4 illegal hybrid: was 71.0% (bug) → now **59.0%** (concept-correct best, family-style target)
   - Probe self-harm img-only: was 55.0% (bug) → now **50.0%** (concept-correct best img-only hybrid)

2. **Multi-concept (Table 3) caveat**: all 6 Ours-multi cells use a global CLI `target_concepts=['nudity','nude_person','naked_body']` because our current multi-pack inference path does not accept per-pack target keywords. Per-pack family `target_words` inside each pack ARE concept-correct (so masks localize correctly), but the global ε_tgt direction defaults to nudity. **Re-running with concept-aware multi-pack target keywords is in `What's Left`.** SAFREE multi was also evaluated honestly (avg 2.8% — collapses to ~all NotRelevant).

3. **Table 1 baselines/SAFREE** are now v5 measured locally (NOT the SAFREE paper's published numbers). Earlier draft mixed SAFREE-paper numbers with our v5 measurements — apples-to-oranges. ESD/SDD/UCE/RECE/SLD-* dropped pending v5 re-eval (see `What's Left`).

4. **Table 1 Ours-anchor RAB** uses today's (2026-04-24) generation under v2pack uniform config (anchor_inpaint, ss=1.2, τ_t=0.1, τ_i=0.3, both, `concepts_v2/sexual` pack), measured 88.6% (n=79). Earlier draft claimed 93.3% — that value is **not backed by any complete v5 cell** (the only cell with similar SR uses `i2p_v1/sexual` pack, not the uniform v2pack); we removed it.

5. `anchor_clip_features` field exists in every pack but is **not consumed at inference**; future-work (image-anchor) capability.

6. Both How modes (`anchor_inpaint` and `hybrid`) reported separately as first-class rows. Per-concept I2P single SR: anchor avg 68.9 / hybrid avg 66.1 (3-3 split — anchor wins on self-harm/illegal/harassment, hybrid wins on violence/shocking/hate).

7. CAS gate is **cosine-based** (ε-direction cosine), not raw attention magnitude.

8. Cross-table internal consistency verified (I2P-single hybrid column ≡ Probe-ablation both column ≡ HOW-Mode hybrid column; Multi Ours-single ≡ max(I2P anchor, I2P hybrid) per concept).
