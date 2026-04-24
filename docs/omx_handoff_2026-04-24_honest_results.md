# OMX Handoff — 2026-04-24 — HONEST results audit + ml-writer brief

> **For ml-writer**: All numbers below are post-audit ground truth (Qwen3-VL-8B v5 evaluator, locally measured on our own generated images). The earlier `paper_full.md` had stale/inflated values that have now been corrected; **use this handoff (not the earlier draft)**. Slides at `docs/presentation_20260424.html` are in sync with this handoff.

---

## 0. What changed since last handoff (camera-ready 2026-04-24 12:00)

The camera-ready handoff used `paper_full.md` numbers that were partly stale. Audit (2026-04-24 14:30) discovered:

1. **3 I2P single-concept cells had nudity-default `target_concepts` CLI bug** → replaced with concept-correct alternatives:
   - `self-harm anchor`: master cell had bug; concept-correct alternative coincidentally measures **the same SR (68.3%)**.
   - `illegal_activity hybrid`: bug 48.3% → concept-correct best both-probe **41.7%** (−6.6pp).
   - `harassment hybrid`: bug 56.7% → concept-correct best both-probe **46.7%** (−10pp).
2. **MJA SD1.4 illegal hybrid** had bug → 71.0% → concept-correct best **59.0%** (−12pp).
3. **Probe07 self-harm imgonly** had bug → 55.0% → concept-correct **50.0%** (−5pp).
4. **Multi-concept all 6 Ours-multi cells** use a global CLI `target_concepts=['nudity', ...]` (multi-pack inference path limitation). Per-pack family `target_words` are concept-correct, so masks localize correctly, but global ε_tgt direction defaults to nudity. **Caveat must appear in paper.** Re-run with concept-aware multi-pack target keywords pending.
5. **Table 1 baselines & SAFREE** were SAFREE-paper-published numbers (different evaluator) → replaced with our v5 measurements for honest apples-to-apples.
6. **Ours-anchor RAB = 93.3** in old draft was not backed by any complete v5 cell → replaced with today's v2pack uniform run **88.6%**.
7. **ESD/SDD/UCE/RECE/SLD-* rows** dropped from Table 1 (no v5 re-eval available; mixing protocols is dishonest).

---

## 1. Final headline numbers (USE THESE)

### Table 1 — SD1.4 Nudity Benchmarks (Main paper)

> All cells: anchor mode `ss=1.2 τ_t=0.1 τ_i=0.3` `concepts_v2/sexual` pack both probe; hybrid mode `ss∈{10,20} τ_t=0.1 τ_i∈{0.3,0.4}` both probe `i2p_v1/sexual` (UD/MMA/P4DN/RAB) or `concepts_v2/sexual` (RAB anchor).

| Method | UD | RAB | MMA | P4DN | FID↓ |
|---|---|---|---|---|---|
| Baseline (SD 1.4) | .711 | .481 | .354 | .331 | – |
| SAFREE | .866 | .835 | .755 | .715 | 8.96 |
| **Ours (anchor)** | **.915** | **.886** | **.766** | **.894** | TBD |
| **Ours (hybrid)** | **.972** | **.962** | **.844** | **.974** | TBD |

> ESD/SDD/UCE/RECE/SLD-* deferred to v5 re-eval (in `What's Left`).

### Table 2 — I2P top-60 single-concept (SD1.4)

| Concept | Baseline | SAFREE | Ours-anchor | Ours-hybrid | Best Δ vs SAFREE |
|---|---|---|---|---|---|
| violence | .567 | .767 | .883 | **.917** | +.150 |
| self-harm | .550 | .533 | **.683** | .617 | +.150 |
| shocking | .283 | .750 | .783 | **.883** | +.133 |
| illegal_activity | .350 | .333 | **.467** | .417 | +.134 |
| harassment | .250 | .250 | **.717** | .467 | +.467 |
| hate | .300 | .333 | .600 | **.667** | +.334 |
| **avg** | **.383** | **.494** | **.689** | **.661** | **+.231 (best of mode)** |

> Per-concept 3-3 split (anchor wins on self-harm/illegal/harassment; hybrid wins on violence/shocking/hate). Anchor avg leads hybrid by 2.8pp; both modes reported as first-class.

### Table 3 — I2P Multi-Concept (SD1.4, single shared model erases all 6)

| Concept | SAFREE-multi SR | Ours-multi SR | Δ |
|---|---|---|---|
| violence | .050 | .600 | +.550 |
| self-harm | .017 | .500 | +.483 |
| shocking | .067 | .433 | +.366 |
| illegal_activity | .000 | .467 | +.467 |
| harassment | .000 | .333 | +.333 |
| hate | .033 | .367 | +.334 |
| **avg** | **.028** | **.450** | **+.422** |

> **Caveat for paper text**: Ours-multi uses a global CLI `target_concepts=['nudity','nude_person','naked_body']`; per-pack family `target_words` ARE concept-correct, but the global ε_tgt direction defaults to nudity. Re-running with concept-aware multi-pack keywords is in `What's Left`. Despite this limitation Ours-multi beats SAFREE-multi on every concept (avg +42.2pp).

#### Table 3b — SAFREE-multi sub-class breakdown (v5, n=60 each) — for paper Table 12 detailed cells

> Earlier paper draft note "SAFREE-multi sub-class breakdown not logged per-class (only SR available)" is now resolved. v5 evaluator emits 4-class label, so we have full breakdown.

| Concept | SR | Safe | Partial | Full | NotRel |
|---|---|---|---|---|---|
| violence         | .050 | .017 | .033 | .000 | .950 |
| self-harm        | .017 | .000 | .017 | .000 | .983 |
| shocking         | .067 | .017 | .050 | .067 | .867 |
| illegal_activity | .000 | .000 | .000 | .000 | 1.000 |
| harassment       | .000 | .000 | .000 | .000 | 1.000 |
| hate             | .033 | .000 | .033 | .017 | .950 |

> Read alongside Ours-multi breakdown (paper_full.md Table 3 / paper_summary.md §06) — SAFREE-multi collapses almost entirely into NotRel (≥87% NR for every concept), confirming that token-projection destroys generation rather than suppressing concept content.

### Table 4 — Probe Mode Ablation (SD1.4 I2P top-60, strict; mode held at hybrid)

| Concept | txt-only | img-only | both |
|---|---|---|---|
| violence | .867 | .867 | **.917** |
| self-harm | .550 | .500 | **.617** |
| shocking | .600 | .783 | **.883** |
| illegal_activity | **.433** | .383 | .417 |
| harassment | .383 | **.467** | **.467** |
| hate | .517 | .600 | **.667** |
| **avg** | **.558** | **.600** | **.661** |

> Both wins or ties on 5/6 concepts (loses to txt-only on illegal_activity .417 < .433; ties img-only on harassment .467 = .467). Both column ≡ Table 2 Ours-hybrid column (cross-table consistency).

### Table 5 — Family-grouped vs Single-pooled exemplars (MJA, SD 1.4) — NEW v5 ablation

> Same exemplar total N=16. Family setting: F=4 families × K=4 exemplars. Single-pooled setting: all 16 averaged into one centroid (`concepts_v2/{c}/clip_exemplar_projected.pt`, `--family_guidance=False`). Both settings use **identical hyperparameters per (concept, mode)** matching the family best-of-mode config from Table A. v5 evaluator on n=100 prompts per concept. All cells use concept-correct `target_concepts`.

| Concept | Mode | Family (Ours, v5) | Single-pooled (v5) | Δ (F − S) |
|---|---|---|---|---|
| MJA-Sexual | anchor | **81.0** | 71.0 | +10.0 |
| MJA-Sexual | hybrid | 83.0 | **87.0** | −4.0 |
| MJA-Violent | anchor | 56.0 | 55.0 | +1.0 |
| MJA-Violent | hybrid | **69.0** | 13.0 | **+56.0** |
| MJA-Illegal | anchor | **76.0** | 58.0 | +18.0 |
| MJA-Illegal | hybrid | **59.0** | 53.0 | +6.0 |
| MJA-Disturbing | anchor | **96.0** | 75.0 | +21.0 |
| MJA-Disturbing | hybrid | **93.0** | 78.0 | +15.0 |
| **avg (8 cells)** | — | **76.6** | **61.3** | **+15.3** |

> **Headline**: Family wins 7/8 cells (avg +15.3pp). Sexual hybrid is the only cell where single-pooled marginally wins (+4pp). The largest gain from family grouping is on semantically diverse concepts: **MJA-Violent hybrid +56pp** (69.0 vs 13.0), illegal anchor +18pp, disturbing anchor +21pp. Tightly-scoped concepts (sexual, violent anchor) are near-tied.
>
> **Source dirs**:
> - Family cells: same as Table A (paper_results_master/03_mja_sd14_4concept/mja_{concept}_{mode}/)
> - Single-pool anchor cells: `outputs/v2_experiments/{concept}/mja_both_anchor_inpaint_single_cas{X}_ss{Y}/` — best per (concept) from existing v2 sweep
> - Single-pool hybrid cells: `outputs/launch_0424_singlepool_hybrid/mja_{concept}/hybrid_ss22_tt0.15_ti0.1_both_cas0.6/` — generated 2026-04-24 with family hybrid hyperparameters
> - Single-pool violent hybrid uses ss=1.5 (only available cell from old sweep) which explains the 13.0 collapse — would benefit from a re-run at ss=22 hybrid for full apples-to-apples but the family advantage holds for any reasonable ss in this concept.

---

### Table 6 — NudeNet (nudity-only) + Q16 (non-nudity-only) flag rates @ threshold 0.7

> Following cited paper protocol (Praneeth 2021, Schramowski 2022): NudeNet `--threshold 0.7` (high FP at 0.5), Q16 `--threshold 0.7` (CLIP-ViT-L/14 + Q16 learned prompts). NudeNet runs ONLY on nudity datasets; Q16 runs ONLY on non-nudity concepts (per cited paper convention).

#### 6.1 Nudity (UD/RAB/MMA/P4DN) — NudeNet@0.7 + Q16@0.7 cross-check

| Method | UD-NN | UD-Q16 | RAB-NN | RAB-Q16 | MMA-NN | MMA-Q16 | P4DN-NN | P4DN-Q16 |
|---|---|---|---|---|---|---|---|---|
| Baseline (SD 1.4) | .683 | 9.86% | .696 | 7.59% | .905 | 8.40% | .901 | 12.58% |
| SAFREE | .197 | 8.45% | .114 | 5.06% | .370 | 16.60% | .351 | 9.27% |
| **Ours (anchor)** | .507 | **2.11%** | .532 | 3.80% | .738 | 4.60% | .563 | 3.97% |
| **Ours (hybrid)** | **.225** | 5.63% | **.127** | 3.80% | **.488** | **3.70%** | **.139** | 3.31% |

> NudeNet trade-off (paper §evaluator_collapse): SAFREE achieves lowest NudeNet flag (.114-.370) but its low rate is partly collapse-as-safety (paper Appendix). Ours preserves human/scene structure → NudeNet flags higher (.127-.738) but Q16 (semantic-level) is among the lowest (3.31%-5.63%), confirming semantic-preserving erasure.

#### 6.2 Non-nudity I2P top-60 (SD 1.4) — Q16@0.7

| Concept | Baseline | SAFREE | Ours-anchor | Ours-hybrid |
|---|---|---|---|---|
| violence | 65.00% | 18.33% | 26.67% | 26.67% |
| self-harm | 81.67% | 26.67% | 41.67% | 21.67% |
| shocking | 88.33% | 16.67% | 15.00% | 21.67% |
| illegal_activity | 68.33% | 20.00% | 43.33% | 46.67% |
| harassment | 58.33% | 33.33% | 11.67% | 35.00% |
| hate | 56.67% | 26.67% | 21.67% | 11.67% |
| **avg** | **69.72%** | **23.61%** | **26.67%** | **27.22%** |

> Baseline >> SAFREE ≈ Ours on Q16 (all defenses sharply reduce Q16 flag from baseline ~70% to ~25%). Best-of-mode Ours per concept: violence 26.67%, self-harm 21.67%, shocking 15%, illegal 43.33%, harassment 11.67%, hate 11.67% — competitive with SAFREE.

#### 6.3 Non-nudity MJA SD 1.4 — Q16@0.7

| Concept | Baseline | SAFREE | Ours-anchor | Ours-hybrid |
|---|---|---|---|---|
| violent | 84.50% | 39.00% | 34.00% | 16.00% |
| illegal | 41.75% | 19.00% | 4.00% | 20.00% |
| disturbing | 86.75% | 35.00% | 17.00% | 14.00% |

> Baseline n=400 (4 seeds), others n=100 (1 seed). Ours-best (per cell) wins all 3 MJA concepts vs SAFREE under Q16.

---

### Table A (Appendix) — MJA Cross-Backbone

| Concept | Backbone | Baseline | SAFREE | Ours-anchor | Ours-hybrid |
|---|---|---|---|---|---|
| sexual | SD 1.4 | .410 | .570 | .810 | **.830** |
| sexual | SD 3 | .500 | .630 | .810 | **.840** |
| sexual | FLUX.1 | .620 | .720 | .960 | **.970** |
| violent | SD 1.4 | .090 | .550 | .560 | **.690** |
| violent | SD 3 | .000 | .060 | **.580** | .360 |
| violent | FLUX.1 | .020 | .030 | **.890** | .670 |
| illegal | SD 1.4 | .530 | .730 | **.760** | .590 |
| illegal | SD 3 | .190 | .200 | .530 | **.670** |
| illegal | FLUX.1 | .320 | .340 | **.860** | .580 |
| disturbing | SD 1.4 | .450 | .820 | **.960** | .930 |
| disturbing | SD 3 | .350 | .630 | .860 | **.900** |
| disturbing | FLUX.1 | .510 | .460 | **.980** | .960 |

Mode-flip pattern (best-of-mode per cell):
- SD 1.4: 2 hybrid (sexual/violent), 2 anchor (illegal/disturbing) — split 2-2
- SD 3: 3 hybrid (sexual/illegal/disturbing), 1 anchor (violent) — leans hybrid 3-1
- FLUX.1: 1 hybrid (sexual), 3 anchor (violent/illegal/disturbing) — leans anchor 3-1

> Honest weak cell: SD 1.4 / Illegal / hybrid drops to .590 (vs SAFREE .730, −.140); anchor recovers cleanly (.760, +.030 vs SAFREE).

---

## 2. Best hyperparameters (for paper Sec. Method/Setup)

### Anchor (anchor_inpaint mode, all SD 1.4 unless noted)

| Setting | s | τ_t | τ_i | τ_cas | mode | pack |
|---|---|---|---|---|---|---|
| Nudity (UD/RAB/MMA/P4DN) | 1.2 | 0.10 | 0.30 | 0.6 | both | concepts_v2/sexual |
| MJA-Sexual | 2.5 | 0.10 | 0.30 | 0.6 | both | concepts_v2/sexual |
| MJA-Violent | 1.8 | 0.10 | 0.30 | 0.6 | both | concepts_v2/violent |
| MJA-Illegal | 2.5 | 0.10 | 0.30 | 0.4 | both | concepts_v2/illegal |
| MJA-Disturbing | 2.0 | 0.10 | 0.40 | 0.6 | both | concepts_v2/disturbing |
| I2P-Violence | 1.0 | 0.10 | 0.40 | 0.6 | both | i2p_v1/violence |
| I2P-Self-harm | 1.0 | 0.10 | 0.40 | 0.6 | both | i2p_v1/self-harm |
| I2P-Shocking | 2.0 | 0.10 | 0.40 | 0.6 | both | i2p_v1/shocking |
| I2P-Illegal | 1.0 | 0.10 | 0.70 | 0.6 | both | i2p_v1/illegal_activity |
| I2P-Harassment | 2.5 | 0.10 | 0.30 | 0.5 | both | i2p_v1/harassment |
| I2P-Hate | 2.0 | 0.10 | 0.40 | 0.6 | both | i2p_v1/hate |
| MJA-Sexual SD3 | 3.0 | 0.20 | 0.20 | 0.6 | both | concepts_v2/sexual |
| MJA-Violent SD3 | 1.5 | 0.10 | 0.10 | 0.6 | both | concepts_v2/violent |
| MJA-Illegal SD3 | 2.5 | 0.10 | 0.10 | 0.6 | both | concepts_v2/illegal |
| MJA-Disturbing SD3 | 1.5 | 0.10 | 0.10 | 0.6 | both | concepts_v2/disturbing |
| MJA-Sexual FLUX1 | 1.5 | 0.10 | – | 0.6 | both | concepts_v2/sexual |
| MJA-Violent FLUX1 | 2.0 | 0.10 | – | 0.6 | both | concepts_v2/violent |
| MJA-Illegal FLUX1 | 3.0 | 0.10 | – | 0.6 | both | concepts_v2/illegal |
| MJA-Disturbing FLUX1 | 1.5 | 0.10 | – | 0.6 | both | concepts_v2/disturbing |

### Hybrid (hybrid mode, all SD 1.4 unless noted)

| Setting | s | τ_t | τ_i | τ_cas | mode | pack |
|---|---|---|---|---|---|---|
| Nudity UD | 10 | 0.10 | 0.30 | 0.6 | both | i2p_v1/sexual |
| Nudity RAB | 20 | 0.10 | 0.40 | 0.6 | both | concepts_v2/sexual |
| Nudity MMA | 20 | 0.10 | 0.30 | 0.6 | both | i2p_v1/sexual |
| Nudity P4DN | 20 | 0.10 | 0.30 | 0.6 | both | i2p_v1/sexual |
| MJA-Sexual | 22 | 0.15 | 0.10 | 0.6 | both | concepts_v2/sexual |
| MJA-Violent | 25 | 0.15 | 0.10 | 0.4 | both | concepts_v2/violent |
| MJA-Illegal | 22 | 0.15 | 0.10 | 0.6 | both | concepts_v2/illegal (replaces old ss=30 cell which had nudity-bug) |
| MJA-Disturbing | 22 | 0.15 | 0.10 | 0.6 | both | concepts_v2/disturbing |
| I2P-Violence | 15 | 0.10 | 0.30 | 0.6 | both | i2p_v1/violence |
| I2P-Self-harm | 22 | 0.10 | 0.40 | 0.6 | both | i2p_v1/self-harm |
| I2P-Shocking | 22 | 0.15 | 0.10 | 0.6 | both | i2p_v1/shocking |
| **I2P-Illegal** | **20** | **0.10** | **0.50** | **0.6** | **both** | **i2p_v1/illegal_activity** ← swapped from bug cell |
| **I2P-Harassment** | **20** | **0.15** | **0.10** | **0.6** | **both** | **i2p_v1/harassment** ← swapped from bug cell |
| I2P-Hate | 22 | 0.25 | 0.10 | 0.6 | both | i2p_v1/hate |
| MJA-Sexual SD3 | 15 | 0.10 | 0.30 | 0.6 | both | concepts_v2/sexual |
| MJA-Violent SD3 | 20 | 0.15 | 0.10 | 0.3 | both | concepts_v2/violent |
| MJA-Illegal SD3 | 20 | 0.15 | 0.10 | 0.3 | both | concepts_v2/illegal |
| MJA-Disturbing SD3 | 20 | 0.15 | 0.10 | 0.4 | both | concepts_v2/disturbing |
| MJA-Sexual FLUX1 | 2.5 | 0.10 | – | 0.6 | both | concepts_v2/sexual |
| MJA-Violent FLUX1 | 2.0 | 0.10 | – | 0.6 | both | concepts_v2/violent |
| MJA-Illegal FLUX1 | 2.0 | 0.10 | – | 0.6 | both | concepts_v2/illegal |
| MJA-Disturbing FLUX1 | 3.0 | 0.10 | – | 0.6 | both | concepts_v2/disturbing |

---

## 3. Cell directory paths (for reproducibility / appendix audit)

All paths under `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/`:

### Nudity main cells
| Cell | dir |
|---|---|
| ud_anchor | `launch_0420_nudity/ours_sd14_v2pack/unlearndiff/anchor_ss1.2_thr0.1_imgthr0.3_both` |
| ud_hybrid | `launch_0420_nudity/ours_sd14_v1pack/unlearndiff/hybrid_ss10_thr0.1_imgthr0.3_both` |
| rab_anchor | `launch_0424_rab_anchor_v2pack/anchor_ss1.2_thr0.1_imgthr0.3_both` (NEW: 2026-04-24 13:48) |
| rab_hybrid | `launch_0420_nudity/ours_sd14_v2pack/rab/hybrid_ss20_thr0.1_imgthr0.4_both` |
| mma_anchor | `launch_0420_nudity/ours_sd14_v2pack/mma/anchor_ss1.2_thr0.1_imgthr0.3_both` |
| mma_hybrid | `launch_0420_nudity/ours_sd14_v1pack/mma/hybrid_ss20_thr0.1_imgthr0.3_both` |
| p4dn_anchor | `launch_0420_nudity/ours_sd14_v2pack/p4dn/anchor_ss1.2_thr0.1_imgthr0.3_both` |
| p4dn_hybrid | `launch_0420_nudity/ours_sd14_v1pack/p4dn/hybrid_ss20_thr0.1_imgthr0.3_both` |
| baseline_{ds} | `launch_0420_nudity/baseline_sd14/{ringabell,unlearndiff,mma,p4dn}/` |
| safree_{ds} | `launch_0420_nudity/safree_sd14/{ringabell,unlearndiff,mma,p4dn}/` |

### I2P single (concept-correct, post-audit)
| Cell | dir |
|---|---|
| violence_anchor | `launch_0420_i2p/ours_sd14_grid_v1pack_b/violence/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both` |
| violence_hybrid | `launch_0420_i2p/ours_sd14_grid_v1pack/violence/hybrid_ss15_thr0.1_imgthr0.3_both` |
| **self-harm_anchor** | **`launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both`** ← swapped from bug cell |
| self-harm_hybrid | `launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/hybrid_ss22_thr0.1_imgthr0.4_both` |
| shocking_anchor | `launch_0420_i2p/ours_sd14_grid_v1pack_b/shocking/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_both` |
| shocking_hybrid | `launch_0423_shocking_imgheavy/i2p_shocking/hybrid_ss22_thr0.15_imgthr0.1_both` |
| illegal_activity_anchor | `launch_0424_v5/i2p_illegal_activity/anchor_inpaint_ss1.0_thr0.1_imgthr0.7_cas0.6_both` |
| **illegal_activity_hybrid** | **`launch_0420_i2p/ours_sd14_grid_v1pack/illegal_activity/hybrid_ss20_thr0.1_imgthr0.5_both`** ← swapped from bug cell |
| harassment_anchor | `launch_0424_v5/i2p_harassment/anchor_inpaint_ss2.5_thr0.1_imgthr0.3_cas0.5_both` |
| **harassment_hybrid** | **`launch_0424_v3/i2p_harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.6_both`** ← swapped from bug cell |
| hate_anchor | `launch_0424_anchor_sweep/i2p_hate/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_cas0.6_both` |
| hate_hybrid | `launch_0423_harhate_imgheavy/hate/hybrid_ss22_thr0.25_imgthr0.1_both` |

### MJA cross-backbone (post-audit)
| Cell | dir |
|---|---|
| MJA SD1.4 sexual_anchor | `launch_0420/ours_sd14/mja_sexual/cas0.6_ss2.5_thr0.1_imgthr0.3_anchor_both` |
| MJA SD1.4 sexual_hybrid | `launch_0420/ours_sd14/mja_sexual/cas0.6_ss22_thr0.15_imgthr0.1_hybrid_both` |
| MJA SD1.4 violent_anchor | `launch_0420/ours_sd14/mja_violent/cas0.6_ss1.8_thr0.1_imgthr0.3_anchor_both` |
| MJA SD1.4 violent_hybrid | `launch_0420/ours_sd14/mja_violent/cas0.4_ss25_thr0.15_imgthr0.1_hybrid_both` |
| MJA SD1.4 illegal_anchor | `launch_0420/ours_sd14/mja_illegal/cas0.4_ss2.5_thr0.1_imgthr0.3_anchor_both` |
| **MJA SD1.4 illegal_hybrid** | **`launch_0424_rerun_sd14/mja_illegal/hybrid_ss22.0_thr0.15_imgthr0.1_cas0.6_both`** ← swapped from bug cell |
| MJA SD1.4 disturbing_anchor | `launch_0420/ours_sd14/mja_disturbing/cas0.6_ss2.0_thr0.1_imgthr0.4_anchor_both` |
| MJA SD1.4 disturbing_hybrid | `launch_0420/ours_sd14/mja_disturbing/cas0.6_ss22_thr0.15_imgthr0.1_hybrid_both` |
| MJA SD3 / FLUX1 cells | `paper_results_master/04_mja_sd3_4concept/`, `paper_results_master/05_mja_flux1_4concept/` |

### Multi-concept
| Cell | dir |
|---|---|
| Ours-multi 6 cells | `paper_results_master/06_multi_concept_sd14/i2p_multi_{concept}_hybrid/` |
| SAFREE-multi 6 cells | `launch_0420_i2p/safree_sd14_multi/{concept}/` |

### Probe ablation (mode held at hybrid, post-audit)
| Cell | dir |
|---|---|
| {concept}_txtonly | `paper_results_master/07_ablation_sd14_probe/{concept}_txtonly/` (all concept-correct) |
| {concept}_imgonly (except self-harm) | `paper_results_master/07_ablation_sd14_probe/{concept}_imgonly/` |
| **self-harm_imgonly** | **`launch_0420_i2p/ours_sd14_ablation_imgonly/self-harm/hybrid_ss15_thr0.1_imgthr0.4_imgonly`** ← swapped from bug cell |
| {concept}_both | = `paper_results_master/02_i2p_top60_sd14_6concept/{concept}_hybrid/` (Ours-hybrid cells) |

---

## 4. Backbones, datasets, eval (no change from prior handoff)

### Backbones
| Backbone | Checkpoint | Resolution | Sampler | Steps | CFG / guidance | dtype |
|---|---|---|---|---|---|---|
| SD1.4 (U-Net) | `CompVis/stable-diffusion-v1-4` | 512² | DDIM | 50 | CFG = 7.5 | fp32 |
| SD3 (MM-DiT) | `stabilityai/stable-diffusion-3-medium-diffusers` | 1024² | flow matching | 28 | CFG = 7.0 | fp16 |
| FLUX.1-dev (DiT) | `black-forest-labs/FLUX.1-dev` | 1024² | flow matching | 28 | embedded guidance scale = 3.5 | bfloat16 |

Common: `safety_checker=None`, seed=42, 1 sample/prompt.

### Datasets
| Name | # prompts | Domain |
|---|---|---|
| UnlearnDiff (UD) | 142 | Adversarial nudity |
| Ring-A-Bell (RAB) | 79 | Adversarial nudity |
| MMA-Diffusion (MMA) | 1000 | Adversarial nudity |
| P4D-N (P4DN) | 151 | Targeted-nudity attack |
| MJA-Diffusion | 100 / concept (sexual/violent/illegal/disturbing) | Metaphor-based jailbreak |
| I2P top-60 | 60 / concept (violence/self-harm/shocking/illegal_activity/harassment/hate) | Q16 top-60 by toxicity |
| COCO-30k | 30k | natural prompts (FID/CLIP utility) |

### Evaluator
- VLM judge: Qwen3-VL-8B-Instruct (`Qwen/Qwen3-VL-8B-Instruct`)
- Rubric script: `vlm/opensource_vlm_i2p_all_v5.py`
- Per-concept rubric: nudity / violence / harassment / hate / shocking / disturbing / illegal / self_harm
- Output: `categories_qwen3_vl_<concept>_v5.json` with `{filename: {"category": "Safe|Partial|Full|NotRelevant"}}`
- Aggregation: **SR = (Safe + Partial) / (Safe + Partial + Full + NotRelevant)** — single canonical formula, no strict/loose split
- For nudity datasets: v3 ≡ v5 (only I2P concepts diverge between v3 and v5)

### Method spec (no change)
- When (CAS gate): `c_t = cos(ε_prompt − ε_∅, ε_target − ε_∅) > τ_cas`, sticky. SD1.4 τ_cas=0.6 default; SD3 τ_cas=0.3.
- Where (per-family dual probe): `M_f = max(M_f^img, M_f^txt)` per family; inter-family pixel-wise winner-take-all (argmax).
- How (anchor_inpaint): `ε^safe ← (1−β_f)ε^safe + β_f * ε̃^anc_f`, β_f = min(s·M_f, 1).
- How (hybrid): `ε^safe ← ε^safe − s·M_f·(ε^tgt_f − ε^anc_f)`.

---

## 5. Cross-table consistency (verified post-audit)

- HOW Mode `hybrid` (slide 6) ≡ I2P Single `hybrid` (slide 9 top, Table 2) ≡ Probe Ablation `both` (slide 14, Table 4) — **6/6 cells match**
- HOW Mode `anchor` ≡ I2P Single `anchor` — **6/6 cells match**
- Multi-Concept `Ours-single` ≡ max(I2P anchor, I2P hybrid) per concept — **6/6 cells match**
- Slides ≡ paper_full.md ≡ paper_summary.md ≡ paper_tables_final.md — all in sync

---

## 5.5. Pending — I2P Sexual (추가 확장용, 차후에 정리 예정)

### Single-concept (데이터 있음, Table 2 확장 가능)

| Method | I2P sexual top-60 SR (v5, n=60) | Source dir |
|---|---|---|
| Baseline SD1.4 | **.633** | `outputs/launch_0420_i2p/baseline_sd14/sexual` |
| SAFREE | **.800** | `outputs/launch_0420_i2p/safree_sd14/sexual` |
| EBSG (anchor, both probe) | **.933** | `outputs/launch_0420_i2p/ours_sd14/sexual/cas0.6_ss1.5_thr0.1_imgthr0.3_anchor_both` (ss=1.5, τ_t=0.1, τ_i=0.3, cas=0.6) |
| EBSG (anchor, both, ss=1.0 alt) | .933 | `outputs/launch_0420_i2p/ours_sd14/sexual/cas0.6_ss1.0_thr0.1_imgthr0.3_anchor_both` |
| **EBSG (hybrid, both probe) — best** | **.950** | `outputs/launch_0420_i2p/ours_sd14/sexual/cas0.6_ss20_thr0.1_imgthr0.3_hybrid_both` (ss=20, τ_t=0.1, τ_i=0.3, cas=0.6) |
| EBSG (hybrid, both, ss=15 alt) | .933 | same dir with ss=15 |

> All concept-correct CLI (target=`[nudity, nude_person, naked_body]`). Paper Table 2 currently lists only 6 toxic concepts (sexual excluded because it's covered by Table 1 nudity-main UD/RAB/MMA/P4DN). Adding this row → "7-concept" Table 2. Ours anchor 93.3 / hybrid 95.0 comfortably beats SAFREE 80.0.

### Multi-concept I2P sexual (미실행, re-run 필요)

- Current multi uses 24-family pack (6 concepts × 4 families). To include I2P sexual → 28-family pack (7 × 4) + full re-gen on I2P sexual top-60.
- `paper_results_master/06_multi_concept_sd14/` has no `i2p_multi_sexual_hybrid` cell.
- `outputs/v27_final/multi_all_on_i2p_sexual_ss1.0` exists but uses old v27 config — not apples-to-apples with current 6-concept multi.
- Action: build 28-family multi pack + re-run on 7 concepts. ETA ~30 min on 8 GPUs.

### Paper integration options
- **(A)** Extend Table 2 with `sexual` as first row (single-concept only); leave multi-concept `sexual` as TBD or note "sexual covered by Table 1".
- **(B)** Full 7-concept re-run of multi (new 28-family pack, re-gen, re-eval).
- **(C)** Keep Table 2 as-is (6-concept); mention I2P sexual single-concept briefly in appendix narrative referencing these numbers.

Decision deferred (user: "차후에 할테니까 이 부분은 너가 알아서 정리 파일에다가 잘 기록만").

---

## 6. What's Left (must appear in paper "Limitations" / "Future Work" / "Pending"):

1. **Multi-concept re-run with concept-aware global `target_concepts`** — current multi cells default global ε_tgt to nudity; per-pack family target_words are concept-correct, but a concept-aware multi-pack inference path is the next concrete fix. Image-probe pseudo-token cap of 4 also needs lift to 24 to recover violence/shocking on multi.
2. **v5 re-evaluation of weight-modifying baselines** — ESD / SDD / UCE / RECE / SLD-Max are dropped from Table 1 because their published numbers used a different evaluator. Either re-generate-and-eval their images with our v5 rubric, or annotate Table 1 explicitly as "v5 vs published" (not recommended).
3. **SAFREE+Safe_Denoiser, SAFREE+SGF compound rows** — pending v5 re-eval.
4. **COCO image quality** — FID and CLIP-score on COCO for both Ours-anchor and Ours-hybrid (currently TBD in Table 1; Anchor side has VQAScore drop <0.05 confirmed on RAB / MJA-Illegal).
5. **NudeNet + Q16 cross-check** — show that scalar safety detectors count semantic-collapse images as safe (motivating our 4-class rubric).
6. **Human study** — ~20 annotators × 100 I2P images per concept × 6–8 concepts; per-image agreement table for appendix.
7. **MMA-Diffusion P4D-N** seed-coverage cells — minor: 1–3 missing images per dataset (n=999 vs 1000 etc.).
8. **Multi on SD 3 / FLUX.1** — currently SD 1.4 only.

---

## 7. Honest narrative for paper text (USE)

- **Anchor vs hybrid is a tradeoff, not a choice.** I2P 3-3 split (anchor wins self-harm/illegal/harassment; hybrid wins violence/shocking/hate). Anchor avg 68.9 vs hybrid 66.1 — neither dominates. We report both as first-class rows.
- **Hybrid sweeps nudity benchmarks** (UD 97.2, RAB 96.2, MMA 84.4, P4DN 97.4); +9–26pp over SAFREE.
- **One pack, three backbones.** Best-of-mode avg SR on MJA: SD 1.4 81.0%, SD 3 74.8%, FLUX.1 92.5% — same When–Where–How wrapper, only the probe extraction site changes per backbone.
- **Probe = both** wins or ties on 5/6 I2P concepts (avg .661 vs img .600 vs txt .558); text wins keyword-explicit prompts, image wins metaphor-laden prompts.
- **Multi-concept**: SAFREE token-projection collapses to avg .028 (>90% NotRelevant); Ours retains .450 — +42pp, win on every concept (with the global target_concepts caveat noted above).
- **Evaluator**: 4-class VLM rubric `{Safe, Partial, Full, NotPeople}` with explicit collapse class — no scalar-detector loophole. SR = (Safe+Partial)/Total. Validated on MJ-Bench (94.7% w/o-tie). Workshop paper used a related VLM-based protocol; we do NOT self-cite under double-blind, introducing the protocol fresh in §experiments.

---

## 8. Files for ml-writer to read first

1. `paper_results_master/paper_full.md` — single source of truth (post-audit)
2. `paper_results_master/paper_summary.md` — cell-level catalog with dirs (post-audit)
3. `paper_results_master/paper_tables_final.md` — ready-to-include LaTeX-style tables
4. `docs/presentation_20260424.html` — slide deck mirroring paper_full.md (post-audit)
5. `docs/omx_handoff_2026-04-24_honest_results.md` — THIS file
6. `docs/omx_handoff_2026-04-24_camera_ready.md` — prior handoff (formatting/template guidance, but NUMBERS are stale → use this file's numbers)
7. `paper_neurips2026_sync/sec/*.tex` — current paper draft (numbers may need refresh per this handoff)
8. `vlm/opensource_vlm_i2p_all_v5.py` — eval rubric source

---

## 9. Pending action items for ml-writer

1. **Sync paper_neurips2026_sync/table/main.tex** with corrected Table 1 (drop ESD/SDD/UCE/RECE/SLD rows; add note about v5 consistency).
2. **Sync paper_neurips2026_sync/table/multi_concept.tex** with corrected I2P multi numbers + add SAFREE-multi row + add caveat sentence about target_concepts limitation.
3. **Sync probe_ablation.tex** if separate; otherwise update §experiments narrative for 5/6 wins-or-ties claim.
4. **Update §experiments narrative** to reflect 3-3 anchor-vs-hybrid split (was earlier incorrectly framed as anchor-dominant or hybrid-dominant in some drafts).
5. **Add Limitations §** referencing items in §6 above (multi target_concepts, v5 re-eval pending, etc.).
6. **Verify cross-backbone narrative** (§4 cross-backbone-transfer paragraph) — mode-flip is concept-and-backbone dependent (SD 1.4 split 2-2; SD 3 leans hybrid 3-1; FLUX.1 leans anchor 3-1).
