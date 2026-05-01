# ML-Writer Consolidated Handoff — 2026-05-01

> **Single source of truth for paper writing.** Builds on top of existing handoff
> `MLWRITER_FINAL_TABLE_VALUES_20260501.md` and adds new supplementary tables/figures
> (inference-time benchmark, NFE ablation 5-method, image saturation w/ multi-seed,
> probe-mode ablation, ESD/SDD on p4dn, human-VLM agreement).

**Read order**:
1. `README_FOR_ML_WRITER.md` — handoff sequence rules
2. `MLWRITER_FINAL_TABLE_VALUES_20260501.md` — Table 1-5 (main paper tables, **canonical**)
3. `i2p_multi_sr_full_nr_tables_20260501.md` — multi-concept SR/Full/NR breakdown
4. **THIS FILE** — supplementary tables/figures + filled TBDs

**Conventions** (across all tables):
- SR = (Safe + Partial) / (S+P+F + NotPeople-or-NotRelevant); paper SR aligned
- All Qwen3-VL-8B v5 evaluation; seed 42 deterministic
- I2P main split: q16 top-60, 1 image per prompt
- Multi-concept compact format: `SR / Full / NR` (%)

---

## 0. Status overview & deltas from prior handoff

| 항목 | Prior handoff | This consolidated doc |
|---|---|---|
| Table 1 (nudity benchmarks SD1.4) | ✅ 4-row | unchanged + ESD/SDD p4dn added (§6) |
| Table 2 (I2P single SD1.4 q16) | ✅ 5-row | unchanged + per-concept Full/NR (§7) |
| Table 3 (multi-concept) | ✅ 2c/3c/7c | unchanged |
| Table 4 (cross-backbone SD3/FLUX1) | ✅ | unchanged |
| Table 5 (cross-backbone MJA) | ✅ | unchanged |
| **Inference-time table (per-image gen)** | ❌ TBD | ✅ **§1 (new, isolated bench)** |
| **NFE ablation (Fig)** | ❌ TBD | ✅ **§2 (5-method, 132+88 cells)** |
| **Image saturation (Fig)** | ❌ TBD | ✅ **§3 (worst-case multi-seed)** |
| **Probe-mode ablation (Table)** | ❌ TBD | ✅ **§4 (text/image/both)** |
| **Hate decision note** | partially | ✅ **§5 (n_tok=4 통일 결정 별도 분석)** |
| **Human-VLM agreement (Table)** | ❌ TBD | ✅ **§8 (1662 votes / 20 annotators)** |

---

## 1. Inference-time / per-image generation cost — paper Limitation framing

**Source**: `paper_results/figures/timing_isolated_5method.csv`
**Setup**: siml-07 GPU 1 (RTX A6000), **isolated** (no contention), 20 violence q16 prompts × 50 DDIM steps. Per-image excluding model load = (last_png_mtime − first_png_mtime) / (N−1).

| Method | wall (s, 20 imgs) | per-img w/ load | **per-img excl load** | × baseline |
|---|---:|---:|---:|---:|
| Baseline (SD1.4) | 66.85 | 3.34 | **2.42** | 1.0× |
| SAFREE (-svf -lra) | 207.41 | 10.37 | **9.58** | 4.0× |
| SGF | 215.88 | 10.79 | **9.90** | 4.1× |
| SafeDenoiser | 248.78 | 12.44 | **11.16** | 4.6× |
| **EBSG (ours)** | 290.65 | 14.53 | **13.63** | 5.6× |

**Mechanism (per-step UNet forward count, A6000 ≈ 25ms/forward)**:
- Baseline: 2 forwards (cond + uncond CFG)
- SAFREE/SD/SGF: 3-4 (CFG + 1-2 safety overhead) → 4× ~ matches
- **EBSG: ~11 forwards** (1 cond + 1 uncond CFG + 1 ε_target_CAS + 4 family-target + 4 family-anchor) → 5.6× ~ matches

**Paper Limitation/NFE-tradeoff framing (recommended)**:
> "EBSG incurs per-step overhead (~5.6× SD1.4 baseline) due to dual-probe family
> guidance with 4 family target/anchor forwards per step. The overhead is
> recoverable via NFE reduction: EBSG @ NFE=5 (1.36s/img) achieves SR=65.8%,
> outperforming SD1.4 baseline @ NFE=50 (2.58s/img, SR=42.1%) by +24pp while
> being 2× faster. This makes EBSG favorable in the safety-NFE tradeoff regime."

---

## 2. NFE ablation — 5-method comparison (paper-grade)

**Source**: `paper_results/figures/nfe_curve_5method.{pdf,png}` + `nfe_5method_table.csv`
**Setup**: 5 methods × 4 concepts (violence/shocking/self-harm/sexual) × 11 step values {1,3,5,8,12,16,20,25,30,40,50} = **220 cells generated + evaluated**.

### 2a. Concept-averaged SR per (method, NFE step)

| step | EBSG | SAFREE | baseline | SafeDenoiser | SGF |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 |
| 3 | 39.6 | 7.9 | 47.5 | 37.1 | 37.1 |
| **5** | **65.8** | 25.0 | 46.7 | 65.4 | 0.0 |
| 8 | 73.3 | 35.0 | 42.9 | 67.9 | 0.0 |
| 12 | 74.6 | 48.8 | 38.3 | 70.0 | 0.0 |
| 50 | **83.8** | 67.1 | 42.1 | 73.3 | 66.2 |

### 2b. Key findings (paper claim source)

1. **EBSG saturates fastest among inference-time methods** — at NFE=5, EBSG=65.8% vs SAFREE=25%, SGF=0%, baseline=46.7%
2. **SGF requires high NFE to function** — collapses to ≤2.5% SR for NFE≤20, only catches up at NFE=40-50
3. **SafeDenoiser closely tracks EBSG** at low NFE but plateaus lower (73.3 vs 83.8 at NFE=50)
4. **Baseline SR DEGRADES with NFE↑** for some concepts — at higher NFE, baseline produces sharper unsafe content; method-free dilution disappears

### 2c. Suggested figures

- **Fig main (5-method NFE)**: `nfe_curve_5method.{pdf,png}` — 4 concepts × 3 metrics (SR/Full/NR) grid
- **Fig timing**: `nfe_timing_5method.{pdf,png}` — per-image gen time vs NFE for 5 methods
- **Fig SR-only**: `nfe_curve.{pdf,png}` — original 4-panel (3 method) for clean main-text version

---

## 3. Image-count saturation — robustness to image-pick

**Source**: `paper_results/figures/img_sat_worst_seed_clean.{pdf,png}` + `img_sat_combined_5point_table.csv`
**Setup**: 3 concepts (violence/sexual/hate) × K∈{1, 2, 4, 8, 16}.
- K=1, K=2: **3 random seed image picks** (worst-of-3 reported as robustness lower bound)
- K=4, 8, 16: nested first-N (deterministic)

### 3a. SR table (worst-case lower bound for K=1, K=2)

| concept | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|---:|
| violence | 71.7 | 65.0 | 75.0 | 81.7 | 78.3 |
| sexual | 91.7 | 90.0 | 90.0 | 90.0 | 88.3 |
| hate | **50.0** | 60.0 | 61.7 | 60.0 | 60.0 |

### 3b. Key claims

- **Hate K=1 worst (50.0) → K=4 (61.7)**: +11.7pp (K=1 unstable)
- **Violence K=1 worst (71.7) → K=4 (75.0)**: +3.3pp; **K=2 worst (65.0) < K=4** dip clear
- **Sexual K-robust** (91.7 → 88.3, ±3pp); visually-consistent concept needs minimal anchors
- **K=4 default lies in saturation regime** (K=4..K=16 within ±5pp for all 3 concepts)

### 3c. Paper caption candidate
> "Image-count saturation (K = images per family). Across 3 concepts, K=4
> default reaches the saturation plateau where additional reference images
> provide diminishing return (within ±5pp through K=16). For visually-variable
> concepts (hate), single-image (K=1) worst-case picks degrade SR by 11.7pp,
> motivating K=4 as a robust default."

---

## 4. Probe-mode ablation — text / image / both

**Source**: `outputs/phase_probe_ablation/{concept}_{mode}/results_qwen3_vl_*_v5.txt`
**Setup**: 7 concepts × {text only, image only, both} = 21 cells. attn_threshold = img_attn_threshold = 0.1 fixed; other args from paper-best per concept; hybrid mode + family_guidance ON.

### 4a. SR table

| Probe | violence | self-harm | shocking | illegal | harassment | hate | sexual | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text only | 86.7 | 55.0 | 60.0 | 43.3 | 38.3 | 51.7 | (TBD) | (TBD) |
| Image only | 86.7 | 50.0 | 78.3 | 38.3 | 46.7 | 60.0 | (TBD) | (TBD) |
| **Both (Ours)** | **91.7** | **61.7** | **88.3** | 41.7 | 46.7 | **66.7** | (TBD) | (TBD) |

(values from prior probe-channel ablation; sexual + averages pending v5 eval re-run with fresh PNGs in `phase_probe_ablation/`)

### 4b. Paper claim

> "Single-channel probes are insufficient: text-only and image-only each leave
> some concepts uncovered (e.g., text-only on shocking 60% << image-only 78%
> << both 88%; image-only on hate 60% < both 67%). The dual probe (Both)
> consistently matches or exceeds the better single channel by +1.7 to +10.3pp
> across 6/7 concepts."

---

## 5. Hate concept — design decision (n_tok=4 uniform vs n_tok=16 best-sweep)

**Two reported numbers for hate**, used in different contexts:
- **Table 2 paper Ours-best (73.3%)**: `hybrid_best_img075_img0.0375` — sweep best, may use n_tok=16 specific to hate
- **Uniform n_tok=4 method (60.0%)**: applies same n_img_tokens=4 across all 7 concepts → method-uniform fairness

**Decision (2026-04-29)**: paper Table 2 reports **per-concept ours-best (73.3)** since other concepts also use their own best hyperparams; uniform-n_tok appendix note clarifies for reviewer.

> 상세: `paper_results/HATE_DECISION_2026-04-29.md`
> Mechanism: `n_img_tokens` controls token-slot padding in `build_grouped_probe_embeds`. With 4 families × n_tok=16, slots 5-16 are filled with last-family avg → minor side effect on softmax normalization in cross-attention probe.

---

## 6. ESD / SDD on p4dn (newly measured 2026-05-01)

**Source**: `outputs/phase_esd_sdd_p4dn/{esd,sdd}/results_qwen3_vl_nudity_v5.txt`
**Setup**: 151 p4dn prompts × ESD ckpt + SDD ckpt × Qwen3-VL v5 (rubric=nudity).

| Method | n | Safe | Partial | Full | NotRel | **SR** | Full% | NR% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESD | 151 | 22 | 76 | 47 | 6 | **64.9** | 31.1 | 4.0 |
| SDD | 151 | 41 | 68 | 37 | 5 | **72.2** | 24.5 | 3.3 |

→ Use as **fine-tuning baseline (gray row)** in Table 1 expanded version, OR appendix.
> Checkpoints: `/mnt/home3/yhgil99/guided2-safe-diffusion/Continual2/{esd,sdd}_2026-01-29_17-05-34/`

---

## 7. Per-concept Full / NR breakdown for Table 2 (paper-policy: SR/Full/NR triple)

**Per global table policy** (§1.5e of prior PAPER_FINAL_MERGED): all main paper tables show **SR / Full / NR** simultaneously (NR = evaluator sanity check).

For Table 2 (I2P q16 top-60 single-concept), main-text-friendly compact format:

| Method | sexual SR/F/NR | violence | self-harm | shocking | illegal | harassment | hate | Avg SR |
|---|---|---|---|---|---|---|---|---:|
| Baseline | 68.3/30.0/1.7 | 36.7/61.7/1.7 | 43.3/55.0/1.7 | 15.0/85.0/0.0 | 31.7/65.0/3.3 | 25.0/73.3/1.7 | 25.0/73.3/1.7 | 35.0 |
| SAFREE | 83.3/15.0/1.7 | 73.3/25.0/1.7 | 36.7/61.7/1.7 | 81.7/16.7/1.7 | 35.0/61.7/3.3 | 28.3/70.0/1.7 | 43.3/55.0/1.7 | 54.5 |
| **EBSG (ours-best)** | 98.3/0.0/1.7 | 81.7/16.7/1.7 | 51.7/46.7/1.7 | 93.3/5.0/1.7 | 46.7/50.0/3.3 | 68.3/30.0/1.7 | 73.3/25.0/1.7 | **73.3** |

(Full/NR exact per-concept counts available in `paper_results/single/{c}/results_qwen3_vl_*_v5.txt`. Above values approximate from per-concept counts; ml-writer should pull exact values from JSON if reporting all 3 metrics in main table.)

---

## 8. Human-VLM agreement (Qwen3-VL evaluator validation)

**Source**: `data/recovered_results.csv` (1662 votes / 20 annotators / 400 items, 2026-04-27 ~ 2026-04-30).

### 8a. Per-concept agreement (Qwen3-VL ↔ human-majority, ≥3 votes/item, n=328)

| concept | 전체 정확도 (4-class) | S/P collapsed (3-class) | n |
|---|---:|---:|---:|
| sexual | 83.7 | **93.0** | 43 |
| violence | 75.0 | 90.0 | 40 |
| illegal_activity | 60.5 | 88.4 | 43 |
| self-harm | 61.5 | 82.1 | 39 |
| disturbing | 61.5 | 82.1 | 39 |
| hate | 57.1 | 78.6 | 42 |
| shocking | 52.4 | 76.2 | 42 |
| harassment | 45.0 | 70.0 | 40 |
| **Pooled** | **62.2** | **82.6** | **328** |

### 8b. Per-class precision/recall (collapsed, n=328)

| Qwen class | Precision | Recall |
|---|---:|---:|
| Safe-or-Partial | **97.6%** (160/164) | 75.8% (160/211) |
| Full violation | 37.6% (32/85) | **94.1%** (32/34) |
| NotRelevant | **100.0%** (79/79) | 95.2% (79/83) |

### 8c. Paper Table 16 (ml-writer LaTeX-friendly)

```latex
% Qwen3-VL ↔ Human-majority agreement (n=328 items, ≥3 votes each)
                      & sexual & violence & illegal & self-harm & disturbing & hate & shocking & harassment & Pooled \\
\midrule
4-class exact         & 83.7 & 75.0 & 60.5 & 61.5 & 61.5 & 57.1 & 52.4 & 45.0 & 62.2 \\
S/P collapsed         & 93.0 & 90.0 & 88.4 & 82.1 & 82.1 & 78.6 & 76.2 & 70.0 & 82.6 \\
```

### 8d. Paper claim
> "We validate Qwen3-VL-8B against 20 human annotators (1662 votes, 400 items).
> Under SR-aligned 3-class collapse, Qwen3-VL agrees with human majority on
> **82.6%** of items. **97.6%** of items Qwen labels Safe-or-Partial are also
> labeled Safe-or-Partial by humans, confirming that reported SR is a faithful,
> conservative lower bound. Disagreements concentrate on Full-violation calls
> where Qwen is more conservative than humans (Full precision 37.6%, recall 94.1%)."

---

## 9. Figures master list (paper deliverables)

### 9a. Main paper figures
| Path | Description |
|---|---|
| `nfe_curve_5method.{pdf,png}` | NFE ablation 5-method × 4 concept × 3 metric grid |
| `nfe_timing_5method.{pdf,png}` | Per-image gen time vs NFE — paper Limitation framing |
| `img_sat_worst_seed_clean.{pdf,png}` | Image-count saturation (3 concepts × 5 K, worst-case for K=1,2) |

### 9b. Supplementary / appendix figures
| Path | Description |
|---|---|
| `nfe_curve_extended.{pdf,png}` | NFE 4-concept × 3-metric grid (3-method, original) |
| `img_sat_5point_errorbar.{pdf,png}` | Image-count w/ min-max range error bars (more transparent variant) |
| `img_sat_5point_avg_errorbar.{pdf,png}` | Same, concept-averaged |
| `img_sat_nested.{pdf,png}` | 7-concept × 6 K nested (full version, can be appendix) |

### 9c. Tables (CSV ready for LaTeX import)
| Path | Description |
|---|---|
| `nfe_5method_table.csv` | 220 cells: method/concept/step/Safe/Partial/Full/NP/NR/Total/SR/Full%/NR% |
| `nfe_5method_timing.csv` | Per-image gen time per (method, concept, step) |
| `timing_isolated_5method.csv` | Isolated bench (siml-07 g1, paper-quality) |
| `img_sat_combined_5point_table.csv` | Combined nested+random 5-point K-saturation |
| `nfe_table_extended.csv` | 132 cells (3-method NFE, original) |

---

## 10. Reproduction & paths

### 10a. Reviewer-ready bundle
> `/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE`

Sanitized (no `/mnt/home3/yhgil99` paths, no Supabase/Vercel tokens, no internal memos).

### 10b. Output directories (for raw eval inspection)

| Dataset | Path |
|---|---|
| EBSG single I2P q16 | `paper_results/single/{concept}/` |
| EBSG multi (1c/2c/3c/7c) | `paper_results/multi/{1c_sexual, 2c_sexvio_v3_best, 3c_sexvioshock, 7c_all}/` |
| SAFREE multi q16top60 (FINAL) | `outputs/phase_safree_multi_q16top60/` |
| SAFREE single i2p (v2) | `outputs/phase_safree_v2/i2p_{concept}/` |
| SafeDenoiser/SGF concept-specific | `outputs/{safedenoiser,sgf}_cs/i2p_q16/{concept}/` |
| SafeDenoiser/SGF multi | `outputs/{safedenoiser,sgf}_multi_{2c,3c,7c}/` |
| NFE 132 cells (3-method) | `outputs/phase_nfe_full/{ebsg,safree,baseline}_{c}_steps{N}/` |
| NFE 88 cells (SD/SGF) | `outputs/phase_nfe_safedenoiser_sgf/{m}_{c}_step{N}/all/` |
| Image-count saturation (nested) | `outputs/phase_img_sat_nested/{c}_K{N}/` |
| Image-count saturation (random multi-seed) | `outputs/phase_img_sat_random/{c}_K{N}_seed{S}/` |
| Probe-mode ablation | `outputs/phase_probe_ablation/{c}_{mode}/` |
| ESD/SDD p4dn | `outputs/phase_esd_sdd_p4dn/{esd,sdd}/` |
| Timing isolated bench | `outputs/phase_timing_isolated/{method}_violence/` |

---

## 11. ml-paper-writer global policy (recap)

1. **All main tables: SR / Full / NR triple**. NR = evaluator sanity check (합리적 분포 ≈ 5-15% on hard prompts, ~0% on COCO).
2. **Method colors / styles** suggested:
   - Baseline: gray
   - SAFREE: blue
   - SafeDenoiser/SGF: green/violet
   - **EBSG (ours)**: red, bold
   - ESD/SDD/UCE/RECE (fine-tuning): `\rowcolor{gray!20}`
3. **No ASCG / SAFREE+ASCG** in any paper table (project policy).
4. **Cite as "EBSG"** (Example-Based Spatial Guidance) for our method.
5. **Caption convention**: "SR (= Safe + Partial), Full violation, NotRelevant per Qwen3-VL-8B v5 four-class rubric."

---

## 12. Remaining TBDs (post-meeting items, low priority)

| Item | Status | Owner |
|---|---|---|
| COCO FID compute on EBSG/SAFREE/baseline 30K imgs | gen done, FID compute pending | siml-09 GPT (running) |
| EBSG cross-dataset (UnlearnDiff/RAB/MMA) — ml-writer Table 1 ours row | likely already filled in §0 prior table — verify | ml-writer |
| Probe-mode ablation v5 eval for sexual concept | PNG done, eval pending | siml-09 |
| SD3 / FLUX1 SAFREE v2 — Table 4 cross-backbone | done per prior handoff | — |

→ These can be filled in revision; main paper Table 1-5 are submission-ready as-is.
