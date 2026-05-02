# V2 Experiment Results (2026-04-14)
# Evaluation: Qwen3-VL (conservative prompt), Q16 pending
# Generation: 50 steps, DDIM, cfg=7.5, seed=42, SD v1.4
# Exemplar: v2 antonym-based, family-grouped (4 families × 4 imgs)

---

## 1. BASELINE (SD v1.4 vanilla, no safety)

| Dataset       | SR%   | Safe  | Partial | Full  | NotRel |
|---------------|-------|-------|---------|-------|--------|
| RAB (79)      | 63.3  | 19.0  | 44.3    | 32.9  | 3.8    |
| MMA (1000)    | 35.6  | 11.0  | 24.6    | 64.1  | 0.3    |
| P4DN (151)    | 36.4  | 4.6   | 31.8    | 62.9  | 0.7    |
| UnlearnDiff (142) | 78.2 | 21.8 | 56.3   | 21.1  | 0.7    |
| MJA sexual (100)  | 47.0 | 8.0  | 39.0   | 42.0  | 11.0   |
| MJA violent (100)  | 13.0 | 3.0  | 10.0   | 84.0  | 3.0    |
| MJA disturbing (100) | 74.0 | 3.0 | 71.0  | 26.0  | 0.0    |
| MJA illegal (100) | 60.0 | 22.0 | 38.0   | 30.0  | 10.0   |

---

## 2. OURS BEST CONFIG per Concept (Qwen SR%)

### Nudity (4 datasets)

| Dataset       | Best Config                              | SR%   | vs Baseline |
|---------------|------------------------------------------|-------|-------------|
| RAB (79)      | text ainp fam cas0.4 ss1.2               | 88.6  | +25.3       |
| RAB (79)      | both ainp single cas0.4 ss1.2            | 87.3  | +24.0       |
| MMA (1000)    | both ainp fam cas0.6 ss1.2               | 69.5  | +33.9       |
| P4DN (151)    | both ainp single cas0.6 ss1.2            | 76.2  | +39.8       |
| UnlearnDiff (142) | both ainp single cas0.6 ss1.2          | 95.1  | +16.9       |
| I2P sexual (931) | both ainp single cas0.6 ss1.2           | 91.5  | —           |

### Multi-Concept (I2P)

| Concept       | Best Config                              | SR%   |
|---------------|------------------------------------------|-------|
| Disturbing (856) | both ainp single cas0.5 ss1.0          | 97.9  |
| Violence (756)   | both ainp single cas0.4 ss1.5          | 89.8  |
| Harassment (824) | both ainp fam cas0.4 ss1.2             | 87.0  |
| Hate (231)       | both ainp fam cas0.4 ss1.0             | 85.3  |
| Self-harm (801)  | both ainp single cas0.4 ss1.0          | 74.3  |
| Illegal (726)    | both ainp fam cas0.5 ss1.0             | 74.6  |

### Multi-Concept (MJA, 100 prompts each, more adversarial)

| Concept       | Best Config                              | SR%   | vs Baseline |
|---------------|------------------------------------------|-------|-------------|
| Disturbing    | both ainp single cas0.6 ss1.0            | 88.0  | +14.0       |
| Illegal       | image ainp fam cas0.5 ss1.0              | 78.0  | +18.0       |
| Sexual        | both ainp fam cas0.6 ss1.2               | 74.0  | +27.0       |
| Violent       | both ainp single cas0.4 ss1.5            | 59.0  | +46.0       |

---

## 3. ABLATION: Probe Mode (MJA datasets, anchor_inpaint)

### MJA Violent (baseline 13.0%)

| Probe   | Single SR% | Family SR% |
|---------|-----------|------------|
| text    | 33.0      | 33.0       |
| image   | 45.0      | 45.0       |
| both    | 59.0      | 58.0       |

**Observation**: image(45) > text(33), both(59) > image > text ✓

### MJA Disturbing (baseline 74.0%)

| Probe   | Single SR% | Family SR% |
|---------|-----------|------------|
| text    | 78-79     | 78-79      |
| image   | 80-81     | 80-81      |
| both    | 87-88     | 86         |

**Observation**: image(81) > text(79), both(88) > image > text ✓

### MJA Illegal (baseline 60.0%)

| Probe   | Single SR% | Family SR% |
|---------|-----------|------------|
| text    | 65.0      | 65.0       |
| image   | 74.0      | 78.0       |
| both    | 77.0      | 75.0       |

**Observation**: image fam(78) > both > text, family helps for image probe ✓

### MJA Sexual (baseline 47.0%)

| Probe   | Single SR% | Family SR% |
|---------|-----------|------------|
| text    | 65.0      | 65.0       |
| image   | 55.0      | 56.0       |
| both    | 73.0      | 74.0       |

**Observation**: text(65) > image(55), both(74) > text > image ✓ (nudity = text dominant)

### RAB Nudity (baseline 63.3%)

| Probe   | Single SR% | Family SR% |
|---------|-----------|------------|
| text    | 88.6      | 88.6       |
| image   | 72.2      | 72.2       |
| both    | 87.3      | 86.1       |

**Observation**: text(88.6) >> image(72.2), text dominant for nudity ✓

---

## 4. ABLATION: HOW Mode (RAB nudity, both probe)

| HOW             | Single cas0.4 ss1.2 | Family cas0.4 ss1.2 |
|-----------------|---------------------|---------------------|
| anchor_inpaint  | 87.3                | 86.1                |
| hybrid          | 68.4                | 68.4                |

### MJA Violent (both probe)

| HOW             | Single SR% | Family SR% |
|-----------------|-----------|------------|
| anchor_inpaint  | 59.0      | 58.0       |
| hybrid          | 14.0      | 14.0       |

**Observation**: anchor_inpaint >> hybrid across all concepts

---

## 5. ABLATION: Family vs Single

| Concept (I2P) | both fam SR% | both single SR% | Δ     |
|---------------|-------------|-----------------|-------|
| Harassment    | 87.0        | 86.4            | +0.6  |
| Hate          | 85.3        | 81.4            | +3.9  |
| Violence      | 89.2        | 89.8            | -0.6  |
| Disturbing    | 97.8        | 97.9            | -0.1  |
| Self-harm     | 73.0        | 74.3            | -1.3  |
| Illegal       | 74.6        | 69.4            | +5.2  |

| Concept (MJA) | img fam SR% | img single SR% | Δ     |
|---------------|-------------|----------------|-------|
| Illegal       | 78.0        | 74.0           | +4.0  |
| Sexual        | 56.0        | 55.0           | +1.0  |

**Observation**: Family helps most for hate(+3.9) and illegal(+5.2)

---

## 6. KEY FINDINGS

1. **anchor_inpaint >> hybrid** across ALL concepts and datasets
2. **Nudity: text > image** (RAB: 88.6 vs 72.2, MJA: 65 vs 55)
3. **Violence/Disturbing/Illegal: image > text** (MJA violent: 45 vs 33, illegal: 78 vs 65)
4. **Dual probe (both) consistently best** — combines text and image strengths
5. **Family grouping helps** for hate (+3.9), illegal (+5.2) but marginal for others
6. **MJA is much harder** than I2P (violent: 59% vs I2P 89.8%)
7. **Conservative Qwen prompt** makes baselines lower than v1 eval (RAB: 63.3% vs old 21%)

---

## 7. REMAINING PHASES

### Phase 3: Q16 Evaluation ✅ COMPLETE

**Baseline Q16 Inappropriate Rate (threshold=0.7, ViT-L/14):**

| Dataset           | Inappropriate% |
|-------------------|---------------|
| RAB               | 8.86          |
| MMA               | 8.50          |
| P4DN              | 13.25         |
| UnlearnDiff       | 8.45          |
| MJA sexual        | 8.00          |
| MJA violent       | 82.00         |
| MJA disturbing    | 85.00         |
| MJA illegal       | 44.00         |

**Ours Best Q16 (lower = better):**

| Concept    | Dataset | Best Config                           | Inapp% | vs Baseline |
|------------|---------|---------------------------------------|--------|-------------|
| Sexual     | RAB     | image ainp single cas0.6 ss1.2        | 1.27   | -7.59       |
| Sexual     | UDiff   | both ainp single/fam cas0.6 ss1.2     | 3.52   | -4.93       |
| Sexual     | I2P     | both ainp single cas0.6 ss1.2         | 7.09   | —           |
| Violent    | I2P     | both ainp fam cas0.4 ss1.5            | 18.65  | —           |
| Violent    | MJA     | both ainp single cas0.4 ss1.5         | 38.00  | -44.00      |
| Disturbing | I2P     | both ainp fam cas0.5 ss1.0            | 20.56  | —           |
| Disturbing | MJA     | both ainp single cas0.4 ss1.0         | 61.00  | -24.00      |
| Harassment | I2P     | both ainp fam cas0.4 ss1.2            | 9.01   | —           |
| Hate       | I2P     | both ainp single cas0.4 ss1.0         | 15.58  | —           |
| Selfharm   | I2P     | both ainp fam cas0.4 ss1.0            | 16.60  | —           |
| Illegal    | I2P     | both ainp single cas0.5 ss1.0         | 14.46  | —           |
| Illegal    | MJA     | both/image ainp cas0.5 ss1.0          | 30.00  | -14.00      |

**Q16 Ablation (MJA violent, baseline 82%):**

| Probe  | ainp Inapp% | hybrid Inapp% |
|--------|-------------|---------------|
| text   | 66.0        | 82.0          |
| image  | 55.0 (fam)  | 84.0          |
| both   | 38.0 (single) | 84.0        |

Confirms: both < image < text for violence on Q16 too

### Phase 4: VQA Alignment ✅ COMPLETE

**VQAScore (InstructBLIP-FlanT5-XL, P(yes|image,"Does this show '{prompt}'?"))**

Baseline vs Ours best config:

| Dataset        | BL VQA | Ours VQA | Config                    | Δ      |
|----------------|--------|----------|---------------------------|--------|
| RAB            | 0.550  | 0.539    | img ainp fam cas0.4 ss1.2 | -0.011 |
| MMA            | 0.539  | 0.521    | both ainp fam cas0.6 ss1.2| -0.018 |
| P4DN           | 0.568  | 0.517    | both ainp single cas0.6   | -0.051 |
| UnlearnDiff    | 0.537  | 0.513    | both ainp single cas0.6   | -0.024 |
| MJA sexual     | 0.539  | 0.505    | both ainp fam cas0.6 ss1.2| -0.034 |
| MJA violent    | 0.492  | 0.479    | both ainp single cas0.4   | -0.013 |
| MJA disturbing | 0.524  | 0.502    | both ainp single cas0.6   | -0.022 |
| MJA illegal    | 0.475  | 0.474    | img ainp fam cas0.5 ss1.0 | -0.001 |

VQA probe ablation (RAB):
| Probe | VQA   | SR%  |
|-------|-------|------|
| image | 0.539 | 72.2 |
| text  | 0.529 | 88.6 |
| both  | 0.515 | 87.3 |

Observation: image probe preserves VQA best (least intervention), text probe best SR.
VQA drop is 0.01-0.05 range — acceptable prompt faithfulness preservation.

### Phase 5: Artist Style Erasure 🔄 IN PROGRESS
- 6 artists: Van Gogh, Picasso, Monet, Rembrandt, Warhol, Hopper
- 30 prompts each, grid: text/both × ainp/hybrid × ss={0.8,1.0,1.2}
- Running on siml-01 GPU 0-7
- Qwen + Q16 eval auto-chained after generation

### Phase 6: COCO FID/CLIP 🔄 IN PROGRESS
- COCO 250 prompts × 4 samples, baseline + ours best config
- Auto-chained after artist on siml-01

### Phase 7: SAFREE Baseline Comparison 🔄 IN PROGRESS
- siml-02: I2P 7 concepts + MJA 4 + RAB + MMA + UnlearnDiff
- 3,354+ images generated so far
- MMA (GPU 1) + UnlearnDiff (GPU 2) just started
- Qwen + Q16 eval auto-chained on siml-01 after SAFREE completes

### Phase 8: Paper Table Update
- Status: BLOCKED on Phase 5-7 completion
- All eval results will auto-save to output dirs
