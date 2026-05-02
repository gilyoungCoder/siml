# Baseline Run Order — Selective Representation Unlearning for Multi-turn Recovery

## Goal
We want the fastest path to a convincing early result, not the most exhaustive run order.

---

## Phase 1 — evaluation harness first
### Priority 1
Build the multi-turn recovery evaluation wrappers around one primary benchmark.

Recommended choice:
- **TOFU + custom multi-turn wrappers**

Reason:
- controlled,
- easy to debug,
- strong comparability.

---

## Phase 2 — strongest baseline anchors
Run these first:

### 1. NPO
Why:
- standard optimization baseline,
- simple anchor for forget/retain.

### 2. OBLIVIATE
Why:
- strong practical 2025 baseline.

### 3. Adaptive Localization
Why:
- closest top-tier localization baseline.

### 4. Not Every Token Needs Forgetting
Why:
- selective forgetting comparison.

If time permits in phase 2:
### 5. Reasoning Model Unlearning or R²MU
Why:
- important if we later extend to reasoning-aware recovery.

---

## Phase 3 — our MVP
Run:
- LoRA-based SRU-MR
- contrastive forget prototype
- token × layer soft mask
- multi-turn consistency loss on

Compare against:
- NPO
- OBLIVIATE
- Adaptive Localization
- Selective token forgetting

---

## Phase 4 — ablations
1. no mask (global forgetting)
2. token-only mask
3. layer-only mask
4. token × layer mask
5. no multi-turn loss
6. no sparsity loss
7. mean vs contrastive prototype

---

## Phase 5 — secondary setting
Choose one only:
- reasoning leakage subset
or
- multilingual recovery subset

Recommendation:
- if the model backbone is reasoning-capable, choose reasoning
- otherwise choose multilingual only if evaluation data is easy to build

---

## Minimal publishable result
A minimal convincing story is:
1. stronger multi-turn recovery resistance than NPO/global forgetting,
2. better retain utility than global forgetting,
3. selective ablations show localization matters,
4. at least one secondary-axis result (reasoning or multilingual).
