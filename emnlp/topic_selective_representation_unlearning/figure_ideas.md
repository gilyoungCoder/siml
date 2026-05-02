# Figure Ideas — SRU-MR

## Figure 1 — Problem setup overview
**Title:** Why one-shot forgetting is insufficient

### Panel A
- Standard unlearning benchmark: direct question -> suppressed answer

### Panel B
- Multi-turn recovery: direct question suppressed, but later turns with paraphrase/hint/correlated cue recover target knowledge

### Message
A model can look safe on turn 1 and still fail under dialogue continuation.

---

## Figure 2 — Method overview
**Title:** Selective Representation Unlearning for Multi-turn Recovery

### Flow
1. Input dialogue tokens
2. Hidden states across layers
3. Forget-relevance scorer produces token × layer mask
4. Masked forgetting update applied through LoRA/adapters
5. Multi-turn recovery loss + retain loss + sparsity loss

### Message
Forget only where target knowledge is active, not globally.

---

## Figure 3 — Evaluation protocol
**Title:** Recovery-aware evaluation families

Show 4 dialogue families:
1. direct -> paraphrase
2. direct -> hint -> reveal
3. indirect entity bridge
4. summarize -> extract

Optional fifth:
5. reasoning scaffold or cross-lingual bridge

### Message
Evaluation should test multiple realistic recovery paths.

---

## Figure 4 — Main results
**Title:** Forget-retain-recovery tradeoff

Suggested plots:
- x-axis: utility retention
- y-axis: multi-turn recovery rate (lower is better)
- marker color/shape: direct forgetting success

Compare:
- NPO
- OBLIVIATE
- Adaptive Localization
- Selective token baseline
- SRU-MR

### Message
SRU-MR should move the frontier toward lower recovery at similar or better utility.

---

## Figure 5 — Ablation visualization
**Title:** Why selectivity matters

Bars or heatmaps for:
- global update
- token-only
- layer-only
- token × layer

### Message
Localization granularity changes the forget-retain-recovery balance.
