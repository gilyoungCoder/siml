# 2026-04-08 Plan: EMNLP LLM Unlearning

## 1. Immediate objective
Do **not** lock the final title too early. First, map the literature hard enough to answer:
- What is already saturated?
- What evaluation gaps are still clearly open?
- Which gap best matches an EMNLP-style contribution?
- Where can we reuse our repo's high-level "when/where/selective intervention" instinct in an NLP-native way?

## 2. Topic-selection rubric
We should prefer a topic that satisfies most of the following:
- **EMNLP fit**: strong language/data/evaluation angle, not just systems tuning.
- **Clear benchmark gap**: current papers miss a realistic failure mode.
- **Strong story**: one sentence for the problem, one sentence for why current methods fail.
- **Low-to-moderate implementation risk**: feasible with available open models/benchmarks.
- **Easy ablation story**: can compare against NPO / RMU / ULMR / self-generated-data baselines.
- **Generalizable insight**: not only “works on one dataset split”.

## 3. Research tracks to evaluate this week

### Track A — Compositional / mixed-context unlearning
**Core question:** can a model forget target knowledge **only where it should**, while retaining adjacent useful content in the same prompt / same conversation?

Why it looks promising:
- SEPS shows existing methods break badly when forget and retain content are mixed in one prompt.
- Existing benchmarks are still mostly single-query or loosely compositional.
- This is the cleanest NLP analogue of our repo's "when + where" philosophy.

Possible EMNLP contribution:
- a new benchmark for **mixed prompts + multi-turn dialogue**, and/or
- a method that localizes unlearning to target spans/facts/turns instead of whole-example suppression.

### Track B — Reasoning-trace-aware unlearning
**Core question:** how do we unlearn not just answers, but the hidden/revealed reasoning traces that still leak target knowledge?

Why it looks promising:
- R-TOFU and EMNLP 2025 Reasoning Model Unlearning both suggest answer-only metrics are insufficient.
- This area is fresh and visible.
- Strong narrative for EMNLP if framed as text generation + trace evaluation.

Risk:
- reasoning-model experiments can become expensive and unstable.

### Track C — Multilingual propagation-aware unlearning
**Core question:** if misinformation or sensitive facts are learned in one language, how should unlearning prevent cross-lingual re-emergence?

Why it looks promising:
- EMNLP 2025 multilingual paper already shows English-only unlearning is not enough.
- A stronger multilingual benchmark or intervention policy may still be open.

Risk:
- multilingual data creation/evaluation overhead.

### Track D — Robustness-to-relearning / downstream retuning
**Core question:** does “forgotten” knowledge come back after benign downstream finetuning, tool use, or decoding changes?

Why it looks promising:
- ILU and multiple evaluation papers show relearning/recovery is a major weakness.
- Strong practical angle.

Risk:
- may feel more ML-systems than EMNLP unless framed with language-centric analyses.

## 4. Provisional recommendation
If we had to bet early, the strongest two directions are:

### Preferred
**Track A: Compositional + multi-turn selective unlearning**
Reason:
- Most naturally extends our selective-intervention intuition.
- Strong EMNLP fit.
- Less crowded than plain TOFU-style single-turn forgetting.
- Can support either a benchmark paper or a benchmark+method paper.

### Backup
**Track B: Reasoning-trace-aware unlearning**
Reason:
- Very timely after R-TOFU / EMNLP 2025 reasoning unlearning.
- High upside if we can make the evaluation story sharper.

## 5. Overnight literature plan

### Phase 1 — Benchmark/evaluation map
Read and summarize:
- TOFU
- MUSE
- LUME
- SemEval-2025 Task 4
- Position: LLM Unlearning Benchmarks are Weak Measures of Progress
- UGBench / PERMU
- Harry Potter is Still Here! (LURK)
- R-TOFU
- Reasoning Model Unlearning
- SEPS

### Phase 2 — Method families map
Read and compare:
- Who's Harry Potter?
- NPO
- SimNPO
- ULMR
- UAM
- Reveal and Release
- ILU
- OBLIVIATE
- RMU / Adaptive RMU analysis papers

### Phase 3 — Narrow the paper thesis
For each candidate topic, answer:
1. What exactly is the failure mode?
2. Which existing paper comes closest?
3. What would be genuinely new?
4. What benchmark / dataset could prove it?
5. What is the minimal method needed for a convincing submission?

## 6. Deliverables to build next
- `related_work/paper_shortlist.md` — seed bibliography and why each paper matters.
- Individual paper notes under `related_work/`.
- `idea_synthesis.md` — 2 to 4 paper directions with concrete experiment plans.

## 7. Practical blocker
OMX `$team` is blocked by a dirty leader workspace.

If we want strict team runtime next:
- launch from a **clean auxiliary worktree** rooted at current HEAD,
- run team workers there,
- port resulting markdown docs back into this workspace.

