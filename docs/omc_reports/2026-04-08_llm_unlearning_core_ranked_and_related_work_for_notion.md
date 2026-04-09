# 2026-04-08 Core Latest Papers Ranked + Related-Work Draft

## Executive summary
I resumed the latest-paper unlearning job and converted the broad 2025-2026 survey into two research-useful artifacts:
1. a **ranked list of the most important papers for our likely project**, and
2. a **paper-ready related-work draft** in prose.

The goal was not to rank the entire field by citation count, but to identify which latest papers matter most for **our probable EMNLP direction**.

## Ranked core reading list

### Must read now
1. **Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills** (EMNLP 2025)
   - Why: strongest direct precedent for reasoning-aware / trace-aware forgetting.
2. **A Fully Probabilistic Perspective on Large Language Model Unlearning: Evaluation and Optimization** (EMNLP 2025)
   - Why: strongest recent argument that ordinary output-level evaluation is too weak.
3. **Towards LLM Unlearning Resilient to Relearning Attacks** (ICML 2025)
   - Why: puts relearning robustness at the center of the problem.
4. **Adaptive Localization of Knowledge Negation for Continual LLM Unlearning** (ICML 2025)
   - Why: strongest top-tier localization paper aligned with selective intervention.
5. **Unlearning-Aware Minimization (UAM)** (NeurIPS 2025)
   - Why: best retain-aware optimization principle and conceptual backbone.

### Should read soon
6. **Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness** (NeurIPS 2025)
   - Why: exposes indirect leakage and benchmark weakness.
7. **Learn and Unlearn: Addressing Misinformation in Multilingual LLMs** (EMNLP 2025)
   - Why: shows multilingual propagation is a real unlearning issue.
8. **Not Every Token Needs Forgetting** (Findings of EMNLP 2025)
   - Why: one of the clearest selective-unlearning formulations in NLP.

### Useful secondary papers
- UIPE
- OBLIVIATE
- Explainable LLM Unlearning through Reasoning (ICLR 2026)
- Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning
- Tool Unlearning for Tool-Augmented LLMs

## Related-work draft summary
Recent 2025-2026 LLM unlearning work clusters into four themes.

### 1. Robustness / relearning resistance
These papers ask whether forgetting survives later adaptation, not just whether it appears immediately after the forget update. The ICML 2025 sharpness-aware paper and the invariance-based ICML 2025 paper both argue that brittle forgetting is not enough. UAM strengthens this line by making the forget/retain tradeoff explicit at the optimization level.

### 2. Localization / selective unlearning
These papers argue that broad global updates are too destructive. Adaptive Localization makes the case for concentrating changes on the parts of the model most relevant to the target. Not Every Token Needs Forgetting and UIPE bring the same idea into NLP form: not all tokens and not all correlated knowledge should be treated equally.

### 3. Evaluation faithfulness
The fully probabilistic EMNLP 2025 paper and the NeurIPS 2025 evaluation paper both show that standard unlearning metrics can overstate progress. The model may still retain probability mass, correlated knowledge, or high confidence even when surface outputs look safe.

### 4. Reasoning and multilingual extensions
Reasoning Model Unlearning shows that final-answer suppression is not enough once reasoning traces matter. Learn and Unlearn shows that forgetting in one language does not guarantee forgetting across languages. These settings are especially important for an EMNLP-facing paper.

## Current best-fit paper direction
### Preferred direction
**Evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**

Why this remains strongest:
- the latest evaluation papers show the metrics are still weak,
- the latest method papers move toward localization and robustness,
- NLP-facing settings like reasoning and multilinguality remain underexplored enough to support a strong EMNLP story.

### Backup direction
**Selective representation unlearning with robustness-to-relearning evaluation**

Why:
- combines the cleanest optimization and localization ideas from the top-tier 2025 papers,
- still leaves room for a serious language-centric evaluation contribution.
