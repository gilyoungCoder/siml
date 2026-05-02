# 2026-04-08 LLM Unlearning Ranked Core Papers + Related Work Draft

## Executive summary
I resumed the literature job and converted the large survey into two paper-writing artifacts:
1. a **ranked core-paper list** for our project, and
2. a **paper-ready related-work draft** organized around the latest 2025-2026 themes.

## Ranked core papers for our project
### Must-read now
1. **Reasoning Model Unlearning** (EMNLP 2025)
   - Why: strongest evidence that answer-only forgetting is insufficient once reasoning traces exist.
2. **A Fully Probabilistic Perspective on Large Language Model Unlearning** (EMNLP 2025)
   - Why: shows deterministic output metrics can overstate forgetting.
3. **Towards LLM Unlearning Resilient to Relearning Attacks** (ICML 2025)
   - Why: forgetting that fails after later tuning is not robust enough.
4. **Adaptive Localization of Knowledge Negation for Continual LLM Unlearning** (ICML 2025)
   - Why: strongest top-tier localization/selective-update paper for our likely method direction.
5. **Unlearning-Aware Minimization (UAM)** (NeurIPS 2025)
   - Why: best retain-aware optimization lens; crucial conceptually even if not purely NLP.

### Should-read soon
6. **Do LLMs Really Forget?** (NeurIPS 2025)
   - Why: exposes correlated-knowledge and confidence leakage.
7. **Learn and Unlearn** (EMNLP 2025)
   - Why: multilingual propagation makes English-only forgetting insufficient.
8. **Not Every Token Needs Forgetting** (Findings of EMNLP 2025)
   - Why: clean selective-unlearning formulation for utility preservation.

## Related-work draft synthesis
The latest literature can be cleanly organized into four themes.

### 1. Robustness and relearning resistance
The ICML 2025 sharpness-aware paper and the ICML 2025 invariance-based paper show that post-unlearning forgetting must survive later adaptation, not just look good immediately after the update. UAM contributes the optimization perspective that forgetting should be trained together with retain preservation rather than as a pure forget objective.

### 2. Localization and selective unlearning
Adaptive Localization, Selective Unlearning, and UIPE collectively argue that coarse global forgetting is too destructive. The field is moving toward changing only the parts of the model or input that are most responsible for the target knowledge, while also handling correlated knowledge that could reconstruct the target.

### 3. Evaluation faithfulness
The fully probabilistic EMNLP 2025 paper and the NeurIPS 2025 “Do LLMs Really Forget?” paper both show that existing evaluation often overestimates progress. One-shot outputs may hide latent target probability, and direct-target removal may still leave correlated evidence or high confidence intact.

### 4. Reasoning and multilingual extensions
Reasoning Model Unlearning and the ICLR 2026 reasoning-focused paper show that final-answer suppression is not sufficient for reasoning-capable models. Learn and Unlearn adds the multilingual dimension, demonstrating that misinformation can survive or reappear across languages.

## Best-fit direction for us
The strongest gap still appears to be:
**evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**.

Why this direction is attractive:
- robustness papers show forgetting must survive future updates,
- localization papers show broad forgetting is too blunt,
- evaluation papers show current metrics are too weak,
- reasoning and multilingual papers show answer-only English-only forgetting is not enough.

## Files added locally
- `emnlp/related_work/core_latest_papers_ranked.md`
- `emnlp/related_work/related_work_draft_latest.md`
