# 2026-04-08 Detailed Guide to Latest LLM Machine Unlearning Papers

## Executive summary
I prepared a beginner-friendly but technically serious guide to the most important **2025-2026** LLM machine unlearning papers, including **UAM**. The goal is to make the area understandable even for a general ML master's student while keeping enough detail for project planning.

### Main conclusion
The field's center of gravity has shifted from “can we force the model to forget?” toward four harder questions:
- can forgetting survive **relearning**,
- can it be **selective/localized**,
- does it also remove **reasoning-trace leakage**,
- and are our **evaluation protocols** strong enough to catch indirect or multilingual recovery?

## Paper-by-paper summary

### 1) Explainable LLM Unlearning through Reasoning (ICLR 2026)
- Source: https://openreview.net/forum?id=wec4qy2XIF
- Core message: answer-only forgetting is insufficient; unlearning should also reason about hidden reasoning paths.
- Why it matters: strong 2026 signal that reasoning-aware unlearning is becoming central.
- Limitation: very new; the exact setup is less battle-tested than 2025 papers.

### 2) Towards LLM Unlearning Resilient to Relearning Attacks (ICML 2025)
- Source: https://proceedings.mlr.press/v267/fan25e.html
- Core message: a model that forgets today but relearns tomorrow has not really unlearned.
- Why it matters: makes robustness to future tuning a first-class objective.
- Limitation: robustness is still hard to guarantee across every future update pattern.

### 3) Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning (ICML 2025)
- Source: https://proceedings.mlr.press/v267/wang25en.html
- Core message: forgetting should generalize across many future tuning environments, not just one known attack.
- Why it matters: pushes from attack-specific robustness to more general deployment robustness.
- Limitation: invariance assumptions may still miss real-world complexity.

### 4) Adaptive Localization of Knowledge Negation for Continual LLM Unlearning (ICML 2025)
- Source: https://proceedings.mlr.press/v267/wuerkaixi25a.html
- Core message: broad global updates damage utility; localize unlearning to the most relevant parts of the model.
- Why it matters: one of the best top-tier examples of selective/localized unlearning.
- Limitation: localization is imperfect because knowledge is distributed.

### 5) Tool Unlearning for Tool-Augmented LLMs (ICML 2025)
- Source: https://proceedings.mlr.press/v267/cheng25a.html
- Core message: sometimes the thing to forget is a capability or tool-use behavior, not a factual document.
- Why it matters: broadens the field toward agentic systems.
- Limitation: the setting is less standardized than document/fact unlearning.

### 6) Unlearning-Aware Minimization (UAM) (NeurIPS 2025)
- Source: https://neurips.cc/virtual/2025/poster/116406
- OpenReview: https://openreview.net/forum?id=e3abccbf80
- Core message: forgetting should be optimized together with retain preservation, not as a pure forget objective.
- Why it matters: excellent optimization principle and high-level design reference.
- Limitation: more general machine-unlearning than pure NLP.

### 7) Do LLMs Really Forget? (NeurIPS 2025)
- Source: https://neurips.cc/virtual/2025/poster/119349
- Core message: simple benchmarks miss correlated knowledge and confidence-based leakage.
- Why it matters: one of the strongest recent evaluation critiques.
- Limitation: evaluation papers alone do not fully solve the method problem.

### 8) Elastic Robust Unlearning of Specific Knowledge in Large Language Models (NeurIPS 2025)
- Source: https://openreview.net/forum?id=VrXjAfdwrN
- Core message: robustly remove specific knowledge while preserving utility.
- Why it matters: another signal that robust/specific forgetting is becoming standard.
- Limitation: less established than the most central 2025 main-track papers.

### 9) Reasoning Model Unlearning (EMNLP 2025)
- Source: https://aclanthology.org/2025.emnlp-main.220/
- Core message: unlearning must suppress reasoning traces, not just final answers.
- Why it matters: one of the most important NLP-specific recent papers.
- Limitation: reasoning trace evaluation is harder and more variable than standard QA.

### 10) A Fully Probabilistic Perspective on Large Language Model Unlearning (EMNLP 2025)
- Source: https://aclanthology.org/2025.emnlp-main.452/
- Core message: one-shot deterministic outputs can hide the fact that the target is still high-probability in the model.
- Why it matters: pushes the field toward better probability-aware evaluation.
- Limitation: more expensive and harder to operationalize than simple evaluation.

### 11) Learn and Unlearn: Addressing Misinformation in Multilingual LLMs (EMNLP 2025)
- Source: https://aclanthology.org/2025.emnlp-main.516/
- Core message: misinformation spreads across languages, and English-only unlearning is often insufficient.
- Why it matters: makes multilingual propagation a serious unlearning problem.
- Limitation: multilingual experiments are expensive and language coverage is never complete.

### 12) OBLIVIATE (EMNLP 2025)
- Source: https://aclanthology.org/2025.emnlp-main.183/
- Core message: practical LoRA-based framework balancing forget quality, utility, and robustness.
- Why it matters: strong pragmatic baseline.
- Limitation: framework complexity makes it harder to isolate exactly why it works.

### 13) Not Every Token Needs Forgetting (Findings of EMNLP 2025)
- Source: https://aclanthology.org/2025.findings-emnlp.96/
- Core message: only some tokens in a forget document are truly responsible for the target knowledge.
- Why it matters: very clear selective-unlearning intuition.
- Limitation: token-importance estimation is imperfect.

### 14) UIPE (Findings of EMNLP 2025)
- Source: https://aclanthology.org/2025.findings-emnlp.1374/
- Core message: forgetting only the exact target is not enough; correlated knowledge can reconstruct it.
- Why it matters: strong bridge between direct forgetting and indirect leakage control.
- Limitation: correlated-knowledge scope can explode and hurt utility.

## Overall synthesis
These papers together suggest that the field is moving toward:
- selective/localized intervention,
- relearning robustness,
- reasoning-aware unlearning,
- multilingual-aware evaluation,
- stronger probability-aware and correlation-aware metrics.

## Best project direction for us
### Preferred
**Evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**

### Backup
**Selective representation unlearning with robustness-to-relearning evaluation**
