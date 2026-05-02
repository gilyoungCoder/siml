# 2026-04-08 LLM Unlearning Latest Top-Venue Sweep

## Executive summary
I prioritized **2025-2026** LLM machine unlearning papers with an emphasis on **ICLR, ICML, NeurIPS**, while including several **EMNLP/ACL 2025** papers because our target submission is still NLP-facing.

### Main conclusion
The active frontier is no longer just “forget-loss on TOFU.” The field has moved toward:
- robustness to relearning / downstream fine-tuning,
- localized or selective unlearning,
- stronger evaluation beyond deterministic one-shot metrics,
- reasoning / multilingual / tool-use settings.

## Most important latest papers

### ICLR 2026
- **Explainable LLM Unlearning through Reasoning**
  - Source: https://openreview.net/pdf/a03a961b6a7672a3e7839a90b43c0f26934b7092.pdf
  - Key value: accepted 2026 signal that answer-only forgetting is insufficient; reasoning-aware unlearning is becoming first-class.

### ICLR 2025
- **LLM Unlearning via Loss Adjustment with Only Forget Data**
  - Source: https://openreview.net/submissions?page=269&venue=ICLR.cc%2F2025%2FConference
  - Key value: practical delete-request setting without requiring retain data.
- **Unified Parameter-Efficient Unlearning for LLMs**
  - Source: https://openreview.net/submissions?page=264&venue=ICLR.cc%2F2025%2FConference
  - Key value: low-cost unlearning via PEFT-style updates.
- **A Closer Look at Machine Unlearning for Large Language Models**
  - Source: https://openreview.net/submissions?page=264&venue=ICLR.cc%2F2025%2FConference
  - Key value: diagnostic framing paper for failure analysis.

### ICML 2025
- **Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**
  - Source: https://proceedings.mlr.press/v267/fan25e.html
  - Key value: strongest recent paper on relearning robustness.
- **Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning**
  - Source: https://proceedings.mlr.press/v267/wang25en.html
  - Key value: robustness under downstream fine-tuning, not just one fixed attack.
- **Adaptive Localization of Knowledge Negation for Continual LLM Unlearning**
  - Source: https://proceedings.mlr.press/v267/wuerkaixi25a.html
  - Key value: selectively localizes updates for continual unlearning.
- **Tool Unlearning for Tool-Augmented LLMs**
  - Source: https://proceedings.mlr.press/v267/cheng25a.html
  - Key value: extends the setting from facts to tool capabilities.

### NeurIPS 2025
- **Unlearning-Aware Minimization (UAM)**
  - Source: https://neurips.cc/virtual/2025/poster/116406
  - OpenReview: https://openreview.net/forum?id=e3abccbf80
  - Key value: retain-aware min-max optimization principle.
- **Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness**
  - Source: https://neurips.cc/virtual/2025/poster/119349
  - Key value: evaluation paper showing current metrics overestimate forgetting.
- **Elastic Robust Unlearning of Specific Knowledge in Large Language Models**
  - Source: https://openreview.net/forum?id=VrXjAfdwrN
  - Key value: robust PO-style unlearning.

### EMNLP / ACL 2025
- **Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills**
  - Source: https://aclanthology.org/2025.emnlp-main.220/
  - Key value: reasoning-trace leakage is a real issue.
- **A Fully Probabilistic Perspective on Large Language Model Unlearning: Evaluation and Optimization**
  - Source: https://aclanthology.org/2025.emnlp-main.452/
  - Key value: stricter probabilistic evaluation of leakage.
- **Learn and Unlearn: Addressing Misinformation in Multilingual LLMs**
  - Source: https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2025.emnlp-main.516.pdf
  - Key value: multilingual propagation problem.
- **Not Every Token Needs Forgetting**
  - Source: https://aclanthology.org/2025.findings-emnlp.96/
  - Key value: selective token-level forgetting.
- **UIPE**
  - Source: https://aclanthology.org/2025.findings-emnlp.1374/
  - Key value: forgetting correlated knowledge, not only direct targets.

## Recommended thesis direction
### Preferred
**Evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**

Why this looks strongest:
- latest evaluation papers show the metrics are still weak,
- latest method papers move toward localization and robustness,
- EMNLP is an ideal venue for language-centric failure analysis.

### Backup
**Selective representation unlearning with robustness-to-relearning evaluation**

Why:
- combines ICML 2025 robustness papers with selective localization ideas and EMNLP-facing evaluation.

## Next steps
1. Turn the latest-top-venue list into detailed per-paper notes.
2. Separate papers into method vs evaluation vs setting-expansion buckets.
3. Narrow the paper thesis to one main direction and one backup.
