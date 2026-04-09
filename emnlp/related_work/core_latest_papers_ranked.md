# Core Latest Papers Ranked for Our Project (2025-2026)

This is **not** a ranking of the entire field by citation count. It is a ranking of the papers that matter most **for the paper we are likely to write**.

## Rank 1 — Must read now
### Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills (EMNLP 2025)
**Why #1:** our likely direction already leans toward multi-turn / reasoning-aware / evaluation-faithful unlearning. This paper is the most direct proof that answer-level forgetting is insufficient once reasoning models are involved.

**What to learn from it:**
- how they define reasoning-trace leakage,
- what benchmark mix they use,
- how they preserve reasoning skill while suppressing harmful traces.

**Verdict:** **must-read**.

## Rank 2 — Must read now
### A Fully Probabilistic Perspective on Large Language Model Unlearning: Evaluation and Optimization (EMNLP 2025)
**Why #2:** this is one of the strongest recent arguments that ordinary output-level evaluation is too weak. If we want an evaluation-heavy EMNLP paper, this is one of the key foundation papers.

**What to learn from it:**
- why deterministic metrics can overstate forgetting,
- how probability-aware evaluation changes the interpretation of success,
- what evaluation gaps remain after probabilistic scoring.

**Verdict:** **must-read**.

## Rank 3 — Must read now
### Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond (ICML 2025)
**Why #3:** relearning robustness is now central. A model that forgets only until the next fine-tuning round is not good enough.

**What to learn from it:**
- how relearning attacks are defined,
- what robust optimization buys us,
- how to frame downstream resilience as part of the unlearning objective.

**Verdict:** **must-read**.

## Rank 4 — Must read now
### Adaptive Localization of Knowledge Negation for Continual LLM Unlearning (ICML 2025)
**Why #4:** if our method direction is selective/localized unlearning, this is the top-tier paper most directly aligned with that idea.

**What to learn from it:**
- how they motivate localization,
- how continual deletion requests change the problem,
- what the method does to avoid unnecessary global damage.

**Verdict:** **must-read**.

## Rank 5 — Must read now
### Unlearning-Aware Minimization (UAM) (NeurIPS 2025)
**Why #5:** UAM is not the most NLP-specific paper, but conceptually it is one of the strongest recent papers. It gives us the retain-aware min-max lens we can reuse.

**What to learn from it:**
- how they formalize the forget/retain tradeoff,
- why naive forget-only objectives underperform,
- how to translate the idea into an NLP-specific setting.

**Verdict:** **must-read**.

## Rank 6 — Should read soon
### Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness (NeurIPS 2025)
**Why #6:** this is one of the best recent critiques of shallow unlearning evaluation.

**What to learn from it:**
- correlated-knowledge leakage,
- why direct target removal is not enough,
- how confidence reveals hidden retention.

**Verdict:** **should-read**.

## Rank 7 — Should read soon
### Learn and Unlearn: Addressing Misinformation in Multilingual LLMs (EMNLP 2025)
**Why #7:** if we want a strong EMNLP-native contribution, multilingual propagation is one of the best directions.

**What to learn from it:**
- how misinformation travels across languages,
- why English-only unlearning can fail,
- how multilingual evaluation reshapes the problem.

**Verdict:** **should-read**.

## Rank 8 — Should read soon
### Not Every Token Needs Forgetting: Selective Unlearning Balancing Forgetting and Utility in Large Language Models (Findings of EMNLP 2025)
**Why #8:** this is one of the cleanest selective-unlearning papers in NLP. It translates the vague intuition “forget only what matters” into a token-level story.

**What to learn from it:**
- how they identify important forgetting tokens,
- what utility gains come from not touching everything,
- where token-level selection may still fail.

**Verdict:** **should-read**.

## Secondary papers — read if time allows

### UIPE (Findings of EMNLP 2025)
Important because it expands the forget target to **related knowledge**, not just the exact target.

### OBLIVIATE (EMNLP 2025)
Important as a strong practical baseline if we build a method.

### Explainable LLM Unlearning through Reasoning (ICLR 2026)
Very important signal for where the field is heading, but because it is so new, I would still understand the 2025 foundations first.

### Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning (ICML 2025)
Important for robustness, especially if we push the downstream fine-tuning angle harder.

### Tool Unlearning for Tool-Augmented LLMs (ICML 2025)
Important if we later expand to agentic/tool-use settings.

## Final recommendation
If we only read **six** papers before locking our thesis, read:
1. Reasoning Model Unlearning
2. Fully Probabilistic Perspective
3. SAM/relearning robustness paper
4. Adaptive Localization
5. UAM
6. Do LLMs Really Forget?

Those six already give us:
- one reasoning paper,
- one evaluation paper,
- one robustness paper,
- one localization paper,
- one optimization paper,
- one benchmark-critique paper.

That is enough to define a serious EMNLP project direction.
