# Latest-first LLM Machine Unlearning Survey (2025-2026 top venues)

Date: 2026-04-08
Scope: prioritize **2025-2026** papers, especially **ICLR / ICML / NeurIPS**, while adding a few **ACL/EMNLP 2025** papers because our target is still NLP-facing.

> Note: as of **April 8, 2026**, the mature core of the literature is still **2025**. ICLR 2026 has begun surfacing relevant accepted papers, but the most stable and citable cluster is still 2025.

---

## 1. Fast take: what the latest frontier actually is
The newest high-value papers are not just doing “forget loss on TOFU” anymore. The frontier has shifted toward:

1. **robustness to relearning / downstream re-tuning**,
2. **localized or selective unlearning**,
3. **better evaluation** beyond one-shot deterministic metrics,
4. **reasoning / multilingual / tool-use settings**,
5. **parameter-efficient or low-cost unlearning**.

If we want a publishable angle, these are the zones that still feel alive.

---

## 2. Must-read papers by venue

### A. ICLR 2026

#### 1) Explainable LLM Unlearning through Reasoning (ICLR 2026)
- Source: https://openreview.net/pdf/a03a961b6a7672a3e7839a90b43c0f26934b7092.pdf
- Why it matters:
  - one of the clearest **ICLR 2026 accepted** signals in this area,
  - explicitly pushes unlearning toward **reasoning-aware / explanation-aware evaluation and control**,
  - very relevant if we want a reasoning-trace angle.
- Takeaway for us:
  - strong evidence that answer-only forgetting is no longer enough.

### B. ICLR 2025

#### 2) LLM Unlearning via Loss Adjustment with Only Forget Data (ICLR 2025 Poster)
- Source: https://openreview.net/submissions?page=269&venue=ICLR.cc%2F2025%2FConference
- Why it matters:
  - important because it reduces reliance on retain data,
  - practical if the deletion request only gives us the forget set.
- Takeaway for us:
  - good baseline if we care about realistic deletion settings.

#### 3) Unified Parameter-Efficient Unlearning for LLMs (ICLR 2025 Poster)
- Source: https://openreview.net/submissions?page=264&venue=ICLR.cc%2F2025%2FConference
- Why it matters:
  - PEFT-style unlearning is very attractive operationally,
  - relevant for low-cost or multi-request settings.
- Takeaway for us:
  - if we propose anything practical, this is a comparison anchor.

#### 4) A Closer Look at Machine Unlearning for Large Language Models (ICLR 2025 Poster)
- Source: https://openreview.net/submissions?page=264&venue=ICLR.cc%2F2025%2FConference
- Why it matters:
  - more diagnostic / analytical than purely algorithmic,
  - helps expose where existing methods fail.
- Takeaway for us:
  - useful intro/related-work framing paper.

### C. ICML 2025

#### 5) Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond (ICML 2025)
- Source: https://proceedings.mlr.press/v267/fan25e.html
- Why it matters:
  - one of the strongest 2025 papers on **relearning robustness**,
  - directly addresses the common failure mode where “forgotten” knowledge comes back.
- Takeaway for us:
  - robust unlearning is not optional anymore.

#### 6) Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning (ICML 2025)
- Source: https://proceedings.mlr.press/v267/wang25en.html
- Why it matters:
  - strengthens the robustness story beyond a single attack template,
  - useful if we care about post-unlearning deployment stability.
- Takeaway for us:
  - downstream fine-tuning resilience is now a serious evaluation axis.

#### 7) Adaptive Localization of Knowledge Negation for Continual LLM Unlearning (ICML 2025)
- Source: https://proceedings.mlr.press/v267/wuerkaixi25a.html
- Why it matters:
  - directly about **localizing** updates for continual unlearning,
  - very close to the “forget only where needed” instinct.
- Takeaway for us:
  - strongest ICML-side bridge to our selective-intervention philosophy.

#### 8) Tool Unlearning for Tool-Augmented LLMs (ICML 2025)
- Source: https://proceedings.mlr.press/v267/cheng25a.html
- Why it matters:
  - opens the setting beyond plain factual QA,
  - relevant if we want a more modern tool-augmented LLM framing.
- Takeaway for us:
  - “what must be forgotten” can be a capability/tool, not just a fact.

### D. NeurIPS 2025

#### 9) Unlearning-Aware Minimization (NeurIPS 2025)
- Source: https://neurips.cc/virtual/2025/poster/116406
- OpenReview: https://openreview.net/forum?id=e3abccbf80
- Why it matters:
  - strong **retain-aware min-max optimization** perspective,
  - not purely NLP, but very relevant to LLM unlearning design.
- Takeaway for us:
  - optimization should explicitly target the forget/retain frontier.

#### 10) Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness (NeurIPS 2025)
- Source: https://neurips.cc/virtual/2025/poster/119349
- Why it matters:
  - one of the best recent **evaluation papers**,
  - argues that isolated-fact forgetting misses correlated knowledge and confidence effects.
- Takeaway for us:
  - this is a major warning against overclaiming progress from simple benchmarks.

#### 11) Elastic Robust Unlearning of Specific Knowledge in Large Language Models (NeurIPS 2025 Poster)
- Source: https://openreview.net/forum?id=VrXjAfdwrN
- Why it matters:
  - blends robust unlearning with stronger optimization design,
  - explicitly targets both forget performance and retain utility.
- Takeaway for us:
  - robustness-aware PO-style unlearning is becoming a real sub-line.

### E. ACL / EMNLP 2025 (important because our target venue is NLP)

#### 12) Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills (EMNLP 2025)
- Source: https://aclanthology.org/2025.emnlp-main.220/
- Why it matters:
  - directly attacks **reasoning-trace leakage**,
  - one of the most EMNLP-relevant recent papers.
- Takeaway for us:
  - if we want a reasoning-aware paper, this is mandatory reading.

#### 13) A Fully Probabilistic Perspective on Large Language Model Unlearning: Evaluation and Optimization (EMNLP 2025)
- Source: https://aclanthology.org/2025.emnlp-main.452/
- Why it matters:
  - moves unlearning evaluation beyond deterministic one-shot scoring,
  - introduces stricter leakage-sensitive evaluation.
- Takeaway for us:
  - evaluation methodology itself is still an open problem.

#### 14) Learn and Unlearn: Addressing Misinformation in Multilingual LLMs (EMNLP 2025)
- Source: https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2025.emnlp-main.516.pdf
- Why it matters:
  - shows that unlearning in one language is not enough,
  - cross-lingual propagation is real.
- Takeaway for us:
  - multilingual unlearning is one of the cleanest EMNLP-native directions.

#### 15) OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models (EMNLP 2025)
- Source: https://aclanthology.org/volumes/2025.emnlp-main/
- Why it matters:
  - practical, robustness-minded, and LLM-centered,
  - useful as a strong applied baseline.
- Takeaway for us:
  - if we propose a method, OBLIVIATE is likely in the comparison set.

#### 16) Not Every Token Needs Forgetting: Selective Unlearning Balancing Forgetting and Utility in Large Language Models (Findings of EMNLP 2025)
- Source: https://aclanthology.org/2025.findings-emnlp.96/
- Why it matters:
  - perhaps the cleanest statement of **selective token-level forgetting**,
  - very aligned with “localize intervention” thinking.
- Takeaway for us:
  - selective unlearning is already emerging, but still not saturated.

#### 17) UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets (Findings of EMNLP 2025)
- Source: https://aclanthology.org/2025.findings-emnlp.1374/
- Why it matters:
  - explicitly handles **related/correlated knowledge**,
  - useful bridge between pure target forgetting and knowledge-graph style evaluation.
- Takeaway for us:
  - correlated-knowledge control is becoming increasingly important.

---

## 3. Latest reading order I recommend
If we optimize for “latest + top venue + actually useful”:

1. Explainable LLM Unlearning through Reasoning (ICLR 2026)
2. Towards LLM Unlearning Resilient to Relearning Attacks (ICML 2025)
3. Invariance Makes LLM Unlearning Resilient... (ICML 2025)
4. Adaptive Localization of Knowledge Negation... (ICML 2025)
5. Unlearning-Aware Minimization (NeurIPS 2025)
6. Do LLMs Really Forget? (NeurIPS 2025)
7. Reasoning Model Unlearning (EMNLP 2025)
8. A Fully Probabilistic Perspective... (EMNLP 2025)
9. Learn and Unlearn (EMNLP 2025)
10. Selective Unlearning / UIPE (Findings of EMNLP 2025)
11. Loss Adjustment with Only Forget Data (ICLR 2025)
12. Unified Parameter-Efficient Unlearning for LLMs (ICLR 2025)

---

## 4. What this suggests for our paper
### Most promising direction
**Evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**

Why:
- latest papers keep exposing evaluation weaknesses,
- latest method papers keep moving toward localization and robustness,
- EMNLP will care about language-centric failure modes more than a generic MU objective tweak.

### Strong backup direction
**Selective representation unlearning with robustness-to-relearning evaluation**

Why:
- Adaptive Localization + UAM + robust ICML papers give a clean method story,
- Selective Unlearning / UIPE give NLP-facing precedent,
- still feels publishable if evaluation is strong.

---

## 5. Bottom line
For 2025-2026, the real “important” papers are no longer just TOFU/NPO. The papers that now matter most are the ones about:
- **relearning robustness**,
- **localized updates**,
- **better evaluation**, and
- **reasoning/multilingual/tool settings**.

That is the slice we should build on.
