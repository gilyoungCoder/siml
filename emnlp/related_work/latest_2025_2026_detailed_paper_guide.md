# Latest LLM Machine Unlearning Papers (2025-2026) — Detailed Beginner-Friendly Guide

Date: 2026-04-08
Audience: a general ML master's student who knows basic deep learning / LLM finetuning, but is new to machine unlearning.

---

## 0. Before reading the papers: what “LLM unlearning” means

### The problem in one sentence
We already trained an LLM, but later decide that some data, knowledge, capability, or behavior should **no longer influence the model**. We want to remove that influence **without retraining from scratch** and **without breaking everything else**.

### Why this is hard
If you simply finetune the model to “forget,” you often get one of four bad outcomes:
1. the model still remembers the target indirectly,
2. the model loses too much general ability,
3. the target knowledge comes back after later finetuning,
4. benchmark scores look good but the model still leaks the knowledge under paraphrases, reasoning traces, or other languages.

### The four main axes in this field
When reading papers, almost every paper is really about one or more of these:

1. **Forget quality** — did the model actually forget the target?
2. **Utility retention** — did the model keep normal capabilities?
3. **Robustness** — does the forgetting survive attacks, relearning, or downstream tuning?
4. **Evaluation quality** — are we measuring true forgetting, or just benchmark gaming?

### The latest trend (2025-2026)
The frontier is no longer “just make forgetting stronger.”
The most interesting recent work focuses on:
- **robustness to relearning**,
- **localized/selective forgetting**,
- **reasoning-trace leakage**,
- **multilingual propagation**, and
- **better evaluation protocols**.

---

## 1. Explainable LLM Unlearning through Reasoning (ICLR 2026)
- **Venue**: ICLR 2026
- **Source**: https://openreview.net/forum?id=wec4qy2XIF
- **PDF**: https://openreview.net/pdf/a03a961b6a7672a3e7839a90b43c0f26934b7092.pdf

### What problem is this paper solving?
Most unlearning work checks whether the **final answer** disappeared. But for reasoning models, that is not enough. The model may stop printing the final forbidden answer while still exposing the target through its chain of thought or intermediate reasoning.

### Core idea in plain English
The paper argues that unlearning should be **reasoning-aware** and **explainable**. In other words, we should not only ask “did the answer disappear?” but also “what reasoning path did the model use, and did that path still reveal the forbidden knowledge?”

### Why this matters
Reasoning models are increasingly important. If a benchmark says “good, the answer is gone,” but the model can still reveal the answer step by step, the unlearning is not trustworthy.

### Method intuition
From the title and abstract direction, the paper pushes toward using explicit reasoning structure as part of the unlearning/evaluation loop. The exact technical details matter, but the big conceptual shift is this:

- old view: unlearn the output,
- new view: unlearn the **route** that leads to the output.

### Strengths
- Very modern problem setting.
- Strong conceptual fit with where the field is going.
- Easy for reviewers to see why answer-only metrics are weak.

### Limitations / caution
- It is very new, so the community consensus around its exact setup is still forming.
- Reasoning-based evaluation can be noisy because chain-of-thought itself is unstable.

### Relevance to us
Very high. If we want an EMNLP paper about **reasoning-aware** or **multi-turn selective** unlearning, this paper is one of the clearest signals that the area is real and timely.

---

## 2. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond (ICML 2025)
- **Venue**: ICML 2025
- **Source**: https://proceedings.mlr.press/v267/fan25e.html

### What problem is this paper solving?
A common failure mode in LLM unlearning is: the model seems to forget, but if you fine-tune it later—even on something unrelated—the forgotten knowledge comes back.

### Core idea in plain English
The paper treats unlearning as a **robust optimization** problem. It asks: can we unlearn in a way that lands in a “safer” parameter region, where the forgotten knowledge is harder to recover later?

### Why sharpness matters
In deep learning, “sharpness” roughly refers to whether a small parameter change causes a large loss change. Sharpness-aware methods try to find flatter, more stable regions. This paper uses that intuition for unlearning: if the forgotten state is fragile, later tuning may quickly restore the target knowledge.

### What the paper contributes
- A robustness-oriented lens on LLM unlearning.
- A method inspired by **Sharpness-Aware Minimization (SAM)**.
- Experiments showing stronger resistance to **relearning attacks**.

### Why this paper is important
It changed the conversation from:
> “Did the model forget right now?”

to:
> “Will the forgetting survive future updates?”

That is a much more realistic question.

### Strengths
- Strong practical relevance.
- Excellent motivation.
- Good bridge between optimization theory and deployment reality.

### Limitations
- Robustness to one family of recovery settings does not guarantee robustness to all possible future training conditions.
- It is still largely a weight-update story rather than a language-structure story.

### Relevance to us
Extremely high. This paper is one of the best anchors if we want to argue that **relearning robustness must be part of the evaluation**.

---

## 3. Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning (ICML 2025)
- **Venue**: ICML 2025
- **Source**: https://proceedings.mlr.press/v267/wang25en.html

### What problem is this paper solving?
Even if you defend against a few known relearning attacks, the model may still recover the knowledge after some **new downstream fine-tuning task** that you did not anticipate.

### Core idea in plain English
The paper imports the idea of **invariance** from invariant risk minimization (IRM). The goal is to make the unlearned state hold up across different later training environments, not just one specific attack.

### Intuition
Instead of learning a fragile “forgetting trick” that only survives one kind of update, the method tries to find a forgetting solution that remains stable across varied downstream settings.

### What is ILU?
The paper proposes **Invariant LLM Unlearning (ILU)**, which adds invariance regularization so the unlearning effect generalizes better to unseen future finetuning tasks.

### Why this is interesting
This is a step beyond ordinary robustness. It is trying to get **distributional robustness over future tuning environments**.

### Strengths
- Strong conceptual upgrade over one-attack-at-a-time robustness.
- Good fit for deployment where future adaptation is unpredictable.
- Includes task-vector analysis for interpretation.

### Limitations
- Invariance assumptions may not fully match real production updates.
- Still mainly focused on parameter robustness, not richer language leakage modes like dialogue recovery.

### Relevance to us
High. If we propose a new evaluation protocol, this paper helps justify why “one-shot forgetting” is not enough.

---

## 4. Adaptive Localization of Knowledge Negation for Continual LLM Unlearning (ICML 2025)
- **Venue**: ICML 2025
- **Source**: https://proceedings.mlr.press/v267/wuerkaixi25a.html

### What problem is this paper solving?
Many unlearning methods modify the model too broadly. That hurts utility, and it becomes even worse if we receive multiple deletion requests over time.

### Core idea in plain English
Instead of applying a broad forgetting update everywhere, this paper tries to **localize** the negation/unlearning operation. In other words:
- find where the relevant knowledge lives,
- change those parts more,
- change unrelated parts less.

### Why “continual” matters
In real systems, deletion requests come repeatedly. If each unlearning step damages the model globally, utility will collapse. Localization helps make repeated unlearning more sustainable.

### Why this paper is important
This is one of the clearest top-tier examples of the field moving toward **selective/localized unlearning** rather than brute-force forgetting.

### Strengths
- Strong practical motivation.
- Natural bridge between interpretability/localization and machine unlearning.
- Very close to the “only intervene where needed” intuition.

### Limitations
- Localization in neural networks is never perfect; correlated knowledge can live in overlapping subspaces.
- Continual settings are harder to benchmark cleanly.

### Relevance to us
Very high. This paper is probably the best ICML-side inspiration for a method that borrows our repo’s high-level selective-intervention philosophy.

---

## 5. Tool Unlearning for Tool-Augmented LLMs (ICML 2025)
- **Venue**: ICML 2025
- **Source**: https://proceedings.mlr.press/v267/cheng25a.html

### What problem is this paper solving?
Most unlearning papers treat the forget target as a **fact** or a **document**. But modern LLMs often use external tools. Sometimes what we need to remove is not a fact, but a **tool-use capability** or tool-associated behavior.

### Core idea in plain English
The paper expands the scope of unlearning:
- not just forgetting “what the model knows,”
- but forgetting “what the model knows how to do with tools.”

### Why this matters
This is a more realistic future-facing setting. LLMs are increasingly agents, assistants, and tool users. If a model should no longer call a harmful tool or unsafe workflow, ordinary document-level unlearning is not enough.

### Strengths
- Very modern setting.
- Broadens the field in an important direction.
- Good inspiration for capability-level unlearning.

### Limitations
- Tool ecosystems vary a lot, so generalization is harder.
- The setting is a little less classic than document/fact forgetting, so comparison baselines may be weaker.

### Relevance to us
Medium to high. Maybe not our first target, but extremely useful if we later want an agentic or tool-use evaluation extension.

---

## 6. Unlearning-Aware Minimization (UAM) (NeurIPS 2025)
- **Venue**: NeurIPS 2025
- **Source**: https://neurips.cc/virtual/2025/poster/116406
- **OpenReview**: https://openreview.net/forum?id=e3abccbf80

### What problem is this paper solving?
Approximate unlearning methods often have a bad tradeoff:
- forget more → utility collapses,
- preserve utility → forgetting is weak.

### Core idea in plain English
UAM says this tradeoff should be handled explicitly as a **retain-aware optimization problem**. Instead of optimizing forgetting alone, we should optimize forgetting while deliberately protecting retained knowledge.

### Intuition
The method frames unlearning as a kind of min-max or retain-aware optimization:
- push the model away from the target knowledge,
- but simultaneously avoid destroying the retain set.

### Why this paper is important
UAM is not the most NLP-specific paper, but it is very important conceptually because it provides a clean answer to this question:
> Why are so many unlearning methods bad in practice?

Its answer is: because they optimize the wrong objective.

### Strengths
- Strong optimization perspective.
- Cleanly motivates retain-aware learning.
- Useful beyond any single benchmark.

### Limitations
- It is more of a general machine-unlearning paper with LLM relevance than an NLP benchmark paper.
- By itself, it does not solve language-specific issues like multi-turn recovery or multilingual propagation.

### Relevance to us
High as an **idea source**. If we build a method, UAM is a great conceptual backbone. If we build an EMNLP paper, we probably need to combine UAM-like optimization thinking with richer language-centric evaluation.

---

## 7. Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness (NeurIPS 2025)
- **Venue**: NeurIPS 2025
- **Source**: https://neurips.cc/virtual/2025/poster/119349

### What problem is this paper solving?
A model can appear to forget a target fact under direct questioning, but still retain:
- related facts,
- correlated evidence,
- high internal confidence about the target.

### Core idea in plain English
The paper argues that unlearning evaluation must go beyond “did the exact answer disappear?” It explicitly studies:
- **knowledge correlation**: nearby or supporting facts that still reveal the target,
- **confidence awareness**: whether the model still behaves as though it knows the answer, even when surface outputs change.

### Why this is important
This is one of the best recent papers for showing why simple exact-match or one-shot refusal metrics are inadequate.

### Strengths
- Strong evaluation contribution.
- Forces the field to think about indirect leakage.
- Easy to use when motivating stricter benchmarks.

### Limitations
- Evaluation papers often do not by themselves provide a complete method answer.
- Correlation structure can be hard to measure comprehensively.

### Relevance to us
Extremely high. If we want to argue that current benchmarks overstate progress, this is a core citation.

---

## 8. Elastic Robust Unlearning of Specific Knowledge in Large Language Models (NeurIPS 2025 Poster)
- **Venue**: NeurIPS 2025 poster
- **Source**: https://openreview.net/forum?id=VrXjAfdwrN

### What problem is this paper solving?
We want robust forgetting of specific target knowledge, but not through a brittle one-shot update that harms utility or fails under perturbation.

### Core idea in plain English
The paper pushes a **robust, elastic** view of knowledge removal. The word “elastic” suggests a controlled balance: apply enough force to erase the target, but not so much that the whole model is distorted.

### Why it matters
It fits the 2025 theme that unlearning should be robust and specific, not only strong.

### Strengths
- Specific-knowledge focus is practically useful.
- Aligns with the trend toward robust optimization.

### Limitations
- As a poster, it may be less established than the highest-impact main-track papers.
- Needs to be read alongside stronger evaluation papers to know what kind of robustness it truly guarantees.

### Relevance to us
Medium to high. Useful as a neighboring method paper, especially if we end up in the robustness + localization space.

---

## 9. Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills (EMNLP 2025)
- **Venue**: EMNLP 2025
- **Source**: https://aclanthology.org/2025.emnlp-main.220/

### What problem is this paper solving?
Reasoning models can still leak target information in their **reasoning traces**, even if the final answer is hidden or altered.

### Core idea in plain English
The paper introduces a reasoning-aware unlearning perspective and proposes **R²MU** (Reasoning-aware Representation Misdirection for Unlearning). The goal is to suppress sensitive reasoning traces while keeping the model’s general reasoning skill intact.

### Why this paper is important
This is one of the clearest EMNLP 2025 papers showing that LLM unlearning must adapt to the rise of reasoning models.

### Evaluation setting
The paper evaluates on reasoning and safety benchmarks such as:
- WMDP,
- StrongReject,
- JBB-Behaviors,
- WildJailbreak,
with reasoning-capable models like DeepSeek-R1-distilled variants.

### Strengths
- Strongly aligned with the current reasoning-model wave.
- Makes the leakage problem concrete.
- Very compelling for NLP reviewers.

### Limitations
- Reasoning traces are tricky: removing harmful traces without harming general reasoning is delicate.
- Results may depend on how reasoning traces are elicited.

### Relevance to us
Extremely high. If we want an EMNLP story, this is one of the most important papers to understand deeply.

---

## 10. A Fully Probabilistic Perspective on Large Language Model Unlearning: Evaluation and Optimization (EMNLP 2025)
- **Venue**: EMNLP 2025
- **Source**: https://aclanthology.org/2025.emnlp-main.452/

### What problem is this paper solving?
A lot of unlearning evaluation uses deterministic metrics on a few outputs. But LLM behavior is probabilistic. A model might still assign high probability to forgotten content even if one sampled answer looks safe.

### Core idea in plain English
The paper says we should evaluate and optimize unlearning in a **probabilistic** way, not just by checking one surface response.

### Why this matters
This is crucial because many apparent successes in unlearning may really be **sampling artifacts**. The knowledge might still be latent in the distribution.

### Strengths
- Methodologically strong.
- Important for careful evaluation.
- Good bridge between theory and practical measurement.

### Limitations
- Probabilistic evaluation can be more expensive.
- It is still not the whole story unless combined with richer prompts, paraphrases, and multi-turn settings.

### Relevance to us
Very high. This paper is a strong anchor if we choose an evaluation-heavy EMNLP direction.

---

## 11. Learn and Unlearn: Addressing Misinformation in Multilingual LLMs (EMNLP 2025)
- **Venue**: EMNLP 2025
- **Source**: https://aclanthology.org/2025.emnlp-main.516/

### What problem is this paper solving?
Suppose misinformation is learned in one language. Does unlearning it in English fix the problem everywhere? This paper says: often, **no**.

### Core idea in plain English
Misinformation spreads across languages in multilingual LLMs. Standard unlearning that focuses only on English can fail, or even reinforce the bad content elsewhere. Effective multilingual unlearning must handle both English and the original misinformation language.

### Why this is important
This paper is very EMNLP-native. It shows that language transfer itself becomes an unlearning problem.

### Strengths
- Clear multilingual problem formulation.
- Strong NLP relevance.
- Shows that monolingual evaluation can give false confidence.

### Limitations
- Multilingual setups are expensive and complex.
- Language coverage is always incomplete.

### Relevance to us
Very high if we want a multilingual angle, and high even if we do not—because it proves current evaluation is incomplete.

---

## 12. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models (EMNLP 2025)
- **Venue**: EMNLP 2025
- **Source**: https://aclanthology.org/2025.emnlp-main.183/

### What problem is this paper solving?
We want a practical unlearning pipeline that is efficient, robust, and useful across different targets such as memorized text, hazardous content, or benchmark-specific forget data.

### Core idea in plain English
OBLIVIATE builds a practical framework with three parts in its loss:
- **masking**,
- **distillation**,
- **world fact** preservation,
and uses **LoRA** for parameter-efficient updates.

### Why it matters
This is one of the strongest “practical framework” papers in the 2025 EMNLP cluster.

### Strengths
- Practical and relatively deployment-friendly.
- Uses broad datasets such as Harry Potter, WMDP, and TOFU.
- Includes memorization and MIA-style robustness angles.

### Limitations
- As with many framework papers, the final performance depends on many design choices.
- Practicality does not automatically imply better evaluation faithfulness.

### Relevance to us
High. If we do a method paper, OBLIVIATE is likely a must-compare baseline.

---

## 13. Not Every Token Needs Forgetting: Selective Unlearning Balancing Forgetting and Utility in Large Language Models (Findings of EMNLP 2025)
- **Venue**: Findings of EMNLP 2025
- **Source**: https://aclanthology.org/2025.findings-emnlp.96/

### What problem is this paper solving?
A document contains many tokens that are not actually responsible for the harmful/private/copyrighted knowledge. If we force the model to forget **every** token in the document, we damage general knowledge unnecessarily.

### Core idea in plain English
The paper proposes **Selective Unlearning (SU)**:
- identify the important subset of tokens tied to the unwanted information,
- unlearn only those tokens,
- leave general/common tokens alone.

### Why this matters
This is one of the cleanest formulations of selective forgetting in the NLP literature.

### Strengths
- Easy intuition.
- Strong utility-preservation story.
- Very close to how a human would want forgetting to work.

### Limitations
- Token importance estimation is not perfect.
- Harmful knowledge can be distributed across many non-obvious tokens and contexts.

### Relevance to us
Extremely high. If we want a “forget only what matters” story, this is a key reference.

---

## 14. UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets (Findings of EMNLP 2025)
- **Venue**: Findings of EMNLP 2025
- **Source**: https://aclanthology.org/2025.findings-emnlp.1374/

### What problem is this paper solving?
Even if the direct target is forgotten, the model may reconstruct it from **related knowledge**. For example, associated facts, paraphrases, linked entities, or nearby concepts can act as recovery paths.

### Core idea in plain English
UIPE says that to unlearn robustly, we should not only attack the exact target. We should also weaken knowledge that is highly correlated with that target.

### Why this matters
This paper is important because it explicitly recognizes that knowledge in LLMs is **networked**, not isolated.

### Strengths
- Strong intuition.
- Good bridge between direct forgetting and correlated-knowledge suppression.
- Useful for more realistic leakage analysis.

### Limitations
- Deciding what counts as “related enough” is difficult.
- Over-expanding the correlated set could hurt utility.

### Relevance to us
Very high. This paper pairs naturally with evaluation papers about indirect leakage and knowledge correlation.

---

## 15. What all these papers mean together

### The old story (2023-2024)
- Can we make the model forget at all?
- Can we do better than gradient ascent?
- Can we benchmark forgetting on something like TOFU?

### The new story (2025-2026)
- Will forgetting survive **relearning**?
- Can we forget **selectively** instead of globally?
- Does the model still leak the target via **reasoning traces**?
- Does the target reappear in **other languages**?
- Are our benchmarks measuring true forgetting or just surface behavior?

This is why the latest papers are more interesting than just reading old benchmark baselines again.

---

## 16. My recommendation for a newcomer reading this area
If you are a general ML master's student and want the cleanest path into this field, read in this order:

1. **UAM** — to understand the optimization tradeoff.
2. **ICML 2025 SAM paper** — to understand relearning robustness.
3. **Adaptive Localization** — to understand selective/localized updates.
4. **Reasoning Model Unlearning** — to understand trace leakage.
5. **Fully Probabilistic Perspective** — to understand why evaluation is hard.
6. **Learn and Unlearn** — to understand multilingual propagation.
7. **Selective Unlearning** and **UIPE** — to understand why coarse forgetting is too blunt.
8. **Explainable LLM Unlearning through Reasoning** — to see where 2026 is heading.

---

## 17. Bottom line for our project
The strongest research opportunity is probably **not** another plain forget-loss baseline.
The better opportunity is one of these:

### Option A — evaluation-heavy EMNLP paper
Build a better evaluation protocol for:
- multi-turn recovery,
- reasoning-trace leakage,
- multilingual propagation,
- correlated-knowledge recovery.

### Option B — selective method + strong evaluation
Build a method that:
- detects when/where target knowledge is active,
- applies forgetting pressure selectively,
- and proves robustness under relearning, reasoning, and multilingual evaluation.

That is the slice of the field that still looks genuinely open.
