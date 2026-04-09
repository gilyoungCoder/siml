# EMNLP Proposal Package — LLM Unlearning (2025–2026 Literature-Informed)

## 1. Title candidates

### Preferred title family
1. **Evaluation-Faithful Selective Unlearning for LLMs under Multi-Turn, Reasoning, and Multilingual Recovery**
2. **Do LLMs Really Forget in Dialogue? Selective Unlearning under Reasoning and Multilingual Recovery**
3. **Beyond Answer-Level Forgetting: Evaluation-Faithful Selective Unlearning for Modern LLMs**

### Backup title family
4. **Selective Representation Unlearning for LLMs with Relearning-Robust Evaluation**
5. **Localized LLM Unlearning that Survives Relearning, Reasoning, and Cross-Lingual Recovery**

---

## 2. Problem statement
Current LLM unlearning methods are usually evaluated in settings that are too narrow: one-shot prompts, direct answer extraction, and mostly monolingual benchmarks. As a result, many methods appear successful even when the target knowledge can still be recovered through multi-turn dialogue, reasoning traces, correlated facts, downstream fine-tuning, or cross-lingual transfer. At the same time, globally applied forgetting updates often degrade general utility because they do not distinguish target-relevant knowledge from irrelevant context.

Our core problem is therefore:

> **How can we evaluate and enforce selective LLM unlearning so that forgetting remains effective under realistic language-centric recovery settings—especially multi-turn interaction, reasoning traces, and multilingual transfer—while preserving utility on retained knowledge?**

This is not just a stronger benchmark question. It is also a modeling question, because once realistic recovery settings are included, coarse global forgetting may no longer be sufficient.

---

## 3. Decision: evaluation-first vs method-first vs hybrid

## Recommended choice: **Hybrid, but evaluation-first**

### Why not pure method-first?
A pure method paper is riskier right now because the 2025 literature already has strong baselines in:
- relearning robustness,
- localization,
- selective token forgetting,
- correlated-knowledge control,
- practical LoRA-based pipelines.

To beat all of these cleanly, we would need a strong and well-implemented method plus a very convincing evaluation stack.

### Why not pure evaluation-only?
A pure benchmark paper is viable, but reviewers may ask for at least a minimal intervention or protocol showing how the new evaluation changes conclusions or helps design better unlearning policies.

### Why hybrid is strongest
A **hybrid, evaluation-led paper** gives us the best of both worlds:
- the **main novelty** is a stronger language-centric evaluation setting,
- the **secondary novelty** is a lightweight selective unlearning protocol or diagnostic policy that performs better under that setting.

In short:
- **primary contribution**: better evaluation,
- **secondary contribution**: a selective intervention recipe that is simple enough to implement but strong enough to demonstrate the value of the evaluation.

---

## 4. Recommended thesis

### Main thesis
**Most current LLM unlearning progress is overstated because evaluation is too shallow, and selective forgetting should be judged under language-centric recovery settings rather than one-shot answer suppression alone.**

### Stronger version
A model has not truly unlearned if the target can still be recovered through:
- multi-turn dialogue,
- reasoning traces,
- multilingual transfer,
- or lightweight downstream adaptation,

even when direct one-shot prompts suggest successful forgetting.

---

## 5. Novelty claim

### Primary novelty
We introduce an **evaluation-faithful unlearning framework** that measures whether forgetting holds under:
1. **multi-turn recovery**,
2. **reasoning-trace recovery**,
3. **multilingual recovery**, and
4. **relearning after lightweight adaptation**.

### Secondary novelty
We propose a **selective unlearning protocol** that applies forgetting pressure only when target-relevant content is activated, rather than using uniformly global forgetting.

### Why this is genuinely new
The literature already covers each component separately:
- relearning robustness,
- reasoning-trace leakage,
- multilingual propagation,
- selective/localized forgetting,
- probability-aware evaluation.

But there is still a gap in **putting them together into one coherent, language-centric evaluation and intervention framework** tailored to what modern LLMs actually do.

---

## 6. Recommended paper structure

1. **Introduction**
   - Current LLM unlearning metrics overestimate progress.
   - Modern recovery channels are multi-turn, reasoning-based, and multilingual.
   - Selective unlearning is a better match than global forgetting.

2. **Related Work**
   - robustness/relearning
   - localization/selective forgetting
   - evaluation-faithfulness
   - reasoning/multilingual extensions

3. **Evaluation Framework**
   - multi-turn recovery tasks
   - reasoning-trace leakage tasks
   - multilingual recovery tasks
   - downstream retuning / lightweight relearning tasks

4. **Selective Unlearning Protocol**
   - lightweight intervention recipe
   - activation or token-triggered forgetting policy

5. **Experiments**
6. **Analysis**
7. **Limitations / ethics**

---

## 7. Baseline table

| Baseline | Why include it | Role in our paper |
|---|---|---|
| UAM (NeurIPS 2025) | retain-aware optimization backbone | optimization-aware baseline / conceptual anchor |
| SAM-based relearning-robust unlearning (ICML 2025) | strong robustness baseline | compare on downstream recovery |
| ILU / invariance-based unlearning (ICML 2025) | unseen downstream tuning robustness | stronger robustness reference |
| Adaptive Localization (ICML 2025) | top-tier localized unlearning | compare against localized intervention |
| Reasoning Model Unlearning (EMNLP 2025) | reasoning-trace-aware method | compare on trace leakage |
| OBLIVIATE (EMNLP 2025) | practical strong method | pragmatic baseline |
| Selective Unlearning (Findings EMNLP 2025) | token-level selectivity | localized NLP baseline |
| UIPE (Findings EMNLP 2025) | correlated-knowledge forgetting | indirect leakage baseline |

### Minimal baseline set if compute is tight
- UAM
- SAM robust unlearning paper
- Adaptive Localization
- Reasoning Model Unlearning
- Selective Unlearning

---

## 8. Dataset table

| Dataset / source | Why use it | What it tests |
|---|---|---|
| TOFU | classic controlled forget/retain benchmark | direct forgetting + utility |
| LUME | broader multitask unlearning benchmark | cross-task stability |
| WMDP | hazardous knowledge setting | safety-oriented forgetting |
| R-TOFU / reasoning-style forget set | reasoning trace leakage | answer vs reasoning recovery |
| Learn-and-Unlearn multilingual data | multilingual misinformation | cross-lingual recovery |
| New multi-turn recovery set built from existing forget targets | our likely main contribution | dialogue-based recovery |

### Recommended practical dataset mix
If we have to be realistic and focused:
1. **TOFU** for controlled comparison,
2. **WMDP** for hazardous knowledge,
3. **R-TOFU / reasoning-style split** for trace leakage,
4. **multilingual misinformation subset** for transfer,
5. **our new multi-turn recovery evaluation** built on top of one or two of the above.

---

## 9. Evaluation recipe table

| Evaluation axis | What to measure | Why it matters |
|---|---|---|
| Direct forget accuracy | target answer suppression / refusal / low probability | standard first-pass forgetting |
| Retain utility | retained QA, general benchmarks, perplexity/task accuracy | avoid catastrophic damage |
| Multi-turn recovery | can the target be recovered over a dialogue? | realistic language use |
| Reasoning-trace leakage | does CoT or stepwise reasoning reveal the target? | modern reasoning models |
| Multilingual recovery | can the target reappear in another language? | EMNLP-native gap |
| Correlated-knowledge recovery | can nearby facts reconstruct the target? | indirect leakage |
| Relearning robustness | does lightweight finetuning restore the target? | long-term robustness |
| Confidence/probability | is forgotten content still high-probability? | latent retention |

### Minimal metric bundle
- Forget score
- Retain score
- Multi-turn recovery rate
- Reasoning leakage rate
- Multilingual recovery rate
- Relearning recovery rate

---

## 10. Concrete recommendation for the intervention side

### Recommended method style
Do **not** attempt a huge brand-new training algorithm first.
Instead use a lightweight **selective intervention protocol**:
- trigger forgetting only when target-relevant cues are present,
- localize to target-relevant tokens/spans/representations where possible,
- combine with a retain-aware loss or policy inspired by UAM / selective unlearning.

### Why this is the right scope
This keeps the method contribution believable and allows the evaluation contribution to remain the main story.

---

## 11. 8-week execution sketch

### Week 1
- finalize thesis and scope
- freeze core reading set
- define exact benchmark mix

### Week 2
- implement dataset adapters and prompt generators
- build multi-turn recovery protocol
- build reasoning-trace evaluation harness

### Week 3
- build multilingual recovery protocol
- implement confidence/probability evaluation
- reproduce 2–3 core baselines on a small scale

### Week 4
- implement selective intervention prototype
- run first ablations on one dataset

### Week 5
- run main baseline comparisons
- run robustness/relearning experiments

### Week 6
- run multilingual + reasoning analyses
- perform error analysis and failure clustering

### Week 7
- write paper draft
- refine figures and tables
- strengthen limitations/ethics section

### Week 8
- final experiments for missing ablations
- polish writing
- submission packaging

---

## 12. Final recommendation

### If we want the highest chance of a strong EMNLP submission
Choose:
> **Hybrid, evaluation-first paper**

### Short justification
- safer than a pure method paper,
- more novel than a pure benchmark paper,
- uses the latest 2025–2026 literature in exactly the direction the field is moving,
- and matches our strongest current intuition: **selective forgetting should be tested under realistic language-centric recovery settings**.
