# High-Citation / Famous LLM Unlearning Papers (2026-04-08 snapshot)

> Purpose: prioritize the **most cited / most widely referenced** papers first, so the reading order tracks what the community is actually building on.
>
> Note: citation counts differ across indexes. Below I use the most accessible count I could verify from **Emergent Mind / Semantic Scholar-backed pages** when available, and **OpenAlex** as fallback. Treat the numbers as **approximate snapshot counts**, not absolute truth.

## Tier 0 — read these first no matter what

### 1) Knowledge Unlearning for Mitigating Privacy Risks in Language Models (ACL 2023)
- Link: https://arxiv.org/abs/2210.01504
- Approx. citations: **142** (Emergent Mind / Semantic Scholar-backed page)
- Why it is famous:
  - one of the earliest direct LM unlearning/privacy papers,
  - introduces the classic **gradient-ascent unlearning** framing,
  - still shows up everywhere as an early baseline/reference point.
- Why we care:
  - historically foundational,
  - helps explain why later papers try to move beyond blunt GA-style forgetting.

### 2) TOFU: A Task of Fictitious Unlearning for LLMs (2024)
- Link: https://arxiv.org/abs/2401.06121
- Approx. citations: **85** (Emergent Mind / Semantic Scholar-backed page)
- Why it is famous:
  - probably the single most recognizable **benchmark paper** in LLM unlearning,
  - everyone compares on TOFU,
  - basically defined the modern benchmark conversation.
- Why we care:
  - if we don't understand TOFU deeply, our paper framing will be weak.

### 3) In-Context Unlearning: Language Models as Few-Shot Unlearners (ICML 2024)
- Link: https://arxiv.org/abs/2310.07579
- Approx. citations: **74** (Emergent Mind / Semantic Scholar-backed page)
- Why it is famous:
  - black-box / no-weight-update flavor,
  - memorable and practically attractive,
  - often cited because it asks whether unlearning can happen at inference time.
- Why we care:
  - conceptually close to lightweight / selective intervention thinking,
  - important foil for any training-free or low-cost idea.

### 4) Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning (2024)
- Link: https://arxiv.org/abs/2404.05868
- Approx. citations: **65** (Emergent Mind / Semantic Scholar-backed page; Semantic Scholar snippet surfaced much larger index counts in related search)
- Why it is famous:
  - one of the most reused **optimization baselines**,
  - strong practical influence on later LLM unlearning papers,
  - central to the shift from naive GA to preference-style objectives.
- Why we care:
  - any method paper will almost certainly need to compare against NPO or explain why not.

### 5) Rethinking Machine Unlearning for Large Language Models (2024; later Nat. Mach. Intell. 2025)
- Link: https://arxiv.org/abs/2402.08787
- Approx. citations: **49** (Emergent Mind / Semantic Scholar-backed page)
- Why it is famous:
  - not just a method paper — it helped shape the field's **problem formulation**,
  - strong on definitions, scope, evaluation, and overlooked pitfalls.
- Why we care:
  - excellent framing paper for introduction / related work / threat model.

## Tier 1 — important benchmark / safety anchors

### 6) MUSE: Machine Unlearning Six-Way Evaluation for Language Models (2024)
- Link: https://arxiv.org/abs/2407.06460
- Approx. citations: **19** (Emergent Mind / Semantic Scholar-backed page)
- Why it matters more than the raw citation count suggests:
  - benchmark paper, so citations grow slower than method papers at first,
  - but it is one of the most important **evaluation** references after TOFU,
  - six-axis evaluation is exactly the kind of thing EMNLP reviewers care about.
- Why we care:
  - probably mandatory if we do an evaluation-heavy paper.

### 7) The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning (ICML 2024)
- Link: https://arxiv.org/abs/2403.03218
- Approx. citations: **13** (OpenAlex snapshot)
- Why it is famous:
  - very important in the **hazardous-knowledge / safety** branch,
  - also the home context for **RMU-style** representation unlearning.
- Why we care:
  - if we do safety/harmful-knowledge unlearning, this becomes core.

### 8) Who's Harry Potter? Approximate Unlearning in LLMs (2023)
- Link: https://arxiv.org/abs/2310.02238
- Approx. citations: **12** (OpenAlex snapshot)
- Why it is famous despite modest indexed counts:
  - one of the earliest and most memorable **case-study papers**,
  - frequently referenced historically even when later work evaluates on TOFU instead.
- Why we care:
  - historically foundational for the “copyrighted book forgetting” storyline.

## Tier 2 — important but not yet as citation-heavy

### 9) ULMR: Unlearning Large Language Models via Negative Response and Model Parameter Average (EMNLP Industry 2024)
- Link: https://aclanthology.org/2024.emnlp-industry.57/
- Approx. citations: low / not reliably indexed in the sources I checked
- Why still important:
  - practical industry-flavored EMNLP baseline,
  - useful if we want comparisons against lightweight response-rewriting style methods.

### 10) Unlearning-Aware Minimization (UAM) (NeurIPS 2025)
- Link: https://openreview.net/forum?id=e3abccbf80
- Approx. citations: too new for a stable count; treat as **new but influential-looking**
- Why it matters:
  - important as an **optimization idea source**,
  - retain-aware / min-max framing is valuable even if it is not a canonical NLP benchmark paper.

## Practical reading order I recommend
1. Knowledge Unlearning for Mitigating Privacy Risks in Language Models
2. TOFU
3. NPO
4. Rethinking Machine Unlearning for LLMs
5. In-Context Unlearning
6. MUSE
7. WMDP
8. Who's Harry Potter?
9. ULMR
10. UAM

## My take
If the goal is “read the famous ones first,” the **real core set** is:
- **Knowledge Unlearning (ACL 2023)**
- **TOFU**
- **NPO**
- **In-Context Unlearning**
- **Rethinking Machine Unlearning for LLMs**
- then **MUSE / WMDP** as the benchmark/eval/safety anchors.
