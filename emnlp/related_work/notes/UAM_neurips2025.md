# Unlearning-Aware Minimization (UAM)

- **Venue**: NeurIPS 2025
- **Authors**: Hoki Kim, Keonwoo Kim, Sungwon Chae, Sangwon Yoon
- **Primary source**: https://neurips.cc/virtual/2025/poster/116406
- **OpenReview**: https://openreview.net/forum?id=e3abccbf80

## Why this matters for us
This looks like the most likely paper the user meant by **UAM**.
It is **not purely an NLP paper**, but it is still relevant because it frames unlearning as a **retain-aware min-max optimization problem** and explicitly reports **LLM experiments** on WMDP-Bio and WMDP-Cyber.

## High-level idea
UAM treats approximate unlearning as a two-sided objective:
1. perturb parameters to make forgetting stronger,
2. use the resulting gradients to better preserve retain performance.

The paper's core pitch is that standard approximate unlearning methods often land in poor optima: either forgetting is weak, or retain performance collapses. UAM claims that a min-max view finds better tradeoffs than plain negative-gradient or simple finetuning baselines.

## Relevance to LLM unlearning
Even though the formulation is broader than NLP, the paper is useful as a **design pattern** for LLM unlearning:
- do not optimize forgetting alone,
- explicitly optimize for the **forget/retain frontier**,
- build the method around robustness to bad local optima.

## What to borrow vs. what not to borrow
### Worth borrowing
- retain-aware optimization mindset
- robustness-first framing
- the idea that naive forget-only optimization is often insufficient

### Probably not enough by itself for EMNLP
- evaluation is still closer to general ML / safety MU than rich language interaction
- no obvious emphasis on multi-turn dialogue, paraphrase recovery, reasoning traces, or multilingual transfer

## EMNLP-facing takeaway
UAM is a strong **reference point / inspiration**, but an EMNLP paper should likely extend beyond its setting by adding:
- explicitly language-centric robustness evaluation,
- more localized intervention (activation/token/turn), or
- reasoning-trace / multi-turn forgetting analysis.
