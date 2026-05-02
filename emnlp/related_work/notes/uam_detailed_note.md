# UAM Detailed Note — Unlearning-Aware Minimization (NeurIPS 2025)

- Venue: NeurIPS 2025
- Source: https://neurips.cc/virtual/2025/poster/116406
- OpenReview: https://openreview.net/forum?id=e3abccbf80

## 1. One-sentence summary
UAM argues that machine unlearning should be optimized with an explicit **forget-versus-retain tradeoff objective**, rather than by pushing only on forgetting loss.

## 2. What problem it is trying to fix
A lot of approximate unlearning methods fail in one of two ways:
- they forget too weakly, or
- they forget strongly but damage the model's useful knowledge.

UAM says this is not just bad tuning—it is often a consequence of optimizing the wrong objective.

## 3. Core intuition in simple language
Imagine trying to erase one chapter from a student's memory without harming the rest of what they learned. If your only goal is “make them fail the chapter exam,” you may end up damaging many nearby skills. UAM says the learning rule itself should include a force that protects the retained material while unlearning the target.

## 4. Why this paper matters
Even though UAM is broader than just NLP, it is very valuable for LLM work because it gives a clean design principle:
- unlearning should be **retain-aware by construction**.

That principle can be reused whether the final method is:
- gradient-based,
- PEFT-based,
- representation-based,
- or evaluation-driven.

## 5. What to borrow for our project
Most useful part:
- the optimization philosophy, not necessarily the exact algorithm.

Possible reuse:
- combine a UAM-style retain-aware objective with selective/localized unlearning,
- or use it to justify why our evaluation must always report the forget/retain frontier rather than only forgetting scores.

## 6. Main limitation
UAM by itself does not solve language-specific leakage modes such as:
- multi-turn recovery,
- reasoning-trace leakage,
- multilingual propagation.

So for an EMNLP paper, UAM is best treated as a **methodological backbone**, not the full story.

## 7. Bottom line
If we write a method paper, UAM is one of the best high-level references for the statement:
> “Pure forget optimization is insufficient; strong unlearning must be retain-aware.”
