# 2026-04-08 EMNLP LLM Unlearning Proposal Package

## Executive summary
I completed the four next-step deliverables:
1. a polished latest-focused related-work section,
2. a concrete problem statement,
3. a decision between evaluation-first vs method-first vs hybrid,
4. baseline / dataset / evaluation recipe tables.

## Final decision
### Recommended paper type
**Hybrid, evaluation-led**

### Recommended short description
A paper that demonstrates current LLM unlearning wins are fragile under richer recovery settings, then shows that a selective unlearning strategy performs better under those tests while preserving utility.

### Best direction
**Evaluation-faithful selective unlearning under multi-turn, reasoning, and multilingual recovery**

## Why this direction wins
- latest robustness papers show forgetting must survive relearning,
- localization papers show global updates are too blunt,
- evaluation papers show current metrics overestimate progress,
- reasoning and multilingual papers show answer-level English-only forgetting is no longer sufficient.

## Problem statement
Current LLM unlearning work often reports success under narrow, one-shot evaluation settings, yet recent 2025-2026 papers show that supposedly forgotten knowledge can remain recoverable through correlated facts, reasoning traces, multilingual transfer, or later fine-tuning. At the same time, broad unlearning updates frequently damage retained utility because they are applied too globally. We therefore need an LLM unlearning framework that asks not only whether the target disappears immediately, but whether it stays gone under realistic language-mediated recovery attempts while preserving unrelated utility.

## Related-work synthesis
### Robustness / relearning
Recent ICML 2025 papers show that brittle forgetting solutions can be undone by future model adaptation, motivating robustness-aware unlearning and retain-aware optimization. UAM strengthens this line by making the forget/retain tradeoff explicit.

### Localization / selective forgetting
Adaptive localization, selective token forgetting, and related-knowledge suppression all argue that broad global updates are too destructive. The field is moving toward interventions that target only the parts of the model or input most responsible for the unwanted knowledge.

### Evaluation faithfulness
Recent EMNLP and NeurIPS papers show that deterministic one-shot outputs can hide latent retention, correlated knowledge leakage, and overconfident hidden memory. Stronger evaluation is now a central problem, not a side issue.

### Reasoning / multilingual settings
Reasoning Model Unlearning and multilingual misinformation work show that final-answer suppression is no longer enough. Future unlearning systems must be tested under reasoning traces and cross-lingual transfer.

## Baseline table
| Baseline | Why include it |
|---|---|
| NPO | classic LLM unlearning baseline |
| OBLIVIATE | strong practical 2025 baseline |
| R²MU / Reasoning Model Unlearning | reasoning-aware baseline |
| Adaptive Localization | top-tier localization baseline |
| Selective Unlearning | direct selective-forgetting baseline |
| UIPE | correlated-knowledge baseline |
| SAM-style robust unlearning | relearning robustness baseline |
| UAM-inspired retain-aware variant | optimization principle baseline |

## Dataset table
| Dataset / source | Role |
|---|---|
| TOFU | controlled forget benchmark |
| MUSE | broad evaluation suite |
| WMDP | safety / hazardous-knowledge forgetting |
| R-TOFU or reasoning-oriented forget set | reasoning leakage |
| Learn-and-Unlearn multilingual setup | cross-lingual recovery |
| small custom multi-turn recovery set | main evaluation contribution |

## Evaluation recipe
- direct forgetting
- retain utility
- probabilistic leakage
- reasoning leakage
- multi-turn recovery
- paraphrase / correlation recovery
- multilingual recovery
- relearning robustness

## 8-week plan
1. lock problem framing and baseline shortlist
2. reproduce strongest baselines on one core benchmark
3. build recovery-faithful evaluation slices
4. run baseline recovery evaluations
5. implement lightweight selective method or intervention policy
6. run ablations
7. finalize tables/figures and writing
8. polish paper narrative and appendix
