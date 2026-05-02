# 2026-04-08 EMNLP Proposal Package — LLM Unlearning

## Executive summary
I completed the next four planning steps:
1. polished the related-work draft,
2. fixed a concrete problem statement,
3. chose the paper framing,
4. built baseline / dataset / evaluation tables.

## Final decision
### Recommended paper type
**Hybrid, evaluation-first paper**

### Why
- a pure method paper is riskier because 2025 already has many strong baselines,
- a pure evaluation paper is viable but benefits from at least a lightweight intervention story,
- an evaluation-led hybrid gives us the best balance of novelty and feasibility.

## Problem statement
Current LLM unlearning methods are usually evaluated in settings that are too narrow: one-shot prompts, direct answer extraction, and mostly monolingual benchmarks. Many methods therefore appear successful even when target knowledge can still be recovered through multi-turn dialogue, reasoning traces, correlated facts, downstream fine-tuning, or cross-lingual transfer. At the same time, globally applied forgetting updates often damage utility because they do not distinguish target-relevant knowledge from irrelevant context.

Our core question is:
> How can we evaluate and enforce selective LLM unlearning so that forgetting remains effective under realistic language-centric recovery settings—especially multi-turn interaction, reasoning traces, and multilingual transfer—while preserving retain-side utility?

## Primary novelty claim
We propose an **evaluation-faithful selective unlearning framework** that measures whether forgetting holds under:
- multi-turn recovery,
- reasoning-trace recovery,
- multilingual recovery,
- and lightweight relearning after adaptation.

## Secondary novelty claim
We pair this with a lightweight **selective unlearning protocol** that applies forgetting pressure only when target-relevant content is activated, instead of relying on uniformly global forgetting.

## Baseline table
- UAM
- SAM-based relearning-robust unlearning
- ILU / invariance-based unlearning
- Adaptive Localization
- Reasoning Model Unlearning
- OBLIVIATE
- Selective Unlearning
- UIPE

## Dataset recipe
Recommended mix:
- TOFU
- WMDP
- R-TOFU or reasoning-style forget set
- multilingual misinformation subset
- a new multi-turn recovery evaluation built from existing forget targets

## Evaluation recipe
Track:
- direct forget score,
- retain utility,
- multi-turn recovery,
- reasoning-trace leakage,
- multilingual recovery,
- correlated-knowledge recovery,
- relearning robustness,
- confidence/probability retention.

## Strongest title direction
**Evaluation-Faithful Selective Unlearning for LLMs under Multi-Turn, Reasoning, and Multilingual Recovery**

## Best project direction
Preferred:
**evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**

Backup:
**selective representation unlearning with robustness-to-relearning evaluation**
