# 2026-04-08 Topic Proposal — Selective Representation Unlearning for Multi-turn Recovery

## Executive summary
We selected one concrete methodological EMNLP direction:

**Selective Representation Unlearning for Large Language Models under Multi-turn Recovery**

### One-line pitch
Instead of globally forcing an LLM to forget, detect when the conversation is activating target knowledge and apply forgetting pressure only to the relevant token/layer representations, then test whether the target still re-emerges across later turns.

## Why this topic
- Strong match to the latest 2025-2026 literature.
- Methodological contribution is clear.
- EMNLP fit is strong because the key failure mode is language-native: dialogue recovery.
- High-level match to our repo philosophy: when / where / intervene.

## Core hypothesis
Existing unlearning updates are often too global. They suppress the first direct answer, but leave enough residual knowledge in latent representations that the target can be reconstructed in later turns. If we can detect which token/layer representations are actually carrying target knowledge during generation, then applying forgetting pressure only there will produce a better forget/retain tradeoff and stronger resistance to multi-turn recovery.

## Method sketch
### A. Forget-relevance scorer
Estimate whether each token/layer hidden state is involved in target-knowledge activation.

### B. Selective unlearning update
Use the relevance score as a soft mask so that forget pressure is concentrated on target-relevant representations rather than applied globally.

### C. Multi-turn recovery training signal
Train and evaluate on short dialogue chains where the target may reappear via paraphrases, hints, or indirect references.

## Paper type
### Recommended: Hybrid, evaluation-first
- main framing: stronger evaluation under realistic recovery settings
- method contribution: lightweight selective representation unlearning

## Baseline set
- NPO
- OBLIVIATE
- Reasoning Model Unlearning / R²MU
- Adaptive Localization
- Not Every Token Needs Forgetting
- UIPE
- robust relearning baseline
- UAM-style retain-aware reference

## Dataset scope
- TOFU
- LUME
- WMDP
- reasoning-focused setting
- multilingual recovery setting
- custom multi-turn dialogue templates

## Minimum viable evaluation suite
1. direct forgetting
2. retain utility
3. multi-turn recovery
4. paraphrase recovery
5. correlated knowledge recovery
6. post-finetuning relearning robustness

## Title candidates
1. **Selective Representation Unlearning for Large Language Models under Multi-turn Recovery**
2. **Forget Only When It Matters: Selective Representation Unlearning for Large Language Models**
3. **Beyond One-Shot Forgetting: Selective LLM Unlearning under Multi-turn Recovery**
4. **Recovery-Aware Selective Unlearning for Large Language Models**

## Immediate next steps
1. freeze title direction,
2. implement multi-turn recovery evaluation harness,
3. build contrastive forget prototype,
4. train LoRA-based selective method MVP,
5. compare against strong global and selective baselines.
