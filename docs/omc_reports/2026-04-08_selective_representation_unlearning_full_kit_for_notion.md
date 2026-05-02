# 2026-04-08 Full Kit — Selective Representation Unlearning for Multi-turn Recovery

## Executive summary
I expanded the selected topic into a full working research kit for an EMNLP paper:
- method design,
- formalized objective,
- experiment plan,
- paper skeleton,
- multi-turn recovery templates,
- baseline run order.

## Working topic
**Selective Representation Unlearning for Large Language Models under Multi-turn Recovery**

## One-line pitch
Detect when the conversation is activating target knowledge, apply forgetting pressure only to the relevant token/layer representations, and evaluate whether the target still re-emerges across later turns.

## New artifacts
- `emnlp/topic_selective_representation_unlearning/README.md`
- `emnlp/topic_selective_representation_unlearning/method_design.md`
- `emnlp/topic_selective_representation_unlearning/method_formalization.md`
- `emnlp/topic_selective_representation_unlearning/experiment_plan.md`
- `emnlp/topic_selective_representation_unlearning/paper_skeleton.md`
- `emnlp/topic_selective_representation_unlearning/multi_turn_prompt_templates.md`
- `emnlp/topic_selective_representation_unlearning/baseline_run_order.md`

## Method summary
### A. Forget-relevance scorer
Estimate whether each token/layer hidden state is carrying target knowledge.

### B. Selective update
Apply forgetting pressure only where that score is high.

### C. Multi-turn consistency
Penalize target recovery across follow-up turns, not just the first response.

## Core objective
`L_total = lambda_f L_forget + lambda_r L_retain + lambda_s L_sparse + lambda_m L_multi_turn`

where:
- `L_forget` suppresses target recall,
- `L_retain` preserves utility,
- `L_sparse` keeps the intervention localized,
- `L_multi_turn` reduces dialogue recovery.

## Main experiment framing
### Primary setting
- TOFU or LUME with custom multi-turn wrappers

### Core baselines
- NPO
- OBLIVIATE
- Adaptive Localization
- Not Every Token Needs Forgetting
- UIPE
- robust relearning baseline
- optional reasoning-aware baseline

### Main evaluation axes
1. direct forgetting
2. retain utility
3. multi-turn recovery
4. paraphrase recovery
5. correlated knowledge recovery
6. post-finetuning relearning robustness

## Paper skeleton summary
### Main thesis
Broad global unlearning is too destructive and too easy to game under one-shot evaluation. Selective representation-level forgetting, paired with multi-turn recovery evaluation, gives a more realistic forget/retain testbed.

### Core contribution set
1. identify multi-turn recovery as a central failure mode,
2. propose selective representation unlearning,
3. define a recovery-aware evaluation protocol,
4. show improved forget/retain/recovery tradeoff.

## Immediate execution order
1. build multi-turn wrappers on one primary benchmark,
2. reproduce NPO / OBLIVIATE / Adaptive Localization,
3. train LoRA-based SRU-MR MVP,
4. run localization and multi-turn ablations,
5. validate on one secondary axis (reasoning or multilingual).
