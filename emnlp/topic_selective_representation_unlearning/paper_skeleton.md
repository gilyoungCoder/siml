# Paper Skeleton — Selective Representation Unlearning for Multi-turn Recovery

## 1. Title candidates
1. **Selective Representation Unlearning for Large Language Models under Multi-turn Recovery**
2. **Forget Only When It Matters: Selective Representation Unlearning for Large Language Models**
3. **Beyond One-Shot Forgetting: Selective LLM Unlearning under Multi-turn Recovery**
4. **Recovery-Aware Selective Unlearning for Large Language Models**

---

## 2. Abstract skeleton
### Background
LLM unlearning aims to remove the influence of unwanted knowledge without retraining from scratch.

### Problem
Existing methods are often evaluated on one-shot prompts and rely on broad updates, so target knowledge may still re-emerge across later conversational turns while utility suffers.

### Method
We propose a selective representation unlearning method that estimates which token/layer representations are actively carrying target knowledge and applies forgetting pressure only to those locations.

### Evaluation
We introduce or adapt a multi-turn recovery evaluation protocol that tests whether forgotten content reappears through follow-up turns, paraphrases, and indirect cues.

### Result claim
Compared with global unlearning baselines, the proposed method yields a better forget/retain tradeoff and lower recovery under multi-turn interaction.

### Significance
This suggests that realistic LLM unlearning requires both selective intervention and recovery-aware evaluation.

---

## 3. Intro skeleton
### Paragraph 1
Motivate why post-hoc LLM unlearning matters.

### Paragraph 2
Explain weakness of current methods: answer-level, one-shot, often globally destructive.

### Paragraph 3
Explain realistic failure mode: knowledge comes back through multi-turn interaction.

### Paragraph 4
Our thesis: selective representation-level intervention is better suited than coarse global forgetting.

### Paragraph 5
Contributions list.

---

## 4. Contributions draft
1. We identify **multi-turn recovery** as a critical failure mode for LLM unlearning.
2. We propose a **selective representation unlearning** method that localizes forgetting pressure to target-relevant token/layer representations.
3. We provide a **recovery-aware evaluation protocol** that measures whether target knowledge re-emerges through dialogue continuation.
4. We show improved forget/retain tradeoff and reduced recovery compared with strong baselines.

---

## 5. Method section skeleton
### 5.1 Problem setup
Define forget set, retain set, and recovery-oriented evaluation.

### 5.2 Forget-relevance scoring
Define target-representation relevance gate.

### 5.3 Selective representation update
Define masked forget loss + retain loss + sparsity term.

### 5.4 Multi-turn consistency objective
Define recovery-aware training signal.

### 5.5 Implementation details
LoRA/full finetuning, selected layers, prototype construction.

---

## 6. Experiment section skeleton
### 6.1 Benchmarks and settings
TOFU/LUME/WMDP + custom multi-turn wrappers.

### 6.2 Baselines
NPO, OBLIVIATE, Adaptive Localization, Selective Unlearning, UIPE, etc.

### 6.3 Main results
forget/retain + multi-turn recovery.

### 6.4 Ablations
masking granularity, prototype type, multi-turn loss, correlated-knowledge augmentation.

### 6.5 Case studies
example dialogues where global forgetting fails but selective forgetting succeeds.

---

## 7. Risks to discuss in the paper
- evaluation-template overfitting
- hidden recovery outside tested prompts
- localization errors
- increased complexity vs simpler baselines
