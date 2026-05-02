# Topic Map: Where we can plausibly contribute

## Option 1: Evaluation paper
### Working title sketch
**Evaluation-Faithful LLM Unlearning for Multi-turn and Cross-lingual Recovery**

### Why this is attractive
- EMNLP likes datasets/benchmarks/evaluation when the problem is language-native.
- Existing papers already expose cracks:
  - static one-shot evaluation,
  - adversarial evaluation that may reteach,
  - poor multilingual coverage,
  - weak treatment of dialogue/reasoning traces.

### Core thesis
Current LLM unlearning benchmarks overestimate progress because they ignore:
1. **multi-turn recovery**,
2. **cross-lingual recovery**,
3. **reasoning-trace leakage**,
4. **evaluation-time information injection**.

### Minimal viable deliverable
- new evaluation protocol, not necessarily a new unlearning method
- benchmark adapters for TOFU / MUSE / WMDP / LUME
- strong analysis paper even if method contribution is light

## Option 2: Method paper
### Working title sketch
**Selective Representation Unlearning for LLMs: Forget When the Target is Activated, Not Everywhere**

### Why this is attractive
- Best transfer from our current repo philosophy.
- Fits the intuition behind selective token forgetting, UIPE, RMU, SSPU.

### Core thesis
Global document-level or full-response unlearning is too blunt. We should:
1. estimate **when** a prompt/query genuinely activates forget-target knowledge,
2. estimate **where** in representation/token/subspace that target is active,
3. apply forgetting pressure only there.

### Risks
- Harder to beat strong 2025 baselines quickly.
- Representation-localization ideas are now crowded.

## Option 3: Hybrid paper
### Working title sketch
**Reasoning- and Dialogue-Aware Unlearning for LLMs**

### Why this is attractive
- Feels very EMNLP-native.
- Goes beyond fact QA into trajectories and interaction.

### Core thesis
A model has not truly unlearned if it can still recover the target over:
- a chain of paraphrases,
- a multi-turn conversation,
- a reasoning trace,
- or a translation chain.

### Strongest differentiator
Tie a method and evaluation together:
- evaluation: multi-turn + reasoning + cross-lingual recovery
- method: selective forgetting over trace spans / representation routes

## My current ranking
1. **Evaluation-heavy paper** — safest overnight direction
2. **Hybrid reasoning/dialogue-aware paper** — highest upside for EMNLP
3. **Pure method paper** — viable, but more baseline pressure

## Immediate next reading priority
1. TOFU
2. MUSE
3. WMDP + RMU
4. LUME
5. Existing Adversarial Evaluations Are Inconclusive
6. Learn and Unlearn (multilingual)
7. Reasoning Model Unlearning
8. UIPE / OBLIVIATE / Reveal and Release
