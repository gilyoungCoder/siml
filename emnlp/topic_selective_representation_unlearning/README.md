# Topic Package — Selective Representation Unlearning for Multi-turn Recovery

## Working topic
**Selective Representation Unlearning for Large Language Models under Multi-turn Recovery**

## One-line pitch
Instead of globally forcing an LLM to forget, we detect when the conversation is activating target knowledge and apply unlearning pressure only to the relevant representations, then test whether the target still re-emerges across follow-up turns.

## Why this topic
- Strong fit to the latest 2025-2026 literature.
- Methodological contribution is clear.
- EMNLP fit is strong because the failure mode is language-native: dialogue recovery.
- High-level match to our repo philosophy: **when / where / intervene**.

## Files in this package
- `method_design.md` — concrete method sketch and objective
- `experiment_plan.md` — baselines, datasets, evaluation, ablations
- `paper_skeleton.md` — title/abstract/section skeleton

## Current recommendation
This should be treated as a **hybrid paper**:
- evaluation-first in framing,
- method contribution via selective representation unlearning,
- multi-turn recovery as the signature stress test.
