# Detailed Note — Towards LLM Unlearning Resilient to Relearning Attacks (ICML 2025)

- Venue: ICML 2025
- Source: https://proceedings.mlr.press/v267/fan25e.html

## 1. One-sentence summary
This paper makes a crucial point: if a model can quickly recover forgotten knowledge after later tuning, the unlearning was never very strong in the first place.

## 2. Why this paper changed the conversation
Earlier work often evaluated unlearning only at the moment right after the forget update. This paper asks the much harder and more realistic question:
- what happens after future training?

That is a stronger notion of success.

## 3. Core intuition
The paper borrows ideas from sharpness-aware optimization. Informally:
- brittle parameter states are easy to move out of,
- so brittle forgetting is easy to undo.

The goal is to place the model in a parameter region where the forgotten target is harder to recover.

## 4. Why it matters for us
This is one of the best papers to cite if we want to argue that:
- evaluation must include **future recovery / relearning checks**,
- not just immediate post-unlearning metrics.

## 5. Main limitation
It is still mainly an optimization-level robustness story. It does not directly solve rich language-specific leakage like:
- dialogue accumulation,
- compositional paraphrase chains,
- cross-lingual recovery.

## 6. Bottom line
This paper should probably be one of our anchor citations if we build anything around robust unlearning evaluation.
