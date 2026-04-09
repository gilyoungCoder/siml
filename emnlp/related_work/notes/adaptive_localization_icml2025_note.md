# Detailed Note — Adaptive Localization of Knowledge Negation for Continual LLM Unlearning (ICML 2025)

- Venue: ICML 2025
- Source: https://proceedings.mlr.press/v267/wuerkaixi25a.html

## 1. One-sentence summary
This paper says broad global unlearning is too destructive, especially when deletion requests arrive repeatedly, so we should localize the update to the parts of the model most responsible for the target knowledge.

## 2. Why it is so relevant to us
Among recent top-tier papers, this one is one of the cleanest matches to our high-level instinct:
- intervene only where needed,
- preserve utility elsewhere.

## 3. Core idea in plain English
Instead of telling the whole model to “forget,” first identify the model regions most involved in the target knowledge, then concentrate the negation/unlearning there.

## 4. Why continual matters
In real deployments, unlearning is not a one-time event. New requests keep coming. If each request changes the whole network, performance will degrade over time. Localization is a natural answer.

## 5. What to learn from this paper
For us, the most important lesson is not only the exact method, but the framing:
- **continual unlearning needs locality**.

That strengthens our case for selective/token-/representation-aware approaches.

## 6. Main limitation
Knowledge in LLMs is distributed and overlapping. No localization method is perfect, and correlated knowledge can still survive outside the main localized region.

## 7. Bottom line
If we want to propose selective unlearning, this is one of the strongest modern papers to position ourselves against or build on.
