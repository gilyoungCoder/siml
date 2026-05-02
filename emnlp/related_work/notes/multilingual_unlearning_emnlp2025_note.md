# Detailed Note — Learn and Unlearn: Addressing Misinformation in Multilingual LLMs (EMNLP 2025)

- Venue: EMNLP 2025
- Source: https://aclanthology.org/2025.emnlp-main.516/

## 1. One-sentence summary
The paper shows that misinformation can survive or reappear across languages, so English-only unlearning is not enough in multilingual LLMs.

## 2. Why this paper matters
A lot of unlearning evaluation still assumes the target problem lives in one language. This paper shows that multilingual transfer changes the problem.

## 3. Core idea in plain English
If the model learned a bad fact in language A, simply unlearning it in English does not guarantee the model forgot it globally. The knowledge may still be reachable in another language.

## 4. Why this matters for us
This is a strong argument that current benchmarks are incomplete. Even if we do not choose a multilingual paper, we can cite this as evidence that language transfer creates hidden recovery paths.

## 5. Main limitation
Multilingual evaluation is expensive, and no paper can cover all languages well.

## 6. Bottom line
This paper makes multilingual propagation a serious unlearning issue, not a niche extension.
