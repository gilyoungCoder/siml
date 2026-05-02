# Detailed Note — Not Every Token Needs Forgetting (Findings of EMNLP 2025)

- Venue: Findings of EMNLP 2025
- Source: https://aclanthology.org/2025.findings-emnlp.96/

## 1. One-sentence summary
This paper says we should not force the model to forget every token in a forget document, because only a subset of tokens are actually central to the unwanted knowledge.

## 2. Why this is important
It gives perhaps the simplest and most intuitive form of **selective unlearning** in the recent NLP literature.

## 3. Core idea in plain English
Some tokens matter a lot for the target knowledge, while many others are generic context. If we can identify the important tokens, we can forget more precisely and preserve more utility.

## 4. Why it matters for us
This paper is directly aligned with the idea that forgetting should be **localized** rather than global.

## 5. Main limitation
Token-level importance is not the whole story. Knowledge can also live in relationships across tokens, spans, and latent features.

## 6. Bottom line
This is one of the best recent citations for a selective-unlearning argument.
