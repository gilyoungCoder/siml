# Detailed Note — Reasoning Model Unlearning (EMNLP 2025)

- Venue: EMNLP 2025
- Source: https://aclanthology.org/2025.emnlp-main.220/

## 1. One-sentence summary
This paper argues that unlearning must erase not just final answers, but also the reasoning traces that can still leak the target information.

## 2. Why it is important
As reasoning models become stronger, answer-only evaluation becomes less trustworthy. A model may hide the final answer but still expose it through intermediate steps.

## 3. Core idea in plain English
The paper's core message is:
- an LLM has not really unlearned if it can still reconstruct the target through its reasoning path.

That makes reasoning traces themselves part of the unlearning problem.

## 4. Why this is especially important for EMNLP
This is a very language-native failure mode. It is not just about parameter robustness; it is about how information appears in generated reasoning.

## 5. What we should take from it
If we build a paper around multi-turn recovery or reasoning leakage, this paper becomes one of the most direct precedents.

## 6. Main limitation
Reasoning traces are unstable and model-dependent. Measuring them reliably is harder than measuring direct QA outputs.

## 7. Bottom line
This is probably one of the most important recent papers for an EMNLP-facing unlearning project.
