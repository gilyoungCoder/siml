# Who's Harry Potter? Approximate Unlearning in LLMs

- **Venue**: arXiv 2023
- **Primary source**: https://arxiv.org/abs/2310.02238

## Why this matters
This is one of the earliest influential LLM unlearning papers and a useful anchor for the whole literature map.

## Problem setup
The paper asks whether a pretrained LLM can approximately unlearn a copyrighted or unwanted corpus slice without full retraining. Their concrete testbed is removing Harry Potter knowledge from Llama2-7B.

## Main message
The paper shows that a relatively small amount of post-hoc finetuning can substantially reduce the model's ability to recall or generate target-domain content, while leaving broad benchmark utility relatively intact.

## Why it is important historically
- establishes LLM unlearning as a concrete post-training problem,
- gives a memorable case study with a recognizable corpus,
- makes the forget/retain tradeoff central.

## Limitations from today's perspective
Relative to more recent work, this setup likely under-tests:
- paraphrase robustness,
- multi-turn recovery,
- relearning after downstream finetuning,
- reasoning-trace leakage,
- multilingual/general-domain transfer.

## Relevance to our topic search
This paper is a foundational baseline, but a new EMNLP paper probably should **not** stop at this style of corpus-level approximate forgetting. It should use this as the starting point and then ask what remains recoverable under stronger language-centric attacks.
