# EMNLP LLM Unlearning Sprint

## Current objective
- Target area: **LLM unlearning**
- Immediate priority: **topic framing + heavy related-work sweep**
- Secondary priority: derive **1-3 EMNLP-suitable paper directions** from the survey

## Why this fits our repo
Our existing repo philosophy is roughly:
- **training-free / low-cost intervention**
- **adaptive control** (when to intervene, where to intervene)
- **utility-aware safety tradeoff**
- **robust evaluation under stronger attacks / false-positive checks**

For LLMs, the analogous research question is not diffusion guidance itself, but:
> can we make unlearning more **adaptive, localized, and evaluation-faithful** while preserving downstream NLP utility?

## Important note on `$team`
I tried launching `omx team`, but OMX team startup is currently blocked by the repo's pre-existing dirty workspace state:
- `leader_workspace_dirty_for_worktrees: ... commit_or_stash_before_omx_team`

So for now I am treating this as a **research-first planning sprint** and will continue with:
1. local planning/docs in this workspace,
2. literature collection from primary sources,
3. then either a clean auxiliary worktree or native subagent fan-out for the full survey pass.

## Overnight plan
1. **Lock the topic space**
   - privacy/data deletion unlearning
   - copyrighted long-form content unlearning
   - hazardous knowledge unlearning
   - multilingual / reasoning-trace / multi-turn unlearning
2. **Survey benchmarks first**
   - TOFU, MUSE, WMDP, LUME, SemEval-2025 Task 4
3. **Survey strong method families**
   - gradient/loss-based: NPO, OBLIVIATE, UIPE, Unilogit
   - representation/subspace: RMU, R²MU, SSPU
   - black-box / logit-offset / editing style: Offset Unlearning, REVS
   - robustness-focused: SAM/smoothness line, UAM-related perspective
4. **Extract gaps that sound EMNLP-native**
   - multilingual transfer of forgotten facts
   - multi-turn leakage / conversational recovery
   - reasoning-trace leakage even when answers are removed
   - evaluation contamination: attacks that accidentally reteach
   - selective forgetting of target-bearing spans instead of full-document forgetting
5. **Turn gaps into paper ideas**
   - one benchmark-heavy idea
   - one method-heavy idea
   - one hybrid benchmark+method idea

## Candidate topic directions (current shortlist)

### A. Multilingual propagation-aware unlearning
- Motivation: EMNLP-friendly, grounded in multilingual behavior.
- Core question: if a fact is forgotten in English, does it remain recoverable in Korean/Chinese/low-resource languages or through translation chains?
- Why promising: EMNLP 2025 already shows English-only unlearning is insufficient in multilingual settings.

### B. Reasoning-trace unlearning, not just answer unlearning
- Motivation: strong EMNLP fit because it touches reasoning data formats, chain-of-thought leakage, and downstream reasoning quality.
- Core question: can a model still leak the sensitive reasoning path even if the final answer is suppressed?
- Why promising: recent EMNLP work suggests current methods miss this badly.

### C. Evaluation-faithful unlearning for multi-turn dialogue
- Motivation: very EMNLP-native.
- Core question: how should we evaluate unlearning in dialogue when adversarial probing can *reteach* the model, and when leakage emerges over multiple turns instead of one-shot QA?
- Why promising: existing evaluations appear too static and single-turn.

### D. Related-knowledge-aware selective unlearning
- Motivation: closest high-level transfer from our repo.
- Core question: instead of globally forgetting all tokens/documents, can we estimate **when** a query truly activates the forget target and **where** in representation/token space the forgetting pressure should be applied?
- Why promising: UIPE/target-related knowledge papers hint that forgetting failure often comes from correlated knowledge pathways.

## Files in this folder
- `related_work/paper_inventory.md` — initial paper map + why each paper matters
- `related_work/topic_map.md` — benchmark/method/gap map for topic selection

