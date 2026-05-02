# Core 2025-2026 Project Notes for Our LLM Unlearning Direction

This note is not a general survey. It is a working memo for **our** paper direction.
The question here is: *which recent papers should most influence our design choices?*

## 1. UAM (NeurIPS 2025)
**What to steal:** the idea that forgetting and retention should be optimized **jointly**, not sequentially.

Why it matters for us:
- It gives a principled story for why naive forget-only objectives are weak.
- It helps us justify any method that explicitly tries to protect retain utility while forgetting.

What not to copy blindly:
- UAM alone is too generic for an EMNLP paper.
- If we use UAM-like optimization, we still need stronger language-centric evaluation.

## 2. Relearning-robust papers (ICML 2025)
### SAM-style robust unlearning
**What to steal:** forgetting should survive later fine-tuning.

### Invariance / ILU
**What to steal:** robustness should generalize across future update environments, not just one attack.

Why they matter for us:
- They justify a strong evaluation section built around downstream re-tuning and recovery.
- They raise the bar for what counts as “real” unlearning.

## 3. Adaptive Localization of Knowledge Negation (ICML 2025)
**What to steal:** broad updates are too blunt; target the relevant subspace/region more carefully.

Why it matters for us:
- This is the cleanest top-tier method-side support for our selective-intervention instinct.
- If we build a method, this paper is a direct bridge.

## 4. Reasoning Model Unlearning (EMNLP 2025)
**What to steal:** evaluate hidden reasoning traces, not just final answers.

Why it matters for us:
- It gives us a language-native failure mode.
- It makes our story much more EMNLP-friendly than generic MU optimization alone.

## 5. Fully Probabilistic Perspective (EMNLP 2025)
**What to steal:** evaluation should look at probability mass and leakage risk, not only sampled outputs.

Why it matters for us:
- It gives us a stronger evaluation methodology.
- It pairs naturally with multi-turn and paraphrase recovery.

## 6. Learn and Unlearn (EMNLP 2025)
**What to steal:** cross-lingual propagation is real and can invalidate English-only unlearning claims.

Why it matters for us:
- If we want a distinct EMNLP angle, multilingual recovery is one of the best options.
- Even if we do not go fully multilingual, this paper proves current evaluation is incomplete.

## 7. Selective Unlearning + UIPE (Findings of EMNLP 2025)
**What to steal:**
- Not Every Token Needs Forgetting → only some tokens are responsible.
- UIPE → related knowledge can reconstruct the target.

Why they matter for us:
- Together they suggest the right framing is not just “forget target x”.
- The real problem is “forget the target without leaving reconstruction routes and without damaging everything else.”

## 8. Practical conclusion for our method/eval design
If we build a paper now, the strongest combined recipe looks like this:

### Method side
- UAM-style retain-aware objective
- selective/localized intervention
- explicit robustness-to-relearning checks

### Evaluation side
- multi-turn recovery
- reasoning-trace leakage
- correlated-knowledge recovery
- multilingual recovery (if feasible)
- probabilistic leakage measures

## 9. Strongest paper thesis candidate right now
**Evaluation-faithful selective unlearning for LLMs under multi-turn, reasoning, and multilingual recovery**

Why this still looks best:
- method papers are already moving toward selectivity,
- evaluation papers are already exposing weaknesses,
- the combination still looks underexplored,
- EMNLP reviewers will understand the language-centric importance immediately.
