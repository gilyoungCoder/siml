# Method Taxonomy (initial draft)

## 1. Data/loss-based finetuning unlearning
Representative papers:
- Who's Harry Potter?
- TOFU baselines
- NPO / SimNPO
- ULMR
- Reveal and Release

Typical mechanism:
- gradient ascent / gradient difference / preference-style losses / negative-response finetuning / iterative PEFT

Likely strengths:
- simple to implement
- strong forget metrics on benchmark data

Likely weaknesses:
- brittle utility-retention tradeoff
- vulnerability to relearning / paraphrase recovery
- often weak localization: edits are broad, not targeted

## 2. Representation / activation-space intervention
Representative papers:
- RMU-style representation misdirection
- latent steering analyses
- LUNAR
- reasoning-trace unlearning variants

Typical mechanism:
- redirect, corrupt, or suppress activation pathways associated with forget content

Why this cluster is interesting for us:
- closest conceptual bridge to our repo's selective/adaptive intervention philosophy
- offers a path toward token-, layer-, or turn-local unlearning instead of global parameter drift

## 3. Robustness-enhanced unlearning
Representative papers:
- SAM-based resilient unlearning (ICML 2025)
- UAM
- papers explicitly studying relearning / jailbreak robustness

Typical mechanism:
- min-max or smoothness-aware objectives
- retain-aware regularization
- optimization against future recovery attacks

## 4. Evaluation / benchmark papers
Representative papers:
- TOFU
- LUME
- SemEval-2025 Task 4
- probabilistic evaluation paper (EMNLP 2025)

Why this matters:
- the benchmark often determines what looks like progress
- many current evaluations still appear weak on mixed-query prompts, multi-turn recovery, multilingual transfer, and reasoning-trace leakage

## 5. Likely gap to exploit for EMNLP
A promising paper should probably combine:
1. **localized intervention** (activation/token/turn selective),
2. **robustness** (relearning/jailbreak/paraphrase resistant), and
3. **stronger NLP evaluation** (multi-turn + mixed retain/forget + maybe multilingual or reasoning-trace settings).
