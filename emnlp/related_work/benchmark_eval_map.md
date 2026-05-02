# Benchmark / Evaluation Map (initial draft)

## Existing evaluation artifacts to inspect first

| Artifact | Type | Why it matters |
|---|---|---|
| TOFU | benchmark | canonical fictitious unlearning setup; many baselines report on it |
| LUME | benchmark | multi-task benchmark, likely better for stress-testing generalization |
| MUSE | benchmark / eval suite | commonly used in recent robustness papers |
| WMDP | benchmark | used in safety-oriented forgetting / harmful knowledge removal |
| SemEval-2025 Task 4 | shared task | shows what the community actually optimizes when forced into a common leaderboard |
| EMNLP 2025 probabilistic evaluation paper | evaluation framework | likely exposes metric/pathology issues in current LLM unlearning evaluation |

## Evaluation axes we should explicitly track
- Forget quality
- Retain / utility quality
- Mixed-query behavior (forget + retain in same prompt)
- Paraphrase robustness
- Relearning resistance
- Jailbreak resistance
- Multi-turn dialogue recovery
- Reasoning-trace leakage
- Multilingual transfer
- Calibration / uncertainty under forget prompts

## Current suspicion
The field is still too focused on:
- single-turn prompts
- benchmark-specific forget sets
- answer-level forgetting
- weak robustness testing

## EMNLP-friendly evaluation upgrade opportunity
If we build a new paper, the evaluation should likely include:
1. single-turn benchmark metrics,
2. multi-turn extraction conversations,
3. paraphrase/composition attacks,
4. mixed retain+forget prompts,
5. downstream utility tasks,
6. optionally multilingual or cross-domain transfer.
