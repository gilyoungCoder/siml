# Next Steps Package — EMNLP LLM Unlearning

## 1. Final recommendation
### Chosen framing
**Evaluation-faithful selective unlearning under multi-turn / reasoning / multilingual recovery**

### Why this is the best choice now
- It aligns with the strongest 2025-2026 papers.
- It matches EMNLP's preference for language-centric evaluation and analysis.
- It gives us a cleaner novelty story than a purely optimization-focused method paper.
- It still leaves room for a light/selective method component if needed.

## 2. Problem statement
Current LLM unlearning methods often report strong forgetting under simplified benchmarks, but it remains unclear whether this forgetting persists under richer language settings such as multi-turn dialogue, reasoning-trace exposure, multilingual transfer, and correlated-knowledge recovery. At the same time, broad forgetting updates can unnecessarily harm general utility. We therefore seek an evaluation-faithful unlearning framework that tests whether selective forgetting truly holds under realistic recovery settings while preserving general model capability.

## 3. Paper type decision
### Recommended: Hybrid (evaluation-first with a selective method component)
Why:
- pure evaluation paper = safest and still publishable,
- pure method paper = higher risk because 2025 baselines are already strong,
- hybrid = strongest upside if we keep the method lightweight and the evaluation very convincing.

## 4. Concrete novelty claim
We will argue that existing LLM unlearning work still overestimates success because it evaluates forgetting too narrowly and often uses globally destructive updates. Our contribution is to combine:
1. a richer recovery-oriented evaluation protocol,
2. a selective forgetting view motivated by localization and correlated-knowledge leakage,
3. reasoning/multilingual/multi-turn stress tests.

## 5. Title candidates
1. **Selective Unlearning Beyond One-Shot QA: Evaluating LLM Forgetting under Multi-turn, Reasoning, and Multilingual Recovery**
2. **Did the Model Really Forget? Evaluation-Faithful Selective Unlearning for Large Language Models**
3. **Forgetting Under Recovery: A Language-Centric Evaluation Framework for Selective LLM Unlearning**
4. **Reasoning-, Dialogue-, and Multilingual-Aware Evaluation for Selective LLM Unlearning**
5. **Beyond Answer-Level Forgetting: Evaluating Selective LLM Unlearning under Reasoning and Cross-Lingual Recovery**

## 6. Baseline table
| Category | Candidate baselines | Why include |
|---|---|---|
| Optimization | NPO, UAM-inspired retain-aware objective, OBLIVIATE | strong 2025 optimization/practical baselines |
| Robustness | SAM-based relearning method, ILU/invariance method | downstream recovery robustness |
| Localization | Adaptive Localization, Not Every Token Needs Forgetting, UIPE | selective/localized forgetting reference points |
| Evaluation-only references | Fully Probabilistic Perspective, Do LLMs Really Forget? | evaluation design anchors |

## 7. Dataset table
| Dataset / setting | Purpose |
|---|---|
| TOFU | controlled forget/retain benchmark |
| MUSE | broader evaluation coverage |
| LUME | multi-task unlearning setting |
| WMDP | hazardous knowledge / safety-oriented forgetting |
| multilingual misinformation setup (Learn and Unlearn style) | cross-lingual recovery |
| reasoning setup (R-TOFU / Reasoning Model Unlearning style) | reasoning-trace leakage |
| custom multi-turn recovery prompts | dialogue accumulation / recovery |

## 8. Evaluation recipe table
| Evaluation axis | What to measure |
|---|---|
| Direct forgetting | target answer suppression / probability reduction |
| Utility retention | downstream QA / instruction-following / general task performance |
| Multi-turn recovery | can target re-emerge across a dialogue? |
| Reasoning leakage | can the model reveal the target through intermediate reasoning? |
| Multilingual recovery | does forgotten content reappear in another language? |
| Correlated knowledge recovery | can nearby facts reconstruct the target? |
| Relearning robustness | does downstream fine-tuning bring the target back? |

## 9. 8-week execution sketch
### Week 1-2
- lock benchmarks and recovery protocol
- reproduce 2-3 strongest baselines

### Week 3-4
- build multi-turn + reasoning + multilingual evaluation harness
- validate that current baselines really fail in at least one of these settings

### Week 5-6
- add lightweight selective method component if needed
- run ablations on utility vs forgetting vs recovery robustness

### Week 7
- finalize tables, error analysis, case studies

### Week 8
- write paper, sharpen framing, prepare appendix

## 10. Immediate next commands for us intellectually
1. freeze the preferred title direction,
2. choose the exact benchmark mix,
3. decide whether the method contribution is token-selective, representation-selective, or evaluation-only,
4. write abstract + intro skeleton.
