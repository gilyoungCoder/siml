# Seed Paper Shortlist for EMNLP LLM Unlearning

## Core benchmarks / evaluation

1. **TOFU: A Task of Fictitious Unlearning for LLMs** (arXiv 2024)
   - Link: https://arxiv.org/abs/2401.06121
   - Why it matters: foundational fictitious benchmark; synthetic author-profile setup with a holistic metric suite.

2. **MUSE: Machine Unlearning Six-Way Evaluation for Language Models** (arXiv 2024)
   - Link: https://arxiv.org/abs/2407.06460
   - Why it matters: six-property evaluation frame; highlights privacy leakage, scalability, and sustainability.

3. **LUME: LLM Unlearning with Multitask Evaluations** (Findings of EMNLP 2025)
   - Link: https://aclanthology.org/2025.findings-emnlp.347/
   - Why it matters: multitask benchmark spanning synthetic novels, synthetic biographies, and public biographies.

4. **SemEval-2025 Task 4: Unlearning Sensitive Content from Large Language Models** (SemEval 2025)
   - Link: https://aclanthology.org/2025.semeval-1.329/
   - Why it matters: community shared task; good signal for what the field currently treats as operational LLM unlearning.

5. **Position: LLM Unlearning Benchmarks are Weak Measures of Progress** (SaTML 2025 / arXiv)
   - Link: https://arxiv.org/abs/2410.02879
   - Why it matters: strong warning that current wins may be benchmark artifacts.

6. **Erasing Without Remembering: Implicit Knowledge Forgetting in Large Language Models** (arXiv 2025)
   - Link: https://arxiv.org/abs/2502.19982
   - Why it matters: introduces UGBench / PerMU and explicitly targets paraphrased + implicit knowledge retention.

7. **OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics** (arXiv 2025)
   - Link: https://arxiv.org/abs/2506.12618
   - Why it matters: unified comparison layer across TOFU, MUSE, and WMDP; useful for reproducing method comparisons.

## Core methods

8. **Who's Harry Potter? Approximate Unlearning in LLMs** (arXiv 2023)
   - Link: https://arxiv.org/abs/2310.02238
   - Why it matters: very early practical/copyright-oriented post-hoc unlearning paper.

9. **Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning** (COLM 2024 / arXiv)
   - Link: https://arxiv.org/abs/2404.05868
   - Why it matters: major preference-optimization baseline; still central in later comparisons.

10. **ULMR: Unlearning Large Language Models via Negative Response and Model Parameter Average** (EMNLP Industry 2024)
    - Link: https://aclanthology.org/2024.emnlp-industry.57/
    - Why it matters: practical data rewriting + parameter averaging baseline.

11. **Unlearning-Aware Minimization (UAM)** (NeurIPS 2025)
    - Link: https://openreview.net/forum?id=kAuckbcMvi
    - Why it matters: min-max unlearning framework; importantly, this is a **general machine unlearning** paper with LLM QA experiments (WMDP), not an NLP-only benchmark paper.

12. **Reveal and Release: Iterative LLM Unlearning with Self-generated Data** (Findings of EMNLP 2025)
    - Link: https://aclanthology.org/2025.findings-emnlp.1298/
    - Why it matters: self-generated-data unlearning, relevant if we want low-supervision target expansion.

13. **Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning** (ICML 2025 / arXiv)
    - Link: https://arxiv.org/abs/2506.01339
    - Why it matters: downstream recovery/relearning robustness is a major emerging evaluation axis.

14. **OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models** (arXiv 2025)
    - Link: https://arxiv.org/abs/2505.04416
    - Why it matters: practical robust unlearning recipe with broader evaluation mix.

## Reasoning / compositional / multilingual frontier

15. **R-TOFU: Unlearning in Large Reasoning Models** (arXiv 2025)
    - Link: https://arxiv.org/abs/2505.15214
    - Why it matters: benchmark for CoT-level / reasoning-trace leakage.

16. **Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills** (EMNLP 2025)
    - Link: https://aclanthology.org/2025.emnlp-main.220/
    - Why it matters: first principled LRM unlearning paper centered on trace leakage.

17. **SEPS: A Separability Measure for Robust Unlearning in LLMs** (EMNLP 2025)
    - Link: https://aclanthology.org/2025.emnlp-main.283/
    - Why it matters: mixed forget/retain prompts; very relevant to selective / compositional unlearning.

18. **Learn and Unlearn: Addressing Misinformation in Multilingual LLMs** (EMNLP 2025)
    - Link: https://aclanthology.org/2025.emnlp-main.516/
    - Why it matters: multilingual propagation problem; English-only unlearning is insufficient.

19. **Harry Potter is Still Here! Probing Knowledge Leakage in Targeted Unlearned Large Language Models** (Findings of EMNLP 2025)
    - Link: https://aclanthology.org/2025.findings-emnlp.778/
    - Why it matters: targeted adversarial probing / LURK evaluation for leakage after apparent forgetting.

## Tentative thesis after first-pass reading
The most underexploited EMNLP direction appears to be:
- **localized/compositional unlearning under mixed prompts or dialogue**,
with backups in
- **reasoning-trace-aware unlearning**, and
- **multilingual propagation-aware unlearning**.
