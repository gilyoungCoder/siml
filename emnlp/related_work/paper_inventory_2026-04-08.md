# LLM Unlearning Paper Inventory (initial overnight shortlist)

## A. Core methods

### 1. Large Language Model Unlearning
- **Source:** arXiv 2023
- **Link:** https://arxiv.org/abs/2310.10683
- **Why it matters:** among the earliest papers framing LLM unlearning directly; useful baseline family.
- **Use for us:** historical starting point for forget/retain optimization and evaluation assumptions.

### 2. Who's Harry Potter? Approximate Unlearning in LLMs
- **Source:** arXiv 2023
- **Link:** https://arxiv.org/abs/2310.02238
- **Why it matters:** canonical approximate unlearning setup for copyrighted content/books.
- **Use for us:** especially important for book/news style unlearning benchmarks and for later MUSE comparisons.

### 3. Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning
- **Source:** arXiv 2024
- **Link:** https://arxiv.org/abs/2404.05868
- **Why it matters:** one of the most reused optimization baselines in later LLM unlearning work.
- **Use for us:** baseline to beat or stress-test; likely mandatory in any related-work section.

### 4. Large Language Model Unlearning via Embedding-Corrupted Prompts
- **Source:** NeurIPS 2024 / arXiv
- **Link:** https://arxiv.org/abs/2406.07933
- **Why it matters:** lightweight, inference-time flavored alternative to full parameter updates.
- **Use for us:** closest existing line to our repo's selective / training-free instinct.

### 5. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond
- **Source:** ICML 2025 (PMLR)
- **Link:** https://proceedings.mlr.press/v267/fan25e.html
- **Why it matters:** pushes robustness against relearning attacks; connects unlearning to sharpness/smoothness.
- **Use for us:** strong motivation for robustness-aware objectives.

### 6. Unlearning-Aware Minimization (UAM)
- **Source:** NeurIPS 2025 / OpenReview
- **Link:** https://openreview.net/pdf/e3abccbf8072993459fd4ac2f73f9488157a78c0.pdf
- **Why it matters:** likely the UAM paper the user meant. Not LLM-only, but explicitly relevant to LLM safety unlearning and useful as an optimization principle.
- **Use for us:** candidate inspiration for a robust min-max unlearning objective.

## B. Benchmarks and evaluation

### 7. TOFU: A Task of Fictitious Unlearning for LLMs
- **Source:** arXiv 2024
- **Link:** https://arxiv.org/abs/2401.06121
- **Why it matters:** synthetic author-profile benchmark with a broad metric suite; still a standard reference point.
- **Gap signal:** baseline methods still fail to behave like retraining without the forget data.

### 8. MUSE: Machine Unlearning Six-Way Evaluation for Language Models
- **Source:** arXiv 2024 / ICML 2024 workshop visibility
- **Link:** https://arxiv.org/abs/2407.06460
- **Why it matters:** one of the strongest benchmark papers; stresses six properties including privacy leakage, scalability, and sustainability.
- **Gap signal:** methods often reduce memorization but still fail utility, privacy, or sequential-unlearning requirements.

### 9. LUME: LLM Unlearning with Multitask Evaluations
- **Source:** Findings of EMNLP 2025
- **Link:** https://aclanthology.org/2025.findings-emnlp.347/
- **Why it matters:** newer multitask benchmark spanning creative writing, synthetic biographies with PII, and public biographies.
- **Gap signal:** newer tasks still expose major forgetting-vs-utility weaknesses.

### 10. SemEval-2025 Task 4: Unlearning Sensitive Content from Large Language Models
- **Source:** SemEval 2025
- **Link:** https://aclanthology.org/2025.semeval-1.329/
- **Why it matters:** community benchmark with many system papers; useful to see what actually worked in a shared-task setting.
- **Gap signal:** many methods are still incremental variants of GA / KL / LoRA / model merging rather than principled selective forgetting.

### 11. FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledge
- **Source:** EMNLP 2025
- **Link:** https://aclanthology.org/2025.emnlp-main.657/
- **Why it matters:** shifts focus from raw forgetting to **faithful** forgetting and interconnected knowledge.
- **Gap signal:** current methods may produce superficial unlearning rather than faithful removal.

### 12. Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models
- **Source:** Findings of EMNLP 2025
- **Link:** https://aclanthology.org/2025.findings-emnlp.117/
- **Why it matters:** challenges common audit practice directly.
- **Gap signal:** some "audits" can elicit arbitrary content regardless of whether true forgetting occurred.

### 13. Dissecting Fine-Tuning Unlearning in Large Language Models
- **Source:** EMNLP 2024
- **Link:** https://aclanthology.org/2024.emnlp-main.228/
- **Why it matters:** uses activation/parameter analysis to argue many methods modify retrieval behavior rather than erase knowledge.
- **Gap signal:** mechanistic understanding remains weak.

## C. Opportunity papers for EMNLP-style topic framing

### 14. Learn and Unlearn: Addressing Misinformation in Multilingual LLMs
- **Source:** EMNLP 2025
- **Link:** https://aclanthology.org/2025.emnlp-main.516/
- **Why it matters:** directly shows harmful information propagates across languages and English-only unlearning is insufficient.
- **Opportunity:** strong seed for a multilingual EMNLP paper.

### 15. Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills
- **Source:** EMNLP 2025
- **Link:** https://aclanthology.org/2025.emnlp-main.220/
- **Why it matters:** introduces reasoning-trace leakage as a new unlearning target.
- **Opportunity:** strong seed for reasoning-aware selective unlearning.

### 16. Reveal and Release: Iterative LLM Unlearning with Self-generated Data
- **Source:** Findings of EMNLP 2025
- **Link:** https://aclanthology.org/2025.findings-emnlp.1298/
- **Why it matters:** relaxes the assumption that the forget set is fully accessible.
- **Opportunity:** self-generated forget data could combine well with selective/inference-time ideas.

### 17. A Fully Probabilistic Perspective on Large Language Model Unlearning: Evaluation and Optimization
- **Source:** EMNLP 2025
- **Link:** https://aclanthology.org/2025.emnlp-main.452/
- **Why it matters:** suggests the field is also moving toward better formalization of objective/metric design.
- **Opportunity:** useful for theory-grounded framing.

## D. Extra benchmark realism papers

### 18. LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks
- **Source:** arXiv 2025
- **Link:** https://arxiv.org/abs/2504.10185
- **Why it matters:** argues current benchmarks may be easier/more compressible than expected.
- **Use for us:** warning sign that an EMNLP paper should avoid benchmark overfitting.

### 19. OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics
- **Source:** arXiv 2025
- **Link:** https://arxiv.org/abs/2506.12618
- **Why it matters:** useful infrastructure/meta-benchmark reference across TOFU, MUSE, and WMDP.
- **Use for us:** can accelerate reproduction and baseline comparison.

## Current thesis after initial pass
1. The strongest *NLP-native* opportunities are **multilingual propagation**, **reasoning-trace leakage**, and **faithful/audit-robust evaluation**.
2. The strongest *method inspiration* from outside strict NLP is **UAM / SAM-style robustness-aware optimization**.
3. The most repo-aligned but still novel direction is a **selective, example-conditioned, possibly inference-time or activation-space unlearning method** for LLMs.
