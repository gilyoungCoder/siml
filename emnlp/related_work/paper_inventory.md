# LLM Unlearning Paper Inventory (initial sweep)

> Goal: collect the papers we should definitely read first before locking an EMNLP paper topic.
> Preference order: primary-source links only (ACL Anthology / arXiv / OpenReview / official benchmark sites).

## 1) Benchmark / evaluation papers

### 1. TOFU: A Task of Fictitious Unlearning for LLMs (arXiv 2024)
- Link: https://arxiv.org/abs/2401.06121
- Why it matters:
  - One of the canonical early LLM-unlearning benchmarks.
  - Clean synthetic setup: 200 fictitious author profiles, selective forget set, and a suite of metrics.
  - Strong baseline result: existing methods still do **not** behave like retraining without forget data.
- Use for us:
  - good for controlled ablations,
  - weak on realism / multilinguality / multi-turn interaction.

### 2. MUSE: Machine Unlearning Six-Way Evaluation for Language Models (arXiv 2024; later ICLR 2025)
- Link: https://arxiv.org/abs/2407.06460
- Official site: https://muse-bench.github.io/
- Why it matters:
  - Explicitly broadens evaluation to six properties: memorization, privacy leakage, utility, scalability, sequential requests, etc.
  - Evaluates 7B LMs on Harry Potter books and news data.
  - Key message: most methods fail to satisfy both data-owner and deployer expectations.
- Use for us:
  - likely the most important benchmark paper to structure an EMNLP evaluation section.

### 3. The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning (arXiv 2024)
- Link: https://arxiv.org/abs/2403.03218
- Benchmark site/code: https://wmdp.ai /
- Why it matters:
  - Makes hazardous-knowledge removal central.
  - Also introduces RMU, so it is both benchmark and method paper.
  - Important because it shifts unlearning from privacy/copyright to safety-risk removal.
- Use for us:
  - if we choose hazardous-knowledge unlearning, this is non-optional.

### 4. LUME: LLM Unlearning with Multitask Evaluations (Findings of EMNLP 2025)
- Link: https://aclanthology.org/2025.findings-emnlp.347/
- Why it matters:
  - Multi-task benchmark covering short novels, synthetic biographies with sensitive information, and public biographies.
  - Provides both 1B and 7B target models.
  - Useful because it moves beyond a single benchmark family.
- Use for us:
  - strong candidate for “cross-task generalization” analysis.

### 5. SemEval-2025 Task 4: Unlearning Sensitive Content from Large Language Models (SemEval 2025)
- Link: https://aclanthology.org/2025.semeval-1.329/
- Why it matters:
  - Community benchmark / competition setup.
  - Includes long-form creative documents, PII-bearing biographies, and real training-data documents.
  - Valuable for seeing what practitioners optimize when benchmark pressure is real.
- Use for us:
  - especially helpful if we want an EMNLP paper that feels benchmark-grounded and reproducible.

### 6. Existing Adversarial LLM Unlearning Evaluations Are Inconclusive (OpenReview, submitted 2025)
- Link: https://openreview.net/forum?id=tLY219JUaK
- Why it matters:
  - Very important meta-evaluation critique.
  - Claims current adversarial evaluations can inject new information during testing, effectively **reteaching** the model.
  - Proposes principles like minimal information injection and downstream-task awareness.
- Use for us:
  - probably central if we pursue a new evaluation paper.

### 7. Learn and Unlearn: Addressing Misinformation in Multilingual LLMs (EMNLP 2025)
- Link: https://aclanthology.org/2025.emnlp-main.516/
- Why it matters:
  - Direct evidence that fake information propagates across languages.
  - English-only unlearning is insufficient and can even reinforce misinformation cross-lingually.
- Use for us:
  - strongest anchor if we want a multilingual EMNLP angle.

### 8. Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills (EMNLP 2025)
- Link: https://aclanthology.org/2025.emnlp-main.220/
- Why it matters:
  - Makes the key point that removing the final answer is not enough; reasoning traces themselves can leak sensitive content.
  - Evaluates on WMDP, StrongReject, JBB-Behaviors, and WildJailbreak.
- Use for us:
  - strongest anchor if we want a reasoning / CoT / trajectory-aware EMNLP paper.

## 2) Method papers we should read early

### 9. Offset Unlearning for Large Language Models (arXiv 2024; TMLR track)
- Link: https://arxiv.org/abs/2404.11045
- Why it matters:
  - Black-box-friendly framing via logit offsets.
  - Useful contrast against weight-editing and full finetuning methods.
- Use for us:
  - important if we want practical deployment constraints or API-only settings.

### 10. RMU from WMDP / representation-control line (arXiv 2024)
- Main source: https://arxiv.org/abs/2403.03218
- Why it matters:
  - Representation-space control rather than pure token-level gradient forgetting.
  - Influential for hazardous-knowledge unlearning.
- Use for us:
  - closest ancestor for any “localized/representation-aware” idea.

### 11. Reveal and Release: Iterative LLM Unlearning with Self-generated Data (Findings of EMNLP 2025)
- Link: https://aclanthology.org/2025.findings-emnlp.1298/
- Why it matters:
  - Challenges the unrealistic assumption that we always have the true forget set.
  - Uses model self-revelation plus iterative parameter-efficient unlearning.
- Use for us:
  - highly relevant if we want weak-supervision / realistic deletion-request settings.

### 12. UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets (Findings of EMNLP 2025)
- Link: https://aclanthology.org/2025.findings-emnlp.1374/
- Why it matters:
  - Explicitly argues that logically related knowledge lets the model reconstruct the target.
  - Very relevant to “correlated knowledge pathways” and selective unlearning.
- Use for us:
  - one of the best bridges to our repo’s high-level “when/where” philosophy.

### 13. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models (EMNLP 2025)
- Link: https://aclanthology.org/2025.emnlp-main.183/
- Why it matters:
  - Practical LoRA-based framework with masking, distillation, and world-fact components.
  - Uses document-level memorization score + utility + fluency.
- Use for us:
  - useful strong baseline if we test a new method.

### 14. REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space (Findings of ACL 2025)
- Link: https://aclanthology.org/2025.findings-acl.763/
- Why it matters:
  - Non-gradient method focused on truly sensitive PII-like memorization.
  - Claims robustness to extraction attacks.
- Use for us:
  - relevant if we want a privacy/memorization rather than hazardous-knowledge paper.

### 15. Model Unlearning via Sparse Autoencoder Subspace Guided Projections / SSPU (EMNLP 2025)
- Link: https://aclanthology.org/2025.emnlp-main.1348/
- Why it matters:
  - Interpretable subspace-guided unlearning with adversarial robustness emphasis.
  - Directly ties SAE features to targeted subspace control.
- Use for us:
  - valuable for any representation-localization idea.

### 16. Not Every Token Needs Forgetting: Selective Unlearning Balancing Forgetting and Utility in Large Language Models (Findings of EMNLP 2025)
- Link: https://aclanthology.org/2025.findings-emnlp.96/
- Why it matters:
  - Very aligned with the intuition that blanket document-level forgetting is too coarse.
  - Selective forgetting is likely a fertile EMNLP topic.
- Use for us:
  - another strong bridge to our “adaptive/localized intervention” philosophy.

### 17. Unilogit: Robust Machine Unlearning for LLMs Using Uniform-Target Self-Distillation (Findings of ACL 2025)
- Link: https://aclanthology.org/2025.findings-acl.1154/
- Why it matters:
  - Robustness-oriented distillation view of unlearning.
- Use for us:
  - likely good baseline in robustness-heavy comparisons.

### 18. UNLEARN: Efficient Removal of Knowledge in Large Language Models (Findings of NAACL 2025)
- Link: https://aclanthology.org/2025.findings-naacl.405/
- Why it matters:
  - Efficiency-focused method paper.
- Use for us:
  - useful baseline for speed/efficiency claims.

## 3) Robustness / relearning angle

### 19. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond (arXiv 2025)
- Link: https://arxiv.org/abs/2502.05374
- Why it matters:
  - Explicitly studies relearning attacks after unlearning.
  - Connects robust unlearning to SAM / smoothness optimization.
  - Evaluates on WMDP and MUSE.
- Use for us:
  - likely the closest LLM-side paper to the robustness intuition the user called “UAM-like”.

### 20. Unlearning-Aware Minimization (OpenReview 2025)
- Link: https://openreview.net/forum?id=kAuckbcMvi
- Why it matters:
  - General machine-unlearning paper, **not specifically an NLP paper**, but it includes LLM hazardous-knowledge experiments in the broader machine-unlearning framing.
  - Good conceptual example for min-max / saliency-aware unlearning objectives.
- Use for us:
  - treat as inspiration, not as a direct LLM-unlearning benchmark anchor.

## 4) Current takeaways from the initial paper map

1. **Evaluation is probably the biggest gap.**
   - TOFU is clean but synthetic.
   - MUSE is broader but still mostly single-turn/static.
   - Adversarial evaluation itself may be contaminated by information injection.
   - This smells like a real EMNLP opportunity.

2. **Multilingual and reasoning settings are now clearly active frontiers.**
   - EMNLP 2025 already has strong signals here.
   - If we want novelty, we should avoid just “one more TOFU/WMDP method”.

3. **Selective / localized forgetting is emerging but not settled.**
   - UIPE, selective-token unlearning, SSPU, and REVS all suggest coarse global forgetting is too blunt.
   - This is where our repo’s high-level philosophy may transfer best.

4. **Robustness to relearning / jailbreak / multi-turn recovery is underdeveloped.**
   - Strong candidate for a paper that is both methodologically interesting and evaluation-heavy.

