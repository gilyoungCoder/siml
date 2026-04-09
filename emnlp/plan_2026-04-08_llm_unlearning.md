# Plan — EMNLP-targeted LLM Unlearning (2026-04-08)

## 0. Immediate objective
Since the user is away / sleeping, the minimum useful deliverable is a **strong research plan** that can drive an overnight literature pass and quickly converge on a topic.

---

## 1. Working topic space
Instead of starting from one fixed method, I will search for the best topic among the following three clusters:

### Track A — Adaptive / selective unlearning for LLMs
**Why this matches us**
- High-level match to our repo philosophy: intervene only when needed, preserve utility, avoid blunt forgetting.
- Natural bridge from selective safety intervention to language-model unlearning.

**What to test in related work**
- Do current LLM methods over-forget because they act globally?
- Is there room for token-, span-, layer-, or representation-selective unlearning?
- Can training-free or inference-time steering meaningfully complement weight updates?

### Track B — Robust unlearning under reasoning, multi-turn, or downstream re-tuning
**Why this matters for EMNLP**
- EMNLP is a strong fit for evaluation-heavy papers when the task definition itself is changing.
- Recent work suggests answer-only forgetting misses hidden leakage in reasoning traces and multi-turn generation.

**What to test in related work**
- Reasoning-trace leakage after answer-level forgetting
- Relearning after downstream fine-tuning
- Prompting/adversarial auditing instability
- Whether current benchmarks are too narrow or too synthetic

### Track C — Cross-lingual / multilingual unlearning
**Why this matters for EMNLP**
- Strong NLP angle, less saturated than generic safety unlearning
- Recent evidence suggests harmful or false information can propagate across languages even if only one language was trained/unlearned

**What to test in related work**
- Whether English-centric unlearning leaves residual knowledge in other languages
- Whether multilingual forget sets can be approximated / generated rather than fully collected

---

## 2. Current leading hypothesis
My current best bet is:

### Leading paper direction
**Adaptive representation-space unlearning for reasoning- and multilingual-aware LLM forgetting**

This is still broad, so the literature pass needs to decide which of these narrower variants is best:

1. **Reasoning-trace-aware selective unlearning**
   - Forget sensitive knowledge in intermediate reasoning traces, not just final answers.
   - Likely benchmark targets: R-TOFU, reasoning-model unlearning settings, WMDP-style safety tasks.

2. **Cross-lingual selective unlearning**
   - Forget a target fact / misinformation / private datum across languages while preserving unrelated multilingual ability.
   - Likely benchmark targets: multilingual misinformation / factual editing + multilingual evaluation.

3. **Data-limited adaptive unlearning**
   - Forget when explicit forget data is missing, incomplete, or privacy-restricted.
   - Likely benchmark targets: TOFU / MUSE / self-generated-data settings.

---

## 3. Research plan for the related-work pass

### Phase 1 — Build the paper map
Group papers into these buckets:
1. **Classical / early LLM unlearning setups**
2. **Loss-based / optimization-based methods**
3. **Representation-steering / activation-space methods**
4. **Data-free or self-generated forget-data methods**
5. **Evaluation / auditing / robustness papers**
6. **Reasoning / multilingual / downstream-finetuning extensions**

### Phase 2 — Decide the topic using three filters
For each candidate topic, score:
- **NLP fit**: does it feel like EMNLP rather than generic ML?
- **Gap clarity**: is there an obvious failure in the literature?
- **Feasibility**: can we build and evaluate it without unrealistic infrastructure?

### Phase 3 — Lock the first-paper angle
Pick the topic with the strongest combination of:
- strong evaluation story,
- clear benchmark weakness,
- method that is novel but not too risky.

---

## 4. Starter paper shortlist to read first

### Core task / benchmark papers
- **Who's Harry Potter? Approximate Unlearning in LLMs** (arXiv 2023)
- **TOFU: A Task of Fictitious Unlearning for LLMs** (arXiv 2024)
- **MUSE: Machine Unlearning Six-Way Evaluation for Language Models** (arXiv 2024)
- **LUME: LLM Unlearning with Multitask Evaluations** (arXiv 2025)
- **R-TOFU: Unlearning in Large Reasoning Models** (EMNLP 2025)
- **SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models** (SemEval 2025)

### Core method papers
- **Negative Preference Optimization (NPO)** (arXiv 2024)
- **Representation Misdirection for Unlearning (RMU)** / follow-up analyses
- **LUNAR: LLM Unlearning via Neural Activation Redirection** (arXiv 2025)
- **Reveal and Release: Iterative LLM Unlearning with Self-generated Data** (Findings of EMNLP 2025)
- **Unlearning-Aware Minimization (UAM)** (NeurIPS 2025)
- **Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning** (arXiv 2025)

### Audit / robustness papers
- **Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models** (Findings of EMNLP 2025)
- **Identifying Unlearned Data in LLMs via Membership Inference Attacks** (EMNLP 2025)
- **Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills** (EMNLP 2025)
- **Learn and Unlearn: Addressing Misinformation in Multilingual LLMs** (EMNLP 2025)

---

## 5. What I think is most promising right now
If I had to place an early bet before the full read:

### Best-fit EMNLP topic candidate
**Selective unlearning beyond answer-level forgetting**

More concrete version:
- current benchmarks mostly focus on answer retrieval,
- but real LLMs leak through **reasoning traces**, **cross-lingual transfer**, **multi-turn dialogue**, and **post-unlearning fine-tuning**,
- so an EMNLP paper could win by combining:
  1. a sharper evaluation protocol, and
  2. a lightweight selective method that targets only the leakage-carrying subspace / tokens / layers.

This feels more EMNLP-native than a purely optimization-heavy method paper.

---

## 6. Overnight execution plan
1. Read benchmark/evaluation papers first (TOFU, MUSE, LUME, R-TOFU, SemEval-2025 Task 4).
2. Read 4–6 method papers second (NPO, RMU-related, LUNAR, UAM, Reveal-and-Release, ILU).
3. Extract repeated failure modes.
4. Narrow to 2 final paper ideas.
5. Start writing detailed related-work notes in `emnlp/related_work/`.

---

## 7. Current blocker / operational note
True `omx team` execution is currently blocked by pre-existing uncommitted repo changes, so I am using a research-first fallback while keeping the workspace safe.
