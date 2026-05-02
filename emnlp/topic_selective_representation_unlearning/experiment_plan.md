# Experiment Plan — Selective Representation Unlearning for Multi-turn Recovery

## 1. Paper type
**Hybrid, evaluation-first**

Meaning:
- the main claim is about stronger and more realistic unlearning evaluation,
- the method contribution is a selective representation unlearning approach that is specifically designed to survive that harder evaluation.

---

## 2. Core baselines

| Baseline | Why include |
|---|---|
| NPO | widely used optimization baseline |
| OBLIVIATE | practical strong 2025 baseline |
| Reasoning Model Unlearning / R²MU | reasoning-aware baseline |
| Adaptive Localization | closest top-tier localization baseline |
| Not Every Token Needs Forgetting | selective token-level baseline |
| UIPE | correlated-knowledge-aware baseline |
| robust relearning baseline (SAM-style / ILU-style) | future fine-tuning robustness anchor |
| UAM-style retain-aware variant | optimization principle baseline |

---

## 3. Datasets / settings

| Setting | Role |
|---|---|
| TOFU | controlled forget/retain benchmark |
| LUME | cross-task generalization |
| WMDP | safety/harmful knowledge stress test |
| multilingual misinformation set (Learn and Unlearn style) | multilingual recovery |
| reasoning set (R-TOFU / reasoning-model-unlearning style) | reasoning leakage |
| custom multi-turn dialogue templates | main contribution setting |

### Recommendation
For the first paper version, do **one primary setting + one secondary setting**:
- Primary: TOFU or LUME with multi-turn recovery wrappers
- Secondary: reasoning or multilingual, not both at full scale initially

---

## 4. Evaluation axes

| Axis | Question |
|---|---|
| Direct forgetting | does the model suppress target answers on direct prompts? |
| Utility retention | does normal capability survive? |
| Multi-turn recovery | can the target be reconstructed over follow-up turns? |
| Paraphrase recovery | does indirect phrasing recover the target? |
| Correlated knowledge recovery | do nearby facts rebuild the target? |
| Relearning robustness | does downstream tuning restore the target? |
| Optional reasoning leakage | do intermediate reasoning traces still reveal the target? |
| Optional multilingual recovery | does the target survive in another language? |

---

## 5. Signature metric idea
We should not rely on a single forget score.
A better headline metric is something like:

## **Recovery-Aware Forgetting Score (RAFS)**

A combined score over:
- direct forgetting,
- multi-turn recovery failure rate,
- utility retention.

For example:

`RAFS = ForgetSuccess * (1 - MultiTurnRecoveryRate) * UtilityRetention`

This is only a proposal, but a metric like this makes the paper easier to message.

---

## 6. Key ablations
1. global update vs selective update
2. token-only vs layer-only vs token×layer masking
3. mean prototype vs contrastive prototype
4. with vs without multi-turn consistency loss
5. with vs without correlated-knowledge augmentation
6. LoRA-only vs full-parameter update (if affordable)

---

## 7. Success criteria
A convincing result would show:
1. similar or better direct forgetting than strong baselines,
2. better utility retention than global forgetting,
3. clearly lower multi-turn recovery rate,
4. ideally better post-finetuning robustness as well.

---

## 8. Fast execution plan
### Phase A — build evaluation harness
- create multi-turn dialogue wrappers around TOFU/LUME
- implement recovery prompt families: direct, paraphrase, hint, indirect reference, correlated cue

### Phase B — build method MVP
- extract forget/retain prototypes
- implement token×layer gate
- train LoRA-based selective unlearning

### Phase C — baseline comparison
- compare against NPO, OBLIVIATE, selective token baseline, localization baseline

### Phase D — secondary-axis validation
- reasoning or multilingual recovery check on smaller subset
