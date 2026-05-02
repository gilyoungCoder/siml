# Method Design — Selective Representation Unlearning for Multi-turn Recovery

## 1. Problem setting
We are given:
- a pretrained or instruction-tuned LLM,
- a **forget set** containing target facts/documents/examples,
- a **retain set** containing general utility data,
- and a stronger evaluation setting where the target may reappear through later conversational turns rather than in the first answer.

Our goal is to make the model:
1. forget the target content,
2. preserve general utility,
3. remain robust when the user continues the conversation with paraphrases, hints, or stepwise recovery prompts.

---

## 2. Core hypothesis
Existing unlearning updates are often too global. They suppress the first direct answer, but leave enough residual knowledge in latent representations that the target can be reconstructed in later turns.

Our hypothesis is:
> if we can detect **which token/layer representations are actually carrying target knowledge during generation**, then applying forgetting pressure only there will produce a better forget/retain tradeoff and stronger resistance to multi-turn recovery.

---

## 3. Method summary
We propose a method tentatively called:

## **SRU-MR: Selective Representation Unlearning for Multi-turn Recovery**

The method has three components:

### A. Forget-relevance scorer
For each layer `l` and token position `t`, estimate whether hidden state `h_(l,t)` is currently involved in target-knowledge activation.

Example score:

`r_(l,t) = sigmoid( alpha * ( sim( P_l h_(l,t), c_forget ) - tau ) )`

where:
- `P_l` is a learned or fixed projection,
- `c_forget` is a target representation prototype,
- `tau` is a threshold,
- `r_(l,t)` is a soft gate in `[0,1]`.

Intuition:
- high `r_(l,t)` means “this representation is likely part of a forget-target computation,”
- low `r_(l,t)` means “leave it mostly alone.”

### B. Selective unlearning update
Instead of applying a uniform forget loss everywhere, use `r_(l,t)` as a mask.

Possible update styles:
1. **Masked loss weighting**
   - give stronger forgetting loss where `r_(l,t)` is high.
2. **Masked representation shift**
   - push high-score representations away from the forget prototype and toward a neutral/retain region.
3. **Low-rank masked adapters**
   - update only LoRA/adapter parameters, but only through masked token/layer signals.

A simple formulation is to combine:
- **forget loss** on target behavior,
- **retain loss** on general utility,
- **sparsity/localization loss** so the mask stays selective,
- **multi-turn consistency loss** so the forgetting survives later recovery turns.

### C. Multi-turn recovery training signal
This is what distinguishes the method from ordinary single-turn selective unlearning.

Construct short dialogue chains like:
- Turn 1: direct question about target
- Turn 2: paraphrase or partial cue
- Turn 3: hint or stepwise reconstruction prompt
- Turn 4: indirect reference / related entity

We want the model to remain safe/forgotten across the whole chain.

So we add a loss such as:

`L_total = lambda_f L_forget + lambda_r L_retain + lambda_s L_sparse + lambda_m L_multi_turn`

where:
- `L_forget`: suppress target recall on forget prompts,
- `L_retain`: preserve normal answers on retain prompts,
- `L_sparse`: encourage the intervention mask to stay small/selective,
- `L_multi_turn`: penalize target recovery later in the dialogue.

---

## 4. How to get the forget prototype `c_forget`
Several options:

### Option 1 — mean hidden representation
Average hidden representations over forget examples at selected layers.
- simple,
- easy to implement,
- good starting point.

### Option 2 — contrastive prototype
Build prototypes from forget vs retain examples.
- stronger separation,
- better for reducing false positives.

### Option 3 — correlated-knowledge augmented prototype
Include nearby/related target knowledge as in UIPE-style thinking.
- stronger robustness,
- but more risk of over-forgetting.

**Recommendation:** start with Option 2.

---

## 5. Where to intervene
Three candidate granularities:

### Token-selective
Intervene only on specific token positions.
- easiest to explain,
- aligns with “Not Every Token Needs Forgetting.”

### Layer-selective
Intervene only on a subset of layers.
- cheaper,
- easier to ablate.

### Token × layer selective
Most expressive and probably strongest, but hardest to stabilize.

**Recommendation:** start with token × layer soft masking, but ablate token-only and layer-only versions.

---

## 6. Why this could beat global unlearning
Global unlearning can fail in two opposite ways:
- too weak: target survives indirectly,
- too strong: utility collapses.

Selective representation unlearning may help because it:
- concentrates the forgetting update where the target is active,
- leaves unrelated representations more intact,
- better matches the actual structure of recovery in dialogue.

---

## 7. What is genuinely new here?
Not just “selective unlearning” by itself.
The real novelty comes from the combination:
1. **representation-selective masking**,
2. **multi-turn recovery-aware training/evaluation**,
3. **forget/retain tradeoff measured under richer language interaction**.

That combination is much stronger than only saying “we localize updates.”

---

## 8. Main risks
1. The forget-relevance score may be noisy.
2. Multi-turn training could overfit to our recovery templates.
3. Correlated knowledge may survive outside the masked region.
4. If the method is too complex, the paper may look more ICML than EMNLP.

---

## 9. Minimal viable version
If we want a practical MVP:
- LoRA-only adaptation,
- contrastive forget prototype,
- token × layer soft mask,
- multi-turn recovery evaluation,
- no full-blown reasoning trace training at first.

That is enough for an initial paper-quality prototype.
