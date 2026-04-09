# Method Formalization — SRU-MR

## 1. Setup
Let:
- `M_theta` be the base LLM with parameters `theta`
- `D_f` be the forget set
- `D_r` be the retain set
- `C(x)` be a multi-turn conversational wrapper built from a forget example `x`

For an input sequence, denote the hidden representation at layer `l` and token position `t` as:

`h_(l,t) in R^d`

Our goal is to learn updated parameters `theta'` such that:
1. direct target recall is reduced on `D_f`,
2. utility is preserved on `D_r`,
3. target recovery is reduced across follow-up dialogue turns.

---

## 2. Forget prototype construction
We define a forget prototype `c_f^(l)` for each selected layer `l`.

### Contrastive prototype (recommended)
Using hidden states from forget and retain examples:

`c_f^(l) = mean_{x in D_f} pool(h_x^(l)) - beta * mean_{x in D_r} pool(h_x^(l))`

where:
- `pool(.)` can be mean pooling over selected target spans or answer tokens,
- `beta` controls retain contrast.

Interpretation:
- the prototype captures directions that are more associated with forget content than retain content.

---

## 3. Forget-relevance scoring
For each selected layer-token pair `(l,t)`, define a forget-relevance score:

`r_(l,t) = sigmoid( alpha * ( cos(P_l h_(l,t), c_f^(l)) - tau_l ) )`

where:
- `P_l` is a projection matrix or identity map,
- `cos(.,.)` is cosine similarity,
- `tau_l` is a learned or tuned threshold,
- `alpha` sharpens the gate.

Properties:
- if `r_(l,t)` is high, the hidden state is treated as strongly target-relevant,
- if `r_(l,t)` is low, the hidden state is treated as mostly unrelated.

---

## 4. Selective forgetting objective
We combine four losses.

### 4.1 Forget loss
Suppress target recall on forget examples:

`L_forget = E_{x in D_f} [ ell_forget(M_theta'(x), y_f) ]`

Possible choices:
- negative preference optimization loss,
- target answer suppression loss,
- KL to a neutral/refusal target distribution,
- representation misdirection loss.

### 4.2 Retain loss
Preserve normal capability on retain data:

`L_retain = E_{x in D_r} [ ell_retain(M_theta'(x), y_r) ]`

Typical choices:
- next-token cross-entropy on retain set,
- KL to base model outputs on retain prompts,
- instruction-following task loss.

### 4.3 Sparsity / selectivity loss
Encourage the intervention to remain localized:

`L_sparse = E_x [ sum_{l,t} r_(l,t) ]`

Optionally normalized by sequence length or number of selected layers.

Interpretation:
- the model is rewarded for using as small an intervention region as possible.

### 4.4 Multi-turn recovery loss
Construct multi-turn dialogues `C(x)` from forget targets and penalize later recovery:

`L_multi = E_{x in D_f} [ sum_{k=1}^K ell_recovery(M_theta'(C_k(x)), y_k^safe) ]`

where:
- `C_k(x)` is the dialogue prefix up to turn `k`,
- `y_k^safe` encodes the desired non-recovery behavior at turn `k`.

Interpretation:
- the model should not merely fail at turn 1;
- it should continue to resist reconstruction across turns 2..K.

---

## 5. Full objective
We optimize:

`L_total = lambda_f L_forget + lambda_r L_retain + lambda_s L_sparse + lambda_m L_multi`

where:
- `lambda_f` controls forgetting strength,
- `lambda_r` protects utility,
- `lambda_s` enforces localization,
- `lambda_m` emphasizes dialogue robustness.

---

## 6. Masked update mechanism
There are multiple implementation options.

### Option A — masked loss weighting
Weight token contributions by `r_(l,t)` in the backward pass.

### Option B — masked residual steering
Apply a representation shift:

`h'_(l,t) = h_(l,t) - gamma * r_(l,t) * u_f^(l)`

where `u_f^(l)` is a learned forget direction.

### Option C — masked LoRA adaptation (recommended MVP)
Use LoRA adapters in selected layers and modulate their effective update by the forget-relevance mask.

Reason:
- practical,
- parameter-efficient,
- easy to compare against 2025 baselines.

---

## 7. Multi-turn dialogue generation protocol
For each forget item, generate a recovery chain:

1. **Direct query**
2. **Paraphrase query**
3. **Hint-based query**
4. **Indirect / correlated cue**
5. optional **stepwise reasoning prompt**

This gives a structured view of recovery difficulty rather than a single scalar forget score.

---

## 8. Suggested ablations
1. No selectivity mask (`r_(l,t)=1` everywhere)
2. Token-only mask
3. Layer-only mask
4. Token × layer mask
5. Mean prototype vs contrastive prototype
6. With vs without `L_multi`
7. With vs without `L_sparse`
8. With vs without correlated-knowledge augmentation

---

## 9. Main empirical claim to test
The primary claim is:
> selective representation unlearning reduces multi-turn recovery more effectively than broad global unlearning while preserving more retain utility.

A strong secondary claim would be:
> the benefits remain visible under at least one richer extension such as reasoning leakage or multilingual recovery.
