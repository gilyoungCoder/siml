# Method Spec — Family-Grouped Exemplar-Probed Spatial Guidance for Safe Diffusion

> **For ml-writer**: This document describes exactly what the codebase does (file-level references in parentheses). All math/notation should be lifted from here without paraphrase. Code locations: `SafeGen/safegen/generate_family.py` (single concept) and `SafeGen/safegen/generate_family_multi.py` (multi-concept).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $z_t \in \mathbb{R}^{4 \times h \times w}$ | latent at diffusion step $t$ (SD1.4 $h=w=64$) |
| $c_p$ | text embedding of the user prompt |
| $\mathcal{F} = \{f_1, \dots, f_K\}$ | offline-built families for the target concept ($K=4$ in all our packs) |
| $E_f^{\text{tgt}}, E_f^{\text{anc}} \in \mathbb{R}^{1 \times 77 \times 768}$ | per-family target / anchor text embeddings |
| $\phi_f \in \mathbb{R}^{768}$ | per-family CLIP image-feature centroid (built offline, stored in pack) |
| $M_f \in [0,1]^{1 \times 1 \times h \times w}$ | per-family spatial guidance mask at step $t$ |
| $\epsilon^{\text{cfg}}_t, \epsilon^{\varnothing}_t$ | classifier-free-guided / unconditional noise predictions |
| $s, \tau_{\text{txt}}, \tau_{\text{img}}, \tau_{\text{cas}}$ | safety scale, text/img attention thresholds, CAS gate |

---

## 2. Family-grouped exemplar pack (offline)

For each erasure-target concept $C$ (e.g., harassment, violence, illegal\_activity), an exemplar pack `clip_grouped.pt` is precomputed with $K=4$ families $f$ that semantically partition $C$. Each family $f$ stores:

* **`target_words`** — short text phrases describing $f$ (e.g., harassment / political\_mockery: `["trump in jail", "trump caricature", "south park parody", "political mockery"]`).
* **`anchor_words`** — text phrases describing the *safe inverse* of $f$ (e.g., `["trump speech", "dignified politician", "formal portrait", "official portrait"]`).
* **`target_clip_features`** — $N_f \times 768$ CLIP image features extracted from $N_f$ exemplar images of $f$.
* **`anchor_clip_features`** — analogous CLIP features for the anchor side (currently *not* used at inference; reserved for ablation).
* **`family_token_map`** — index $f \mapsto$ pseudo-token slot used by the image probe (Sec. 3).

Pack construction is dataset-agnostic; the same pack is reused across all backbones (SD1.4 / SD3 / FLUX1) for a given concept.

---

## 3. Per-family probe construction

Two parallel probes are attached to the U-Net cross-attention layers at resolutions $\{16, 32\}$ ([code] `args.attn_resolutions`).

### 3.1 Image probe (per-family pseudo-tokens)

Function `build_grouped_probe_embeds(target_clip_features, te, tok, n_img_tokens=4)` (`generate_family.py:222`):

1. Tokenize an empty string and run the CLIP text encoder $te$ to obtain a baseline hidden-state sequence $b \in \mathbb{R}^{1 \times 77 \times 768}$.
2. For each family $f \in \{1, \dots, K\}$:
   * Compute $\phi_f = \text{mean}_n(\text{target\_clip\_features}[f, n, :])$.
   * Inject $\phi_f$ into the $n_{\text{img}}$ token slots reserved for $f$ inside $b$, yielding the **probe embedding** $b'$ with $K$ family-specific pseudo-token positions.
3. `precompute_target_keys(unet, b', resolutions)` projects $b'$ through the U-Net's cross-attention $K, V$ projections; the resulting $K_{\text{img}}$ tensors are the *fixed query targets* the probe attempts to detect during inference (`register_attention_probe`, `generate_family.py:434`).

At step $t$, attention scores $A^{\text{img}}_t \in \mathbb{R}^{H \times W}$ aligned to family $f$'s pseudo-token slot are read out via `compute_attention_spatial_mask(img_probe, token_indices=[f+1])` (`generate_family.py:506`).

### 3.2 Text probe (per-family keyword tokens)

A **text probe** is registered using the family-level text embedding $E_f^{\text{tgt}} = \text{encode\_concepts}(\text{te}, \text{tok}, \text{target\_words}_f[:3])$. Function `encode_concepts` (`generate_family.py:200`):
\[
E_f^{\text{tgt}} = \frac{1}{3}\sum_{j=1}^{3} \text{te}\bigl(\text{tok}(\text{target\_words}_f[j])\bigr)\quad \in \mathbb{R}^{1 \times 77 \times 768}
\]
i.e., **token-position-wise mean over 3 target keyword phrases** (last hidden state, padding tokens included).

In `both` mode, the text probe is registered alongside the image probe via `register_dual_attention_probe` (`generate_family.py:438`). At step $t$, attention scores at the keyword-token positions of family $f$ inside the *user prompt* are read out as $A^{\text{txt}}_{t,f} \in \mathbb{R}^{H \times W}$.

> Anchor embeddings $E_f^{\text{anc}} = \frac{1}{3}\sum_j \text{te}(\text{tok}(\text{anchor\_words}_f[j]))$ are computed identically and used only in the *guidance* step (Sec. 6), not the probe.

---

## 4. Per-family mask computation

For each family $f$ at step $t$ (`generate_family.py:498-528`):

\[
M_f^{\text{img}}(i,j) = \sigma\!\bigl(\alpha\bigl[\text{Blur}_{\sigma_b}(A^{\text{img}}_{t,f})(i,j) - \tau_{\text{img}}\bigr]\bigr)
\]
\[
M_f^{\text{txt}}(i,j) = \sigma\!\bigl(\alpha\bigl[\text{Blur}_{\sigma_b}(A^{\text{txt}}_{t,f})(i,j) - \tau_{\text{txt}}\bigr]\bigr)
\]
\[
\boxed{M_f(i,j) = \max\!\bigl(M_f^{\text{img}}(i,j),\ M_f^{\text{txt}}(i,j)\bigr)}
\]

* $\text{Blur}_{\sigma_b}$: Gaussian blur (`blur_sigma=1.0`).
* $\sigma$: sigmoid; $\alpha$ = `attn_sigmoid_alpha` (default 10.0).
* If only one probe is enabled (`probe_mode in {image, text}`), the corresponding term is omitted.

---

## 5. Inter-family overlap resolution (winner-take-all)

When two families fire at the same pixel, the stronger one wins (`generate_family.py:531-536`):
\[
w(i,j) = \arg\max_{f \in \{1, \dots, K\}} M_f(i,j), \quad
M_f(i,j) \leftarrow M_f(i,j) \cdot \mathbb{1}[w(i,j)=f]
\]
After this step, the family masks are mutually disjoint over space.

---

## 6. Family-conditioned guidance

For each family $f$, two extra U-Net forward passes are issued at step $t$ (`generate_family.py:540-547`):
\[
\epsilon^{\text{tgt}}_{t,f} = U(z_t, t,\ E_f^{\text{tgt}}), \quad
\epsilon^{\text{anc}}_{t,f} = U(z_t, t,\ E_f^{\text{anc}})
\]

`apply_family_guidance` (`generate_family.py:100-141`) then composes the safe noise prediction by accumulating per-family contributions over $f$:

### 6.1 `anchor_inpaint`
\[
\tilde{\epsilon}^{\text{anc}}_{t,f} = \epsilon^{\varnothing}_t + w_{\text{cfg}} \,(\epsilon^{\text{anc}}_{t,f} - \epsilon^{\varnothing}_t)
\]
\[
\beta_f = \min(s \cdot M_f, 1)
\]
\[
\epsilon^{\text{safe}}_t \leftarrow (1 - \beta_f)\,\epsilon^{\text{safe}}_t + \beta_f\,\tilde{\epsilon}^{\text{anc}}_{t,f}
\]
(blend-replace with the CFG-amplified anchor in masked regions)

### 6.2 `hybrid`
\[
\epsilon^{\text{safe}}_t \leftarrow \epsilon^{\text{safe}}_t - s\,M_f\,(\epsilon^{\text{tgt}}_{t,f} - \epsilon^{\varnothing}_t) + s\,M_f\,(\epsilon^{\text{anc}}_{t,f} - \epsilon^{\varnothing}_t)
\]
(spatially-weighted negative-target + positive-anchor push, additive over families)

### 6.3 `target_sub` (rarely used)
\[
\epsilon^{\text{safe}}_t \leftarrow \epsilon^{\text{safe}}_t - s\,M_f\,(\epsilon^{\text{tgt}}_{t,f} - \epsilon^{\varnothing}_t)
\]

NaN/Inf are caught (`isfinite` mask) and reverted to $\epsilon^{\text{cfg}}_t$.

---

## 7. CAS gate (when to apply guidance)

`GlobalCAS(threshold=τ_cas)` (`generate_family.py` near step loop) tracks the per-step max attention score across the registered probe:
\[
\text{trigger}_t = \mathbb{1}\!\bigl[\max_{f, i, j} A_{t,f}(i,j) > \tau_{\text{cas}}\bigr]
\]
Only when $\text{trigger}_t = 1$ are the per-family forwards (Sec. 6) executed; otherwise the standard CFG noise prediction is kept ($\epsilon^{\text{safe}}_t = \epsilon^{\text{cfg}}_t$). This saves $\sim 2K$ U-Net calls per skipped step.

* Sticky variant: once triggered, remains on for subsequent steps until end (paper option).
* SD1.4 default $\tau_{\text{cas}} = 0.6$. SD3 requires lower $\tau_{\text{cas}} \in \{0.3, 0.4\}$ (DiT attention magnitude differs).

---

## 8. Implementation details

| Item | Value |
|---|---|
| Attention resolutions probed | $\{16, 32\}$ for SD1.4 U-Net; analogous for SD3 / FLUX1 DiT blocks |
| `n_img_tokens` per family | 4 |
| Number of target/anchor word phrases used | first 3 from pack (`[:3]`) |
| Probe-mask sigmoid sharpness $\alpha$ | 10.0 |
| Mask Gaussian blur $\sigma_b$ | 1.0 |
| Default sampler | DDIM, 50 steps (SD1.4), 28 (SD3, FLUX1) |
| Default cfg scale | 7.5 (SD1.4), 7.0 (SD3), 3.5 (FLUX1, embedded) |
| Resolution | 512×512 (SD1.4 default), 1024×1024 (SD3, FLUX1) |
| Backbones supported | CompVis SD1.4, stabilityai SD3-medium, BFL FLUX.1-dev |

### Inter-backbone differences

* **SD1.4 (U-Net)**: cross-attn at $h=16, 32$ resolutions, $\tau_{\text{cas}}=0.6$, hybrid `safety_scale` $\in [10, 30]$, anchor `safety_scale` $\in [0.5, 3]$.
* **SD3 (MM-DiT)**: scaled-dot-product attention, $\tau_{\text{cas}}\in \{0.3, 0.4\}$, hybrid `safety_scale` $\approx 20$, anchor $\in [0.5, 3]$.
* **FLUX.1-dev**: distilled flow model, *embedded guidance* (no CFG negative pass), `safety_scale` $\approx 2$ for hybrid, $\in [1, 3]$ for anchor.

---

## 9. What the framework is *not* doing

* `target_prompts` / `anchor_prompts` (long natural-language sentences in pack) are **only used during offline pack construction** to extract `target_clip_features`. At inference, only the short `target_words` / `anchor_words` are encoded.
* `anchor_clip_features` (image side of the anchor) are loaded into memory but currently **not consumed** by any forward path; reserved for future ablation.
* Mask combination is **`max` not sum**; cross-family overlaps are **winner-take-all**, not blended.

---

## 10. Pseudocode (one denoising step in `family_guidance` mode)

```python
# Precomputed per concept (offline + at session start):
#   E_f^tgt, E_f^anc  : (K x 1 x 77 x 768), one per family
#   phi_f             : (K x 768) CLIP image centroids → pseudo-tokens b'
#   K_img, K_txt      : per-resolution attention key tensors

for t in DDIMSteps:
    eps_cfg, eps_null = cfg_pass(z_t, c_p)           # standard CFG
    if max(probe_attn(z_t)) <= τ_cas:                 # CAS gate
        z_t = step(z_t, eps_cfg);  continue

    # 1. Per-family masks
    fam_masks = []
    for f in 1..K:
        M_img_f = make_mask(probe_img.attn_at(token=f), τ_img)
        M_txt_f = make_mask(probe_txt.attn_at(tokens=keyword_idx[f]), τ_txt)
        fam_masks.append(max(M_img_f, M_txt_f))

    # 2. Winner-take-all over families
    winner = argmax_f(fam_masks)
    fam_masks = [m * (winner == f) for f, m in enumerate(fam_masks)]

    # 3. Per-family target/anchor U-Net forwards
    fam_targets = [unet(z_t, t, E_f^tgt) for f in 1..K]
    fam_anchors = [unet(z_t, t, E_f^anc) for f in 1..K]

    # 4. Compose safe ε
    if mode == "anchor_inpaint":
        eps_safe = eps_cfg
        for f in 1..K:
            anc_cfg = eps_null + w_cfg * (fam_anchors[f] - eps_null)
            beta = min(s * fam_masks[f], 1)
            eps_safe = eps_safe * (1 - beta) + anc_cfg * beta
    elif mode == "hybrid":
        eps_safe = eps_cfg
        for f in 1..K:
            eps_safe -= s * fam_masks[f] * (fam_targets[f] - eps_null)
            eps_safe += s * fam_masks[f] * (fam_anchors[f] - eps_null)

    z_{t-1} = step(z_t, eps_safe)
```

---

## 11. File / function map (for paper Appendix)

| Functionality | File : function : line |
|---|---|
| Family pack loading | `generate_family.py:357-391` |
| Per-family text/anchor encoding | `encode_concepts` `:200-207` |
| Per-family image pseudo-token construction | `build_grouped_probe_embeds` `:222-241` |
| Image probe registration | `register_attention_probe` `:434` |
| Text probe registration (dual) | `register_dual_attention_probe` `:438` |
| Per-family mask (max of img/txt) | `:498-530` |
| Inter-family winner-take-all | `:531-536` |
| Per-family target/anchor U-Net forwards | `:540-547` |
| `apply_family_guidance` (anchor / hybrid / target_sub) | `:100-141` |
| `apply_single_guidance` (no-family ablation) | `:143-168` |
| CAS sticky gate | `GlobalCAS` class (top of file, near `class GlobalCAS`) |
