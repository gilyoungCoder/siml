# SHGD (Self-Healing Guided Diffusion) — Experiment Results

**Date**: 2026-03-13
**Model**: CompVis/stable-diffusion-v1-4 (fp32)
**Eval**: Qwen2-VL-7B-Instruct (4-class: NotPeople/Safe/Partial/Full)
**Harmful prompts**: RingABell (79 prompts)
**Safe prompts**: COCO 10k subset

---

## 1. Method Overview

SHGD is a **training-free** inference-time safety method using:

1. **Dual-Anchor CFG**: `ε̂ = ε_cfg + α·(ε_anchor − ε_harm)`
   - Pushes generation away from harmful concepts toward safe anchors
2. **Critical Time Window**: Guidance only in early denoising steps where semantic layout forms
3. **Self-Healing**: After guidance window, SDEdit-style re-noise + re-denoise to recover quality
4. **CLIP Text Trigger**: Skip guidance on safe prompts to preserve quality (FID)

### Key v2 Improvements

| Component | v1 (old) | v2 (new) |
|---|---|---|
| Heal step | Standard CFG (undoes guidance!) | Safe CFG: `ε_harm + gs·(ε_cond − ε_harm)` |
| Trigger | None (guidance on all prompts) | CLIP text cosine sim (97.8% accuracy) |
| FID impact | 63.84 | **15.32** |

---

## 2. Safety Results (RingABell, Qwen VLM)

### v1 (Old Pipeline — heal undoes guidance)

| Experiment | α | Window | Heal | Safe | Partial | Full | SafeRate |
|---|---|---|---|---|---|---|---|
| **Baseline (no guidance)** | 0 | — | — | 10 | 29 | **40** | 49.37% |
| Default | 10 | [1.0, 0.78] | standard | 6 | 29 | **44** | 44.30% |
| Aggressive | 15 | [1.0, 0.78] | standard | 9 | 35 | 35 | 55.70% |
| Brutal | 25 | [1.0, 0.78] | standard | 11 | 31 | 37 | 53.16% |
| Wide window | 10 | [1.0, 0.50] | standard | 6 | 29 | 44 | 44.30% |
| Narrow window | 10 | [1.0, 0.90] | standard | 7 | 27 | 45 | 43.04% |
| Micro heal | 10 | [1.0, 0.78] | micro | 8 | 31 | 40 | 49.37% |
| No heal | 10 | [1.0, 0.78] | none | 14 | 25 | **26** | 49.37% |
| **Wide+weak** | **5** | **[0.8, 0.0]** | **standard** | **32** | **32** | **15** | **81.01%** |

> **Key finding**: Standard heal *undoes* guidance — "Default" (Full=44) is worse than baseline (Full=40).
> "No heal" has fewer Full (26) because it doesn't undo guidance, confirming the heal bug.
> "Wide+weak" works best because low α=5 over many steps + heal has less to undo.

### v2 (Fixed Pipeline — safe heal + CLIP trigger)

| Experiment | α | Window | Heal | Safe | Partial | Full | SafeRate |
|---|---|---|---|---|---|---|---|
| **v2 Default** | 10 | [1.0, 0.78] | safe | 33 | 27 | **19** | **75.95%** |
| **v2 Aggressive** | 15 | [1.0, 0.78] | safe | 33 | 26 | 20 | 74.68% |
| **v2 Wide+weak** | **5** | **[0.8, 0.0]** | **safe** | **31** | **33** | **15** | **81.01%** |

> **Safe heal fix**: v2 Default Full=19 vs old Default Full=44 (57% reduction)
> **Best config**: Wide+weak (α=5, [0.8, 0.0]) — Full=15, SafeRate=81.01%

---

## 3. Quality Results (FID)

COCO 500 prompts, FID computed vs baseline (no guidance) images.

| Method | FID ↓ |
|---|---|
| v1 SHGD (no trigger, guidance on all) | 63.84 |
| **v2 SHGD (CLIP trigger, t=0.32)** | **15.32** |

CLIP text trigger calibration (40 harmful + 50 safe prompts):
- Harmful sim range: [0.32, 0.61], mean=0.41
- Safe sim range: [0.12, 0.35], mean=0.22
- Optimal threshold: **0.321** → 97.78% accuracy (40/40 harmful TP, 48/50 safe TN)

---

## 4. Ablation Analysis

### Why does wide+weak (α=5, [0.8, 0.0]) work best?

1. **Gradual guidance**: Low α=5 applied over ~40 steps (80% of denoising) vs strong α=10-25 over ~11 steps
2. **Less distortion**: Smaller per-step perturbation stays closer to data manifold
3. **Heal is simpler**: By end of guidance, latents are less distorted → heal has less to fix
4. **Quality preserved**: Lower α means less artifact introduction

### Safe heal impact by config

| Config | Old heal Full | Safe heal Full | Δ |
|---|---|---|---|
| Default (α=10, narrow) | 44 | 19 | **-25** |
| Aggressive (α=15, narrow) | 35 | 20 | **-15** |
| Wide+weak (α=5, wide) | 15 | 15 | 0 |

Safe heal matters most for narrow-window configs where heal duration is long.
Wide-window configs barely affected because heal covers only the last few steps.

---

## 5. Config Files

### Best config (recommended)
```yaml
shgd:
  anchor_guidance_scale: 5.0      # moderate α
  guide_start_frac: 0.8           # start at 80% noise
  guide_end_frac: 0.0             # guide until end
  heal_strength: 0.3
  adaptive_heal: true
  enable_self_consistency: true
  consistency_threshold: 0.0      # disable early heal trigger
  enable_trigger: true
  trigger_sim_threshold: 0.32
```

### Default config (current)
```yaml
shgd:
  anchor_guidance_scale: 10.0
  guide_start_frac: 1.0
  guide_end_frac: 0.78
  heal_strength: 0.4
  adaptive_heal: true
  enable_self_consistency: true
  consistency_threshold: 0.85
  enable_trigger: true
  trigger_sim_threshold: 0.32
```

---

## 6. Output Directories

| Directory | Description |
|---|---|
| `outputs/full_run_20260313_031803/ablation/` | v1 ablation (8 configs) |
| `outputs/full_run_20260313_023408/ablation/` | v1 earlier run (incl. wide+weak) |
| `outputs/v2_safeheal_20260313_141716/` | v2 experiments |
| `outputs/fid_baseline_coco500/` | Baseline COCO 500 (no guidance) |
| `outputs/fid_shgd_coco500/` | v1 SHGD COCO 500 (symlinks) |
