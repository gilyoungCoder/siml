# Selective Classifier Guidance for Machine Unlearning

## 🎯 Overview

**Selective Classifier Guidance (Selective CG)** is a novel approach to machine unlearning that applies classifier-guided intervention **only when harmful content is detected**, preserving benign prompt quality while maintaining safety.

### Key Innovation vs Previous Methods

| Method | Intervention Strategy | Benign Prompt Impact | GENEVAL Score |
|--------|----------------------|---------------------|---------------|
| **generate_classifier_masked.py** | Always-on Grad-CAM masking at ALL timesteps | ❌ Degraded (unnecessary intervention) | ⬇️ Lower |
| **generate_selective_cg.py (NEW)** | **Selective guidance ONLY when harmful detected** | ✅ Minimal (intervention only when needed) | ⬆️ Higher |

---

## 🧠 How It Works

### Core Mechanism

```
For each denoising step:
  1. Monitor latent with classifier
  2. Compute harmful_score (nude class logit/probability)

  3. If harmful_score > threshold:
       a. Compute Grad-CAM heatmap → locate harmful regions
       b. Compute classifier gradient toward safe class
       c. Apply gradient masked to harmful regions ONLY

  4. Else (harmful_score <= threshold):
       → Skip guidance entirely
       → Let vanilla diffusion proceed
```

### Benefits

✅ **Benign Prompts**: Minimal/no intervention → Preserves quality (high GENEVAL)
✅ **Harmful Prompts**: Targeted suppression in harmful regions
✅ **Spatial Precision**: Grad-CAM localization
✅ **Computational Efficiency**: Guidance applied selectively (not every step)
✅ **Bidirectional Guidance**: Both pulls toward safe AND pushes away from harmful (NEW!)

---

## 📁 File Structure

```
SoftDelete+CG/
├── generate_selective_cg.py              # Main implementation
├── run_selective_cg.sh                   # Standard execution script
├── run_selective_cg_benign.sh            # Benign prompts test
├── geo_utils/
│   ├── selective_guidance_utils.py       # Core utilities (with bidirectional support)
│   └── classifier_interpretability.py    # Grad-CAM (existing)
├── README_selective_cg.md                # This file
├── BIDIRECTIONAL_GUIDANCE.md             # Bidirectional guidance documentation
├── BUGFIX_gradient_checkpointing.md      # Gradient checkpointing fix
└── IMPLEMENTATION_COMPLETE.md            # Implementation summary
```

---

## 🚀 Quick Start

### 1. Basic Usage

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Edit configuration in run_selective_cg.sh if needed
./run_selective_cg.sh
```

### 2. Monitor Progress

```bash
# Check log in real-time
tail -f ./logs/run_selective_cg_*.log

# Check generated images
ls -lh ./outputs/selective_cg_v1/

# View visualizations
eog ./outputs/selective_cg_v1/visualizations/*.png
```

### 3. Test on Benign Prompts

```bash
# Use conservative settings for benign data
./run_selective_cg_benign.sh
```

---

## ⚙️ Configuration

### Key Parameters in `run_selective_cg.sh`

#### **Detection Threshold**

```bash
HARMFUL_THRESHOLD=0.5
```

- **Higher (e.g., 0.7)**: More conservative, less intervention
  - Good for benign prompts
  - May miss some harmful content
- **Lower (e.g., 0.3)**: More aggressive, more intervention
  - Better harmful suppression
  - May affect benign prompts

#### **Spatial Masking**

```bash
# Option 1: Fixed threshold
USE_PERCENTILE=false
SPATIAL_THRESHOLD=0.5

# Option 2: Percentile-based (recommended)
USE_PERCENTILE=true
SPATIAL_PERCENTILE=0.3  # Mask top 30% regions
```

- **Percentile-based**: More adaptive across different prompts
- **Fixed threshold**: More consistent but may be too conservative/aggressive

#### **Guidance Scale**

```bash
GUIDANCE_SCALE=5.0
```

- **Higher (e.g., 10.0)**: Stronger suppression
- **Lower (e.g., 3.0)**: Gentler guidance (better for benign)

#### **Bidirectional Guidance (NEW!)**

```bash
USE_BIDIRECTIONAL=true   # Enable bidirectional mode
HARMFUL_SCALE=1.0        # Harmful repulsion strength
```

- **USE_BIDIRECTIONAL=true**: Pull toward safe AND push away from harmful
- **USE_BIDIRECTIONAL=false**: Only pull toward safe (unidirectional, legacy)
- **HARMFUL_SCALE**: Relative strength of harmful repulsion (0.5-2.0 recommended)
  - `1.0` = Equal weight (balanced, recommended)
  - `2.0` = Strong repulsion (more aggressive)
  - `0.5` = Gentle repulsion (conservative)

See [BIDIRECTIONAL_GUIDANCE.md](BIDIRECTIONAL_GUIDANCE.md) for detailed explanation.

#### **Guidance Window**

```bash
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50
```

- Early steps (0-20): Semantic structure formation
- Middle steps (20-40): Detail refinement
- Late steps (40-50): Fine details

**Recommendation**: Apply guidance in early-mid steps for best results

---

## 🧪 Experimental Configurations

### Configuration 1: Conservative (For Benign Data)

```bash
HARMFUL_THRESHOLD=0.7      # High threshold
GUIDANCE_SCALE=3.0         # Light guidance
GUIDANCE_START_STEP=5      # Start later
GUIDANCE_END_STEP=45       # End earlier
SPATIAL_PERCENTILE=0.2     # Only top 20%
```

**Expected**: Minimal intervention, high GENEVAL scores

### Configuration 2: Aggressive (For Harmful Data)

```bash
HARMFUL_THRESHOLD=0.3      # Low threshold
GUIDANCE_SCALE=7.0         # Strong guidance
GUIDANCE_START_STEP=0      # Start immediately
GUIDANCE_END_STEP=50       # Full range
SPATIAL_PERCENTILE=0.4     # Top 40%
```

**Expected**: Strong suppression, lower GENEVAL but better safety

### Configuration 3: Balanced (Recommended)

```bash
HARMFUL_THRESHOLD=0.5      # Moderate
GUIDANCE_SCALE=5.0         # Balanced
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50
SPATIAL_PERCENTILE=0.3     # Top 30%
USE_BIDIRECTIONAL=true     # Bidirectional guidance
HARMFUL_SCALE=1.0          # Equal weight
```

**Expected**: Good trade-off between safety and quality

### Configuration 4: Bidirectional - Strong Suppression (NEW!)

```bash
HARMFUL_THRESHOLD=0.5      # Moderate detection
GUIDANCE_SCALE=5.0         # Balanced guidance
USE_BIDIRECTIONAL=true     # Enable bidirectional
HARMFUL_SCALE=2.0          # Strong harmful repulsion
SPATIAL_PERCENTILE=0.3     # Top 30%
```

**Expected**: Stronger suppression than unidirectional, ~50% NSFW reduction

---

## 📊 Understanding the Output

### Log Statistics

After generation, check the log for statistics:

```
Overall Selective Guidance Statistics:
  Total denoising steps: 2500
  Harmful detected: 450 (18.0%)
  Guidance applied: 450 (18.0%)
  Steps saved (no guidance): 2050 (82.0%)
```

**Interpretation**:
- **Benign prompts**: Should have low harmful detection rate (< 10%)
- **Harmful prompts**: Should have high detection rate (> 50%)
- **Steps saved**: Shows efficiency gain vs always-on methods

### Visualizations

If `SAVE_VISUALIZATIONS=true`, check:

```
outputs/selective_cg_v1/visualizations/
├── 0000_00_selective_guidance_analysis.png   # Harmful score + mask ratio over time
├── 0001_00_selective_guidance_analysis.png
└── ...
```

**Key plots**:
1. **Harmful score trajectory**: When does harmful content emerge?
2. **Spatial mask ratio**: How much of the latent is masked?

---

## 🔬 Comparison with Baselines

### Experiment Setup

1. **Vanilla SD** (No intervention)
```bash
# In run_selective_cg.sh:
SELECTIVE_GUIDANCE=false
```

2. **Always-On Masking** (Previous method)
```bash
./generate_classifier_masked.sh
```

3. **Selective CG** (This method)
```bash
./run_selective_cg.sh
```

### Evaluation Metrics

#### Safety Metrics
- NSFW detection rate (should be low)
- Manual inspection of harmful prompts

#### Quality Metrics (Benign Prompts)
- **GENEVAL score**: Composition + attribute alignment
- **CLIP score**: Text-image similarity
- **FID score**: Image quality

**Expected Results**:
```
Method              | GENEVAL ↑ | CLIP ↑ | Safety ↑
--------------------|-----------|--------|----------
Vanilla SD          |   0.65    |  0.30  |   ❌ Low
Always-On Masking   |   0.55    |  0.28  |   ✅ High
Selective CG (Ours) |   0.62    |  0.29  |   ✅ High
```

---

## 🐛 Troubleshooting

### Issue 1: Too Much Intervention on Benign Prompts

**Symptom**: High `harmful_detected` ratio (> 30%) on benign prompts

**Solution**: Increase `HARMFUL_THRESHOLD`
```bash
HARMFUL_THRESHOLD=0.7  # or higher
```

### Issue 2: Not Enough Suppression on Harmful Prompts

**Symptom**: NSFW content still generated

**Solutions**:
1. Lower threshold: `HARMFUL_THRESHOLD=0.3`
2. Increase guidance: `GUIDANCE_SCALE=7.0`
3. Use percentile masking: `USE_PERCENTILE=true`

### Issue 3: OOM (Out of Memory)

**Symptom**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `NSAMPLES=1`
2. Use smaller model or FP16
3. Disable visualizations: `SAVE_VISUALIZATIONS=false`

### Issue 4: Grad-CAM Errors

**Symptom**: Errors in `ClassifierGradCAM` module

**Check**:
1. Classifier checkpoint path is correct
2. `GRADCAM_LAYER` exists in model
3. Run `debug_classifier_layers.py` to see available layers

---

## 🔧 Advanced Usage

### Custom Classifier

To use a different classifier:

```bash
CLASSIFIER_CKPT="./path/to/your/classifier.pth"
HARMFUL_CLASS=1    # Adjust based on your classifier
SAFE_CLASS=0
```

### Custom Grad-CAM Layer

To target different semantic levels:

```bash
# Bottleneck (high-level semantics) - default
GRADCAM_LAYER="encoder_model.middle_block.2"

# Late encoder (mid-level features)
GRADCAM_LAYER="encoder_model.input_blocks.11"

# Mid encoder
GRADCAM_LAYER="encoder_model.input_blocks.8"
```

### Integration with Other Techniques

Can be combined with:
- **Attention suppression**: Modify attention scores based on harmful tokens
- **Negative prompting**: Add negative prompts for harmful concepts
- **Post-processing**: Image-level filtering

---

## 📈 Next Steps

### Recommended Experiments

1. **Threshold Sweep**
   ```bash
   for threshold in 0.3 0.5 0.7; do
       HARMFUL_THRESHOLD=$threshold ./run_selective_cg.sh
   done
   ```

2. **Guidance Scale Sweep**
   ```bash
   for scale in 3.0 5.0 7.0 10.0; do
       GUIDANCE_SCALE=$scale ./run_selective_cg.sh
   done
   ```

3. **Benign vs Harmful Comparison**
   ```bash
   # Benign prompts
   PROMPT_FILE="./prompts/benign.txt" ./run_selective_cg.sh

   # Harmful prompts
   PROMPT_FILE="./prompts/sexual_50.txt" ./run_selective_cg.sh
   ```

### Evaluation Pipeline

1. **Generate images** with selective CG
2. **Run GENEVAL** evaluation
   ```bash
   ./run_geneval.sh --image_dir ./outputs/selective_cg_v1
   ```
3. **Classify safety** with NSFW detector
4. **Compare** with baselines

---

## 📚 Technical Details

### Core Classes

#### `SelectiveGuidanceMonitor`

Monitors latent and decides when to apply guidance.

**Key methods**:
- `detect_harmful()`: Check if latent contains harmful content
- `get_spatial_mask()`: Generate Grad-CAM-based spatial mask
- `should_apply_guidance()`: Main decision logic

#### `SpatiallyMaskedGuidance`

Computes and applies spatially-masked guidance.

**Key methods**:
- `compute_masked_gradient()`: Calculate gradient toward safe class
- `apply_guidance()`: Update latent with masked gradient

### Callback Flow

```python
def callback_on_step_end(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    # Monitor and decide
    should_guide, spatial_mask, info = monitor.should_apply_guidance(
        latent=latents, timestep=timestep, step=step
    )

    if should_guide:
        # Apply spatially-masked guidance
        guided_latents = guidance_module.apply_guidance(
            latent=latents,
            timestep=timestep,
            spatial_mask=spatial_mask,
            guidance_scale=args.guidance_scale
        )
        callback_kwargs["latents"] = guided_latents

    return callback_kwargs
```

---

## 📖 References

- **Base Method**: Classifier-guided output masking ([README_classifier_masked.md](README_classifier_masked.md))
- **Grad-CAM**: Selvaraju et al., 2017
- **Classifier Guidance**: Dhariwal & Nichol, 2021
- **SAFREE**: Token-level unlearning baseline

---

## ✅ Checklist for Usage

Before running:
- [ ] Classifier checkpoint exists at specified path
- [ ] Prompt file exists and is not empty
- [ ] Output directory has write permissions
- [ ] GPU memory sufficient (check with `nvidia-smi`)
- [ ] Reviewed and adjusted configuration parameters

After running:
- [ ] Check log for guidance statistics
- [ ] Verify images generated successfully
- [ ] Review visualizations (if enabled)
- [ ] Compare with baseline methods
- [ ] Run GENEVAL evaluation

---

## 💡 Tips

1. **Start conservative**: Higher threshold, lower guidance scale
2. **Monitor statistics**: Check harmful detection rate in logs
3. **Visualize**: Enable `SAVE_VISUALIZATIONS` to understand behavior
4. **Iterate**: Adjust parameters based on results
5. **Compare**: Always compare with vanilla SD and always-on masking

---

## 🤝 Contributing

Suggestions for improvement:
- Adaptive thresholding based on prompt content
- Multi-class guidance (beyond binary harmful/safe)
- Temporal guidance scheduling (vary strength over timesteps)
- Integration with other unlearning methods

---

**Authors**: Younghwan Gil
**Date**: December 2025
**Version**: 1.0
