# Classifier-Guided Output Masking for Machine Unlearning

## 🎯 Overview

This is a **novel approach** to machine unlearning that uses a trained classifier to identify and suppress harmful content generation in Stable Diffusion **without retraining the model**.

### Key Innovation

Instead of just manipulating attention scores or applying gradient-based guidance, we directly **mask the cross-attention output** at spatial positions identified as containing harmful features by a classifier.

## 🧠 How It Works

### Cross-Attention Mechanism Recap

In Stable Diffusion's U-Net, cross-attention layers inject text semantics into the latent representation:

```
Query (Q):   From latent spatial positions [B, 4096, dim]  (64×64 flattened)
Key (K):     From CLIP text embeddings [B, 77, dim]
Value (V):   From CLIP text embeddings [B, 77, dim]

Attention weights = softmax(Q × K^T / sqrt(d))  [B, 4096, 77]
Output = Attention weights × V                   [B, 4096, dim]
```

The output is then reshaped and added back to the latent representation.

### Our Approach

1. **Classifier Detection**: At each diffusion step, we use Grad-CAM on a trained nudity classifier to identify which spatial regions in the latent contain nude features
   - Classifier outputs: `[Not People, Clothed, Nude]`
   - We focus on class 2 (Nude)

2. **Spatial Mask Creation**:
   - Grad-CAM generates heatmap at 64×64 resolution
   - Threshold or percentile-based binarization
   - Downsample to match different U-Net depths (32×32, 16×16, 8×8)

3. **Output Masking**:
   - During cross-attention forward pass
   - After computing `output = attention_weights × value`
   - Reshape output to spatial format: `[B, 4096, C] → [B, H, W, C]`
   - Apply mask: `output_spatial = output_spatial × mask`
   - Reshape back: `[B, H, W, C] → [B, HW, C]`

4. **Result**: Text semantics are **prevented from being injected** into nude regions

## 🔧 Implementation Details

### File Structure

```
generate_classifier_masked.py    # Main implementation
generate_classifier_masked.sh    # Shell script wrapper
geo_utils/
  ├── classifier_interpretability.py   # Grad-CAM implementation
  └── custom_stable_diffusion.py       # Custom pipeline
```

### Key Components

#### ClassifierMaskedAttnProcessor

Custom attention processor that:
- Extends `AttnProcessor2_0` (standard attention implementation)
- Stores current latent state for Grad-CAM computation
- Precomputes spatial masks at different resolutions
- Applies output masking during cross-attention

```python
class ClassifierMaskedAttnProcessor(AttnProcessor2_0):
    def __init__(
        self,
        classifier_model=None,
        mask_strategy="soft",      # hard, soft, adversarial
        mask_threshold=0.5,         # For binary masking
        mask_strength=0.8,          # For soft masking
        ...
    ):
```

#### Masking Strategies

1. **Hard Masking** (`mask_strategy="hard"`):
   ```python
   nude_mask = (heatmap > threshold).float()
   keep_mask = 1.0 - nude_mask
   output = output * keep_mask  # Binary: 0 or 1
   ```

2. **Soft Masking** (`mask_strategy="soft"`):
   ```python
   nude_mask = (heatmap > threshold).float()
   keep_mask = 1.0 - (nude_mask * mask_strength)
   output = output * keep_mask  # Weighted: 0.2 to 1.0
   ```

3. **Adversarial Masking** (`mask_strategy="adversarial"`):
   ```python
   # Can go negative (experimental)
   keep_mask = 1.0 - (nude_mask * mask_strength * 2)
   output = output * keep_mask  # Can be negative
   ```

#### Callback Mechanism

```python
def callback_on_step_end(pipeline, step, timestep, callback_kwargs, **kwargs):
    # Get current latent
    latents = callback_kwargs.get("latents")

    # Update processor with current latent for Grad-CAM
    attn_processor.set_step(step, int(timestep), latent=latents)

    # Enable/disable masking based on step range
    enable_mask = (mask_start_step <= step <= mask_end_step)
    attn_processor.set_masking_enabled(enable_mask)

    return callback_kwargs
```

## 🚀 Usage

### Basic Usage

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
./generate_classifier_masked.sh
```

### Configuration

Edit `generate_classifier_masked.sh`:

```bash
# Enable output masking
VALUE_MASKING=true

# Masking strategy
MASK_STRATEGY="soft"        # hard, soft, or adversarial
MASK_THRESHOLD=0.5          # Heatmap threshold
MASK_STRENGTH=0.8           # Soft masking strength

# Active range
MASK_START_STEP=0           # Start masking
MASK_END_STEP=50            # End masking

# Alternative: percentile-based
USE_PERCENTILE=true
MASK_PERCENTILE=0.3         # Mask top 30%
```

### Combining Techniques

You can combine output masking with other techniques:

```bash
# Attention score suppression
HARM_SUPPRESS=true
BASE_TAU=0.15

# Classifier guidance
CLASSIFIER_GUIDANCE=true
GUIDANCE_SCALE=15
```

## 📊 Expected Results

### What This Achieves

1. **Spatial Precision**: Unlike global attention suppression, this targets specific regions
2. **Dynamic Adaptation**: Mask is recomputed at each diffusion step
3. **Interpretability**: Grad-CAM shows exactly where suppression happens
4. **No Retraining**: Inference-time intervention only

### Comparison with Other Methods

| Method | Mechanism | Granularity | Requires Retraining |
|--------|-----------|-------------|---------------------|
| Fine-tuning | Update weights | Global | ✅ Yes |
| Attention suppression | Reduce attention scores | Token-level | ❌ No |
| Classifier guidance | Gradient-based steering | Global | ❌ No |
| **Output masking (ours)** | **Spatial output suppression** | **Spatial** | **❌ No** |

## 🧪 Debugging

Enable debug flags to monitor masking:

```bash
DEBUG=true
DEBUG_PROMPTS=true    # Per-token analysis
DEBUG_STEPS=true      # Per-step statistics
```

Debug output example:
```
[MASK Step 10] Masked: 1523/4096 (37.2%) | Heatmap: avg=0.423, max=0.987
[MASK Step 11] Masked: 1401/4096 (34.2%) | Heatmap: avg=0.401, max=0.963
```

## 🔬 Technical Notes

### Why Output Masking Works

Cross-attention output has shape `[B, spatial_positions, channels]`:
- Each spatial position corresponds to a location in the latent
- Output values represent "how much text semantic should be injected here"
- By masking output at nude positions, we prevent text from affecting those regions

### Spatial Correspondence

```
Latent:          [B, 4, 64, 64]
Flattened:       [B, 4096, ...]     (64×64 = 4096)
Grad-CAM mask:   [B, 64, 64]
Applied to:      [B, 64, 64, C]     (reshaped output)
```

### Multi-Resolution Handling

U-Net has different resolutions at different depths:
- Input blocks (early): 64×64
- Middle blocks: 8×8
- Output blocks: varies

We precompute masks for all resolutions: 64, 32, 16, 8.

## 📝 TODO / Future Work

1. **Ablation Studies**:
   - Compare hard vs soft vs adversarial masking
   - Test different threshold values
   - Analyze effect of step ranges

2. **Optimization**:
   - Cache Grad-CAM computation if latent doesn't change much
   - Batch Grad-CAM for multiple samples

3. **Extensions**:
   - Multi-class masking (mask different concepts differently)
   - Adaptive threshold per step
   - Combine with other interpretability methods (Integrated Gradients)

4. **Evaluation**:
   - Quantitative metrics (CLIP score, FID)
   - User studies
   - Compare with baseline unlearning methods

## 📚 References

- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- Stable Diffusion: [Rombach et al., 2022](https://arxiv.org/abs/2112.10752)
- Classifier Guidance: [Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233)

## 🙋 Contact

For questions or issues, please refer to the main repository.
