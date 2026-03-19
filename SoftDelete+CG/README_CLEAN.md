# Machine Unlearning: Clean Implementation

A minimal, clean implementation of machine unlearning for diffusion models using:
1. **Attention Manipulation**: Suppressing harmful concepts (e.g., nudity) via cosine similarity
2. **Classifier Guidance**: Guiding generation towards safe concepts (e.g., clothed people)

**Key Feature**: No parameter updates to the diffusion model - everything happens during inference.

## Files

- `generate.py` - Main generation script (clean implementation)
- `generate.sh` - Bash script to run generation with all parameters
- `configs/harm_concepts.txt` - List of harmful concepts to suppress (one per line)

## Core Methodology

### 1. Attention Manipulation (Deletion)
- Builds a harmful concept vector from text embeddings (e.g., "nudity", "naked")
- During cross-attention, computes cosine similarity between token embeddings and harmful vector
- Suppresses attention scores for tokens with similarity ≥ τ (threshold)
- Suppression strength γ decreases linearly from early to late steps

### 2. Classifier Guidance (Maintenance)
- Uses a pre-trained time-dependent classifier
- Guides the generation process towards target class (class 1 = clothed people)
- Applied at each denoising step after `guidance_start_step`

## Quick Start

### 1. Edit Configuration (generate.sh)

```bash
# Model & Data
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/country_nude_body.txt"
OUTPUT_DIR="./output_img/unlearning_clean"

# Harmful Concept Suppression
HARM_SUPPRESS=true
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"
HARM_TAU=0.15           # Cosine similarity threshold
HARM_GAMMA_START=40.0   # Suppression strength (early steps)
HARM_GAMMA_END=0.5      # Suppression strength (late steps)

# Classifier Guidance
CLASSIFIER_GUIDANCE=true
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
GUIDANCE_SCALE=5.0
TARGET_CLASS=1  # 1 = clothed people
```

### 2. Run Generation

```bash
./generate.sh
```

Monitor progress:
```bash
tail -f ./logs/run_*.log
```

### 3. Direct Python Usage

```bash
python generate.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file ./prompts/country_nude_body.txt \
    --output_dir ./output_img/test \
    --harm_suppress \
    --harm_concepts_file ./configs/harm_concepts.txt \
    --harm_tau 0.15 \
    --harm_gamma_start 40.0 \
    --harm_gamma_end 0.5 \
    --classifier_guidance \
    --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth \
    --guidance_scale 5.0 \
    --target_class 1
```

## Configuration Files

### harm_concepts.txt
Add harmful concepts to suppress, one per line:
```
nudity
nude
naked
```

## Key Parameters

### Attention Manipulation
- `--harm_tau`: Cosine similarity threshold (default: 0.15)
  - Higher = more aggressive suppression
  - Lower = more selective suppression

- `--harm_gamma_start`: Initial suppression strength (default: 40.0)
  - Applied at early denoising steps

- `--harm_gamma_end`: Final suppression strength (default: 0.5)
  - Applied at late denoising steps
  - Linear decay from start to end

### Classifier Guidance
- `--guidance_scale`: Strength of classifier guidance (default: 5.0)
  - Higher = stronger push towards target class

- `--target_class`: Target class index (default: 1)
  - 0 = nude, 1 = clothed people, 2 = not people (adjust based on your classifier)

- `--guidance_start_step`: When to start guidance (default: 1)
  - Can delay guidance to later steps if needed

## Architecture

### generate.py Structure

1. **HarmSuppressionAttnProcessor**: Custom attention processor
   - Inherits from `AttnProcessor2_0`
   - Intercepts cross-attention computation
   - Applies cosine-similarity-based suppression

2. **build_harm_vector()**: Creates harmful concept representation
   - Tokenizes harmful concepts
   - Extracts text encoder embeddings (layer -2)
   - Mean pools across concepts and tokens

3. **Main generation loop**:
   - Loads model and prompts
   - Sets up attention processor and classifier guidance
   - Generates images with per-step callbacks

## Comparison with Original Code

### Removed Features
- ❌ SAE Probe integration
- ❌ ADD-LIST / ALLOW-LIST mechanisms
- ❌ EOT hard blocking
- ❌ Multiple harm vector modes (kept only masked_mean)
- ❌ Per-prompt harm vector updates
- ❌ Prompt append/modification logic
- ❌ Debug print utilities

### Kept Features
- ✅ Core attention manipulation via cosine similarity
- ✅ Classifier guidance for target class
- ✅ Linear gamma scheduling
- ✅ Normalized embeddings (layer -2)

## Design Philosophy

**Addition + Deletion Approach**:
- **Deletion** (Attention Manipulation): Remove specific harmful keywords
  - Effective for explicit concepts that can be named
  - Few concepts to delete → use keyword-based approach

- **Maintenance** (Classifier Guidance): Preserve implicit safe concepts
  - Effective for implicit concepts hard to describe with words
  - Many concepts to preserve → use classifier-based approach

This hybrid approach is more effective than either method alone:
- CG alone: Good at following/maintaining, poor at negation
- Attention alone: Good at explicit deletion, limited for preservation

## Troubleshooting

### Generated images still contain harmful content
- Increase `HARM_TAU` (e.g., 0.15 → 0.20)
- Increase `HARM_GAMMA_START` (e.g., 40.0 → 50.0)
- Increase `GUIDANCE_SCALE` (e.g., 5.0 → 7.0)
- Add more concepts to `harm_concepts.txt`

### Generated images are over-suppressed / quality degraded
- Decrease `HARM_TAU` (e.g., 0.15 → 0.10)
- Decrease `HARM_GAMMA_START` (e.g., 40.0 → 30.0)
- Decrease `GUIDANCE_SCALE` (e.g., 5.0 → 3.0)

### Classifier guidance not working
- Verify classifier checkpoint path
- Check `TARGET_CLASS` index matches your classifier's class definitions
- Ensure classifier config YAML is correct

## Citation

If you use this code, please cite the original research on:
- Classifier Guidance for diffusion models
- Machine unlearning via attention manipulation
- Soft deletion vs hard blocking approaches

## License

Same as the original SoftDelete+CG project.
