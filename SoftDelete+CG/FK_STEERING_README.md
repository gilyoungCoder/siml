# FK Steering with Nudity Classifier

Feynman-Kac (FK) Steering implementation for steering Stable Diffusion using your 3-class nudity classifier.

Based on: "A General Framework for Inference-time Scaling and Steering of Diffusion Models" (Singhal et al., 2025)

## Overview

FK Steering is a particle-based inference-time method that:
- ✅ **No training required** - works with pre-trained models
- ✅ **Better than fine-tuning** - smaller models outperform larger fine-tuned ones
- ✅ **Works with any reward** - uses your nudity classifier as reward function
- ✅ **Efficient** - outperforms best-of-n with same number of samples

### How it works

1. **Multiple particles (k)**: Generate k samples simultaneously
2. **Intermediate rewards**: Evaluate classifier on noisy latents at each step
3. **Resampling**: Keep high-reward particles, discard low-reward ones
4. **Target distribution**: Samples from `p(x) ∝ p_model(x) × exp(λ × r(x))`

## Your Classifier

```python
# Input: (latent, timestep)
# Output: [logit_not_people, logit_clothed, logit_nude]

classifier(x_t, t) -> [3]  # 3-class logits
```

## Files

- `fk_steering.py` - Core FK steering implementation
- `run_fk_steering.py` - Main script to run FK steering
- `test_fk_steering.sh` - Test suite with various configurations

## Quick Start

### 1. Simple run (steer towards clothed people)

```bash
python run_fk_steering.py \
    --prompt "a photo of a person at the beach" \
    --target_class 1 \
    --num_particles 4 \
    --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth
```

### 2. Comparison mode (FK vs Best-of-N vs Baseline)

```bash
python run_fk_steering.py \
    --prompt "a portrait of a woman in elegant dress" \
    --target_class 1 \
    --num_particles 4 \
    --compare \
    --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth
```

### 3. Run full test suite

```bash
bash test_fk_steering.sh
```

## Key Parameters

### Target Class
- `--target_class 0`: Not people (steer away from people)
- `--target_class 1`: Clothed people (default, maximize clothing)
- `--target_class 2`: Nude people (for red-teaming/safety testing)

### FK Steering Parameters
- `--num_particles` (k): Number of particles (default: 4)
  - More particles = better quality but slower
  - Paper shows k=4 often sufficient
  - k=2 can beat fine-tuned models!

- `--potential_type`: How to score particles
  - `max`: Prefer particles with highest reward seen so far (default)
  - `difference`: Prefer particles with increasing rewards
  - `sum`: Prefer particles with highest cumulative reward

- `--lambda_scale` (λ): Reward scaling (default: 10.0)
  - Higher = stronger steering
  - Lower = more diversity

- `--resampling_interval`: Resample every N steps (default: 10)
  - Lower = more frequent resampling (stronger steering)
  - Higher = more exploration

## Reward Functions

The classifier logits are converted to rewards:

```python
# Class 0 (not people): minimize people presence
reward = logit[0] - max(logit[1], logit[2])

# Class 1 (clothed): maximize clothing, minimize nudity
reward = logit[1] - logit[2]

# Class 2 (nude): maximize nudity (for red-teaming)
reward = logit[2] - logit[1]
```

## Examples

### Example 1: Ensure clothed people in beach scenes

```bash
python run_fk_steering.py \
    --prompt "people at the beach playing volleyball" \
    --target_class 1 \
    --num_particles 4 \
    --lambda_scale 10.0 \
    --output_dir ./outputs/beach_clothed
```

### Example 2: Red-team testing (generate challenging content)

```bash
python run_fk_steering.py \
    --prompt "artistic human figure study" \
    --target_class 2 \
    --num_particles 8 \
    --lambda_scale 15.0 \
    --output_dir ./outputs/redteam
```

### Example 3: Remove people from scenes

```bash
python run_fk_steering.py \
    --prompt "a crowded street market" \
    --target_class 0 \
    --num_particles 4 \
    --lambda_scale 12.0 \
    --output_dir ./outputs/no_people
```

### Example 4: Scale number of particles

```bash
# Test with different k values
for K in 2 4 8 16; do
    python run_fk_steering.py \
        --prompt "portrait of a person" \
        --target_class 1 \
        --num_particles $K \
        --output_dir ./outputs/scaling_k${K} \
        --compare
done
```

## Expected Results

Based on the paper, you should see:

✅ **FK Steering (k=4) > Fine-tuned models**
- Better reward scores
- Better prompt fidelity
- Faster than fine-tuning

✅ **FK Steering > Best-of-N**
- Same number of particles
- Higher quality samples
- More efficient compute usage

✅ **Scaling benefits**
- k=2: Already beats baselines
- k=4: Strong performance
- k=8+: Marginal improvements

## Output Files

```
fk_steering_outputs/
├── prompt_target1_clothed_comparison.png       # Side-by-side comparison
├── prompt_target1_clothed_all_particles.png    # All k particles
├── prompt_target1_clothed_reward_history.png   # Reward over time
├── prompt_target1_clothed_baseline.png         # Single sample
├── prompt_target1_clothed_best_of_n.png        # Best-of-n result
├── prompt_target1_clothed_fk_steering.png      # FK steering result
└── prompt_target1_clothed_particle0_rewardX.png # Individual particles
```

## Monitoring

The script shows progress with:
- Progress bar for diffusion steps
- Mean reward at each resampling step
- Max reward at each resampling step
- Final results summary

Example output:
```
RESULTS:
--------------------------------------------------------------------------------
Best-of-N best reward: 2.3456
FK Steering best reward: 3.7891  <-- Higher!
FK Steering mean reward: 3.2145
--------------------------------------------------------------------------------
```

## Advanced Usage

### Custom Resampling Schedule

```python
# Resample only at specific steps
resampling_schedule = [0, 15, 30, 45]

results = fk_steering_pipeline(
    pipe=pipe,
    prompt=prompt,
    classifier=classifier,
    resampling_schedule=resampling_schedule,
    ...
)
```

### Use as Library

```python
from fk_steering import FKSteering, NudityClassifierReward

# Create reward function
reward_fn = NudityClassifierReward(
    classifier=classifier,
    target_class=1,  # clothed
)

# Create FK steering
fk = FKSteering(
    reward_fn=reward_fn,
    num_particles=4,
    potential_type='max',
    lambda_scale=10.0,
)

# Use in your pipeline...
```

## Troubleshooting

**Q: CUDA out of memory?**
```bash
# Reduce number of particles
--num_particles 2

# Or reduce image resolution in pipeline
# Or use CPU (very slow)
```

**Q: Rewards are too low/high?**
```bash
# Adjust lambda scale
--lambda_scale 5.0   # weaker steering
--lambda_scale 20.0  # stronger steering
```

**Q: Want more diversity?**
```bash
# Increase resampling interval
--resampling_interval 20

# Or reduce lambda
--lambda_scale 5.0
```

## Paper Results Recreation

To recreate paper-style experiments:

```bash
# Compare different k values
for K in 1 2 4 8 16; do
    python run_fk_steering.py \
        --prompt "your prompt" \
        --num_particles $K \
        --compare \
        --output_dir ./paper_recreation/k${K}
done

# Compare different potentials
for POT in max difference sum; do
    python run_fk_steering.py \
        --prompt "your prompt" \
        --potential_type $POT \
        --compare \
        --output_dir ./paper_recreation/${POT}
done
```

## Citation

If you use this implementation, cite the original paper:

```bibtex
@article{singhal2025feynmankac,
  title={A General Framework for Inference-time Scaling and Steering of Diffusion Models},
  author={Singhal, Raghav and Horvitz, Zachary and Teehan, Ryan and others},
  journal={arXiv preprint arXiv:2501.06848},
  year={2025}
}
```

## References

- Original paper: https://arxiv.org/abs/2501.06848
- Your classifier: `./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth`
- Classes: 0=not people, 1=clothed, 2=nude
