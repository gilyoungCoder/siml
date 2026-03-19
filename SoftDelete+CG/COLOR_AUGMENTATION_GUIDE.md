# Color Augmentation for Nudity Classifier Training

## Problem: Color Bias in Classifier

### Observation
When using the trained classifier for guidance, generated images become **overly blue/cyan-tinted**.

### Root Cause Analysis
The classifier likely learned **spurious correlations**:
- **Nude = Skin tones (red/orange/pink hues)**
- **Clothed = Blue/cyan (clothing colors)**

When classifier guidance pushes away from "nude" class, it inadvertently pushes toward blue colors, causing unnatural color shifts.

---

## Solution: Color Augmentation During Training

### Implemented Changes

#### File Modified: `train_nudity_classifier.py`

**Lines 167-175 (Benign Transform):**
```python
benign_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(0.5),
    # Color augmentation to reduce color bias (prevent classifier from learning skin-tone = nude)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # 10% chance to convert to grayscale
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])
```

**Lines 184-192 (Target Transform):**
```python
target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(0.5),
    # Color augmentation to reduce color bias (prevent classifier from learning skin-tone = nude)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # 10% chance to convert to grayscale
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])
```

---

## Augmentation Parameters Explained

### ColorJitter
```python
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| **brightness** | 0.3 | Randomly vary brightness by ±30% |
| **contrast** | 0.3 | Randomly vary contrast by ±30% |
| **saturation** | 0.3 | Randomly vary saturation by ±30% |
| **hue** | 0.1 | Randomly shift hue by ±10% (color spectrum shift) |

**Impact:**
- Prevents classifier from learning "skin tone = nude"
- Forces classifier to focus on **semantic features** (body parts, poses, context) rather than color
- Training data now includes:
  - Bright/dark skin tones
  - High/low contrast images
  - Desaturated (less colorful) images
  - Hue-shifted images (skin tones → greenish, bluish variations)

### RandomGrayscale
```python
transforms.RandomGrayscale(p=0.1)
```

- **10% of training images** converted to grayscale
- Forces classifier to **ignore color information entirely** for some samples
- Ensures classifier can classify nude vs. clothed **without color cues**

---

## Training Script

### New Script: `train_nudity_color_aug.sh`

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

output_dir=nudity_three_class_color_aug
benign_data_path=/mnt/home/yhgil99/dataset/benign_data
nudity_data_path=/mnt/home/yhgil99/dataset/nudity  # Update path!

train_batch_size=16
save_ckpt_freq=200
num_train_epochs=20
learning_rate=1.0e-4

python train_nudity_classifier.py \
    --benign_data_path $benign_data_path \
    --nudity_data_path $nudity_data_path \
    --output_dir work_dirs/$output_dir \
    --train_batch_size $train_batch_size \
    --num_train_epochs $num_train_epochs \
    --learning_rate $learning_rate \
    --report_to wandb \
    --use_wandb
```

### Running Training

1. **Update dataset paths** in `train_nudity_color_aug.sh`:
   ```bash
   nudity_data_path=/path/to/your/nudity/dataset
   ```

2. **Run training**:
   ```bash
   cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
   ./train_nudity_color_aug.sh
   ```

3. **Monitor logs**:
   ```bash
   tail -f train_nudity_color_aug.log
   ```

4. **Check checkpoints**:
   ```bash
   ls -lh work_dirs/nudity_three_class_color_aug/checkpoint/
   ```

---

## Expected Results

### Before Color Augmentation
- ❌ Classifier learns: "Nude = Red/Orange/Pink, Clothed = Blue/Cyan"
- ❌ Guidance pushes images toward blue tones
- ❌ Unnatural color distributions (cyan-heavy images)

### After Color Augmentation
- ✅ Classifier learns semantic features (body structure, poses, clothing)
- ✅ Color-invariant classification
- ✅ Guidance preserves natural color distributions
- ✅ Better generalization to diverse skin tones and lighting conditions

---

## Testing the New Classifier

### 1. After Training Completes
Find the best checkpoint:
```bash
ls work_dirs/nudity_three_class_color_aug/checkpoint/
# Example: step_11800/classifier.pth
```

### 2. Update Guidance Script
Edit `run_always_adaptive_spatial_cg.sh`:
```bash
# OLD
CLASSIFIER_PATH="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# NEW
CLASSIFIER_PATH="./work_dirs/nudity_three_class_color_aug/checkpoint/step_XXXX/classifier.pth"
```

### 3. Recompute GradCAM Statistics (Optional but Recommended)
```bash
python compute_gradcam_statistics.py \
    --data_dir /path/to/harmful/images \
    --classifier_ckpt ./work_dirs/nudity_three_class_color_aug/checkpoint/step_XXXX/classifier.pth \
    --output_file ./gradcam_nudity_color_aug_stats.json \
    --num_samples 1000
```

Update `run_always_adaptive_spatial_cg.sh`:
```bash
GRADCAM_STATS_FILE="./gradcam_nudity_color_aug_stats.json"
```

### 4. Run Guidance
```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
./run_always_adaptive_spatial_cg.sh
```

### 5. Compare Results
- Check if generated images have more **natural color distributions**
- Verify that **blue tint issue is reduced**
- Evaluate both **safety (removing harmful content)** and **quality (natural colors)**

---

## Comparison: Old vs. New

| Aspect | Old Classifier | New Classifier (Color Aug) |
|--------|---------------|----------------------------|
| **Training Augmentation** | Horizontal flip only | Horizontal flip + ColorJitter + Grayscale |
| **Learned Features** | Color + Semantic | Primarily Semantic |
| **Color Bias** | High (skin-tone dependent) | Low (color-invariant) |
| **Guided Image Quality** | Cyan/blue tint | Natural colors ✅ |
| **Generalization** | Poor on diverse lighting | Better ✅ |
| **Skin Tone Robustness** | Biased toward specific tones | Robust to all tones ✅ |

---

## Troubleshooting

### If classifier performance drops:
- **Reduce augmentation strength**:
  ```python
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
  transforms.RandomGrayscale(p=0.05)  # 5% instead of 10%
  ```

### If color bias persists:
- **Increase augmentation strength**:
  ```python
  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)
  transforms.RandomGrayscale(p=0.15)  # 15%
  ```

### If training is too slow:
- Reduce `num_train_epochs` or increase `train_batch_size`

---

## Alternative Solutions (If Color Augmentation Doesn't Fully Solve)

1. **Multi-Scale Guidance** (adjust guidance strength by timestep)
2. **Perceptual Loss Constraint** (preserve CLIP embeddings)
3. **Channel-wise Gradient Clipping** (limit color channel changes in latent space)
4. **Adversarial Training** (train classifier to be robust to color perturbations)

See main discussion for details on these approaches.

---

## Files Modified/Created

### Modified:
- `../three_classificaiton/train_nudity_classifier.py`
  - Lines 167-175: Added color augmentation to benign transform
  - Lines 184-192: Added color augmentation to target transform

### Created:
- `SoftDelete+CG/train_nudity_color_aug.sh` - Training script
- `SoftDelete+CG/COLOR_AUGMENTATION_GUIDE.md` - This guide

---

## Next Steps

1. ✅ Color augmentation added to training script
2. 🔄 Train new classifier with color augmentation
3. 🔄 Test guided generation with new classifier
4. 🔄 Compare results: color distribution, FID, safety metrics
5. 🔄 Optionally: Implement multi-scale guidance if color bias persists

---

## Summary

**Problem**: Classifier guidance causes unnatural blue tints due to color bias.

**Solution**: Add color augmentation (ColorJitter + RandomGrayscale) during training to force classifier to learn semantic features instead of color patterns.

**Expected Outcome**: More natural color distributions in guided images while maintaining effective harmful content removal.
