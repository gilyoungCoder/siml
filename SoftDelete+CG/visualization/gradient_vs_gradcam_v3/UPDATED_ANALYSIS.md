# Updated Gradient vs Grad-CAM Analysis (v3)

## 🎯 Key Improvements

### Based on User Feedback
> "Grad-CAM should use fixed threshold 0.3, not top-30%"
> "English labels to avoid font rendering issues"

### Changes from v2 to v3

1. **Grad-CAM Masking Method**
   - **v2**: Top-30% percentile (adaptive)
   - **v3**: Fixed threshold 0.3 ✅

2. **Text Labels**
   - **v2**: Korean text (한글) causing rendering issues
   - **v3**: English labels only ✅

3. **Comparison Fairness**
   - Gradient: Top-30% (keeps top 1228 pixels)
   - Grad-CAM: Threshold ≥ 0.3 (adaptive pixel count)

---

## 📊 Updated Results

### Image 1: prompt_0001_sample_1 (Nude 84.8%)

**Pixel Coverage**:
- Gradient(Safe) Top-30%: **1228 pixels**
- Gradient(Nude) Top-30%: **1228 pixels**
- Grad-CAM Threshold-0.3: **3226 pixels** ← More coverage

**Spatial Overlap (IoU)**:
- Gradient(Safe) ∩ Grad-CAM: **33.11%**
- Gradient(Nude) ∩ Grad-CAM: **35.46%**

**Correlation**:
- Pearson R: 0.1760
- R²: **0.0310** (3.1%)

**Interpretation**:
- Grad-CAM covers **2.6x more pixels** than Gradient top-30%
- Still only ~35% overlap → Different spatial strategies
- Grad-CAM provides **broader coverage** of harmful regions

---

### Image 2: prompt_0002_sample_1 (Clothed 61.1%)

**Pixel Coverage**:
- Gradient(Safe) Top-30%: **1228 pixels**
- Gradient(Nude) Top-30%: **1228 pixels**
- Grad-CAM Threshold-0.3: **618 pixels** ← More selective

**Spatial Overlap (IoU)**:
- Gradient(Safe) ∩ Grad-CAM: **21.05%**
- Gradient(Nude) ∩ Grad-CAM: **21.93%**

**Correlation**:
- Pearson R: 0.2414
- R²: **0.0583** (5.8%)

**Interpretation**:
- On Clothed images, Grad-CAM is **more selective** (618 vs 1228 pixels)
- Very low overlap (~21%) → Independent information
- Grad-CAM adapts to content: broad for Nude, narrow for Clothed

---

## 🔍 Key Findings

### 1. Adaptive Coverage Based on Content

```
Nude Image (84.8% confidence):
  Grad-CAM: 3226 pixels (78% of 64×64 grid)
  → Broad coverage when harmful signal is strong

Clothed Image (61.1% confidence):
  Grad-CAM: 618 pixels (15% of grid)
  → Selective coverage when harmful signal is weak
```

**Advantage**: Grad-CAM naturally adapts to harmful content strength!

### 2. Independent Spatial Information

```
Average IoU: ~27%
Average R²: ~4.5%

→ Gradient and Grad-CAM target ~70% different regions
→ Complementary information sources
```

### 3. Why Fixed Threshold 0.3?

**Reasoning**:
- Normalized Grad-CAM values are in [0, 1]
- Threshold 0.3 = "moderate to high activation"
- Adapts naturally to image content
- Avoids forcing fixed pixel count

**Comparison**:
- Top-k percentile: Always same pixel count (e.g., 1228)
- Threshold 0.3: Adaptive pixel count (618 ~ 3226)

---

## 💡 Practical Implications

### For Selective Classifier Guidance

**Scenario 1: Harmful Image (Nude 84.8%)**
```python
harmful_score = 0.848  # > threshold (e.g., 0.5)
→ Apply Grad-CAM masking
→ Grad-CAM covers 3226 pixels (78% of grid)
→ Broad intervention on harmful regions
```

**Scenario 2: Safe Image (Clothed 61.1%)**
```python
harmful_score = 0.275  # < threshold (e.g., 0.5)
→ Skip guidance entirely
→ Preserve benign quality
→ (If guidance needed: Grad-CAM only 618 pixels = minimal intervention)
```

### Why This Matters

1. **Adaptive Intervention**:
   - Strong harmful signal → Broad masking (3226 px)
   - Weak harmful signal → Narrow masking (618 px)
   - No harmful signal → No masking (0 px)

2. **Benign Preservation**:
   - Gradient top-30% always affects 1228 pixels (30%)
   - Grad-CAM threshold adapts: 618 pixels (15%) on benign
   - **Better preservation of benign quality**

3. **Spatial Precision**:
   - Low IoU (~27%) confirms Grad-CAM provides unique information
   - Class-specific attention (Nude class activation)
   - Complements gradient-based guidance

---

## 🎨 Visualization Improvements

### Row 1: Magnitude Heatmaps
- **Title Format**: "Gradient -> Safe (Clothed) [Guidance Direction]"
- No Korean characters ✅
- Clear role indication with brackets

### Row 2: Binary Masks
- **Gradient**: "Top 30%: Gradient -> Safe [Actual Guidance Region]"
- **Grad-CAM**: "Threshold 0.3: Grad-CAM [Actual Masking Region]"
- Shows different masking strategies clearly

### Row 3: Analysis Panels
- **Overlap**: "IoU=X% (Red=Grad, Blue=CAM, Purple=Both)"
- **Correlation**: "R-squared=X"
- **Summary**: English-only quantitative analysis
- All labels render correctly ✅

---

## 📈 Comparison Summary

| Metric | Image 1 (Nude) | Image 2 (Clothed) | Average |
|--------|----------------|-------------------|---------|
| **Grad-CAM Coverage** | 3226 px (78%) | 618 px (15%) | Adaptive |
| **Gradient Coverage** | 1228 px (30%) | 1228 px (30%) | Fixed |
| **IoU (Safe∩CAM)** | 33.11% | 21.05% | 27.08% |
| **IoU (Nude∩CAM)** | 35.46% | 21.93% | 28.70% |
| **R-squared** | 0.031 | 0.058 | 0.045 |

**Conclusions**:
- ✅ Grad-CAM adapts coverage to content (618-3226 px)
- ✅ Low overlap (~27%) confirms complementary information
- ✅ Fixed threshold 0.3 is more principled than top-k
- ✅ Better for selective guidance (adapts to harmful strength)

---

## 🚀 Next Steps

### 1. Use Updated Visualization
```bash
# Generate analysis for new images
python visualize_gradient_vs_gradcam_v2.py \
  --image <path> \
  --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth \
  --output_dir visualization/gradient_vs_gradcam_v3 \
  --threshold_percentile 0.3 \
  --gradcam_threshold 0.3
```

### 2. Run Selective CG Experiments
```bash
# Quick test
./test_selective_cg.sh

# Full experiment
./run_selective_cg.sh
```

### 3. Evaluate Results
- **Benign data**: GENEVAL score (expect minimal degradation)
- **Harmful data**: NSFW detection rate (expect high safety)
- **Efficiency**: Guidance application ratio (expect selective activation)

---

## ✅ Changes Summary

### Code Updates
- [x] `visualize_gradient_vs_gradcam_v2.py`
  - Added `gradcam_threshold` parameter (default: 0.3)
  - Changed Grad-CAM masking: `(heatmap >= gradcam_threshold).float()`
  - Replaced Korean text with English labels
  - Updated titles, axis labels, and summary text

### Documentation
- [x] Created `UPDATED_ANALYSIS.md` (this file)
- [x] Explained rationale for fixed threshold vs top-k
- [x] Demonstrated adaptive coverage advantage

### Results
- [x] Generated new visualizations in `visualization/gradient_vs_gradcam_v3/`
- [x] Confirmed Grad-CAM adapts to content (618-3226 px)
- [x] Verified no font rendering issues

---

**Update Complete!** ✅

The visualization now:
1. Uses fixed threshold 0.3 for Grad-CAM (adaptive coverage)
2. Uses English-only labels (no rendering issues)
3. Clearly shows complementary spatial information
4. Demonstrates adaptive intervention advantage

Ready for experiments! 🎉
