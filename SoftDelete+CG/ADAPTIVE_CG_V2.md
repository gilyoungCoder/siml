# Adaptive Soft Spatial CG V2 - 진짜 Adaptive!

## 🎯 핵심 변경사항

### ❌ 제거: Soft Masking (Sigmoid)
- Sigmoid 기반 soft masking 완전 제거
- Temperature parameter 제거
- Gaussian smoothing 제거

### ✅ 추가: 3가지 Adaptive 메커니즘

#### 1. **Adaptive Threshold by Timestep**
```python
class ThresholdScheduler:
    """Threshold를 timestep에 따라 동적으로 조절"""
    strategies = ["constant", "linear_increase", "linear_decrease", "cosine_anneal"]
```

**사용 예:**
- `threshold: 0.7 → 0.3` (초반 엄격 → 후반 관대)
- `threshold: 0.3 → 0.7` (초반 관대 → 후반 엄격)

#### 2. **Adaptive Guidance Scale by Timestep**
```python
class WeightScheduler:
    """Guidance scale을 timestep에 따라 동적으로 조절"""
    # 이미 구현되어 있음 (SpatiallyMaskedGuidance에서 사용)
```

**사용 예:**
- `guidance_scale: 10.0 → 0.5` (초반 강하게 → 후반 약하게)

#### 3. **Spatial-Adaptive Guidance by GradCAM Score**
```python
# Binary mask로 영역 선택
binary_mask = (heatmap > threshold).float()

# Heatmap 값으로 guidance 강도 조절 (pixel-wise)
if use_heatmap_weighted_guidance:
    mask = binary_mask * heatmap
    # heatmap=0.9 → 90% guidance
    # heatmap=0.6 → 60% guidance
else:
    mask = binary_mask  # 100% or 0%
```

## 📊 사용 방법

### 기본 사용 (Binary Mask Only)
```python
from geo_utils.selective_guidance_utils import SelectiveGuidanceMonitor, ThresholdScheduler, WeightScheduler

# Threshold: constant
monitor = SelectiveGuidanceMonitor(
    classifier_model=classifier,
    harmful_threshold=0.5,
    spatial_threshold=0.5,
    use_adaptive_threshold=False,
    use_heatmap_weighted_guidance=False  # Pure binary
)

# Guidance scale: cosine anneal
weight_scheduler = WeightScheduler(
    strategy="cosine_anneal",
    start_value=10.0,
    end_value=0.5,
    total_steps=50
)
```

### Adaptive Threshold
```python
# Threshold scheduler
threshold_scheduler = ThresholdScheduler(
    strategy="linear_decrease",  # 0.7 → 0.3
    start_value=0.7,
    end_value=0.3,
    total_steps=50
)

monitor = SelectiveGuidanceMonitor(
    classifier_model=classifier,
    harmful_threshold=0.5,  # Base (unused if scheduler provided)
    spatial_threshold=0.5,  # Base (unused if scheduler provided)
    use_adaptive_threshold=True,
    threshold_scheduler=threshold_scheduler,
    use_heatmap_weighted_guidance=False
)
```

### Heatmap-Weighted Guidance
```python
monitor = SelectiveGuidanceMonitor(
    classifier_model=classifier,
    harmful_threshold=0.5,
    spatial_threshold=0.5,
    use_adaptive_threshold=False,
    use_heatmap_weighted_guidance=True  # Pixel-wise adaptive!
)
```

### Full Adaptive (모든 기능)
```python
# 1. Adaptive threshold
threshold_scheduler = ThresholdScheduler(
    strategy="cosine_anneal",
    start_value=0.7,
    end_value=0.3,
    total_steps=50
)

# 2. Adaptive guidance scale
weight_scheduler = WeightScheduler(
    strategy="cosine_anneal",
    start_value=10.0,
    end_value=0.5,
    total_steps=50
)

# 3. Heatmap-weighted guidance
monitor = SelectiveGuidanceMonitor(
    classifier_model=classifier,
    harmful_threshold=0.5,
    spatial_threshold=0.5,
    use_adaptive_threshold=True,
    threshold_scheduler=threshold_scheduler,
    use_heatmap_weighted_guidance=True
)

guidance = SpatiallyMaskedGuidance(
    monitor=monitor,
    guidance_scale=5.0,
    harmful_scale=1.5,
    weight_scheduler=weight_scheduler,  # Adaptive scale
    normalize_gradient=True
)
```

## 🆚 Before vs After

### Before (Soft Masking with Sigmoid)
```python
# Threshold: 0.5 고정
# Mask: sigmoid((heatmap - 0.5) / temperature)
# 문제:
#   - Temperature tuning 필요
#   - Multi-concept에서 cross-contamination
#   - 의미가 애매함
```

### After (Binary + Heatmap Weighting)
```python
# Threshold: adaptive (0.7 → 0.3) or fixed (0.5)
# Mask: binary * heatmap (optional)
# 장점:
#   - 명확한 영역 분리
#   - Multi-concept 친화적
#   - Pixel-wise adaptive guidance
#   - 더 직관적
```

## 🎯 실험 권장 설정

### Single Concept (Nudity only)
```python
# Option 1: Binary only
threshold = 0.5  # Fixed
guidance_scale: 10.0 → 0.5  # Cosine anneal
heatmap_weighted = False

# Option 2: Heatmap-weighted
threshold = 0.5  # Fixed
guidance_scale: 10.0 → 0.5  # Cosine anneal
heatmap_weighted = True  # Pixel-wise adaptive
```

### Multi-Concept (Nudity + Violence)
```python
# Binary mask (명확한 영역 분리)
threshold = 0.5  # Fixed
guidance_scale: 10.0 → 0.5  # Cosine anneal
heatmap_weighted = False  # Pure binary

# OR with adaptive threshold
threshold: 0.7 → 0.3  # Adaptive
guidance_scale: 10.0 → 0.5  # Cosine anneal
heatmap_weighted = False
```

## 📝 주요 차이점 정리

| Feature | V1 (Soft Masking) | V2 (Adaptive) |
|---------|-------------------|---------------|
| Mask Type | Sigmoid (continuous) | Binary or Binary*Heatmap |
| Threshold | Fixed | Adaptive by timestep |
| Guidance Scale | Adaptive | Adaptive |
| Spatial Guidance | Uniform in mask region | Pixel-wise adaptive (optional) |
| Multi-concept | Cross-contamination ❌ | Clear separation ✅ |
| Tuning Complexity | Temperature + others | More intuitive |

## ✅ 다음 단계

1. 기존 실험 재실행 (Binary mask)
2. Heatmap-weighted guidance 실험
3. Adaptive threshold 실험
4. Multi-concept 실험

