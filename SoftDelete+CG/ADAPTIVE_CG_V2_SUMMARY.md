# ✅ Adaptive CG V2 구현 완료!

## 🎯 구현된 3가지 Adaptive 메커니즘

### 1. **Adaptive Threshold by Timestep**
```python
threshold_scheduler = ThresholdScheduler(
    strategy="cosine_anneal",
    start_value=0.7,  # 초반: 엄격
    end_value=0.3,    # 후반: 관대
    total_steps=50
)
```
- **효과**: 초반에는 높은 threshold로 확실한 harmful만 감지
- **장점**: 노이즈 많은 초반 단계에서 false positive 감소

### 2. **Adaptive Guidance Scale by Timestep**
```python
weight_scheduler = WeightScheduler(
    strategy="cosine_anneal",
    start_value=10.0,  # 초반: 강하게
    end_value=0.5,     # 후반: 약하게
    total_steps=50
)
```
- **효과**: 초반 강한 guidance로 방향 설정, 후반 약하게 디테일 보존
- **장점**: 이미 구현되어 있던 기능 활용

### 3. **Spatial-Adaptive Guidance by GradCAM Score**
```python
# Binary mask로 영역 선택
binary_mask = (heatmap > threshold).float()

# Heatmap 값으로 pixel-wise guidance 강도 조절
if use_heatmap_weighted_guidance:
    mask = binary_mask * heatmap
    # heatmap=0.9 → 90% guidance
    # heatmap=0.6 → 60% guidance
else:
    mask = binary_mask  # 0 or 1
```
- **효과**: GradCAM score가 높은 픽셀에 더 강한 guidance
- **장점**: 확실한 harmful 영역에 집중, 애매한 영역은 약하게

## ❌ 제거된 기능

### Soft Masking (Sigmoid)
```python
# 이전 (제거됨)
mask = sigmoid((heatmap - threshold) / temperature)
```
- **제거 이유**:
  1. Multi-concept에서 cross-contamination
  2. Temperature tuning 복잡
  3. 의미 애매함

## 🚀 사용 방법

### 실행 스크립트
```bash
./run_adaptive_cg_v2.sh all              # 모든 실험
./run_adaptive_cg_v2.sh binary           # Binary baseline
./run_adaptive_cg_v2.sh adaptive-threshold  # Adaptive threshold
./run_adaptive_cg_v2.sh heatmap          # Heatmap-weighted
./run_adaptive_cg_v2.sh full             # Full adaptive
```

### 4가지 실험 비교

| 실험 | Threshold | Guidance Scale | Mask Type |
|------|-----------|----------------|-----------|
| **Binary Baseline** | 0.5 (고정) | 10.0→0.5 (adaptive) | Binary (0 or 1) |
| **Adaptive Threshold** | 0.7→0.3 (adaptive) | 10.0→0.5 (adaptive) | Binary |
| **Heatmap-Weighted** | 0.5 (고정) | 10.0→0.5 (adaptive) | Binary * Heatmap |
| **Full Adaptive** | 0.7→0.3 (adaptive) | 10.0→0.5 (adaptive) | Binary * Heatmap |

## 📊 예상 결과

### Binary Baseline
- 명확한 영역 분리
- Multi-concept 친화적
- 하지만 모든 픽셀에 동일한 강도

### Adaptive Threshold
- 초반 엄격 (0.7) → 후반 관대 (0.3)
- False positive 감소 기대
- 후반 디테일 단계에서 더 많은 영역 guidance

### Heatmap-Weighted
- Pixel-wise adaptive guidance
- 확실한 harmful (0.9) → 90% 강도
- 애매한 영역 (0.6) → 60% 강도
- 더 자연스러운 결과 기대

### Full Adaptive
- 모든 adaptive 기능 활성화
- 최고의 성능 기대
- 초반: 엄격한 threshold + 강한 guidance
- 후반: 관대한 threshold + 약한 guidance + pixel-wise adaptive

## 🔍 분석 포인트

생성 후 확인할 사항:

1. **Visualization 분석**
   - `visualizations/` 폴더의 analysis.png 확인
   - Threshold 변화 확인 (adaptive threshold 실험)
   - Masked region ratio 변화 확인

2. **이미지 품질 비교**
   - Binary vs Heatmap-weighted
   - Fixed threshold vs Adaptive threshold
   - 경계 자연스러움, 디테일 보존 확인

3. **Harmful 감지율**
   - 각 실험별로 guidance 적용 비율
   - False positive/negative 비교

## 💡 다음 단계

1. **실험 실행**
   ```bash
   ./run_adaptive_cg_v2.sh all
   ```

2. **결과 분석**
   - 각 실험의 visualization 확인
   - 이미지 품질 비교
   - Guidance 통계 확인

3. **최적 설정 찾기**
   - Threshold range 조정 (0.7→0.3 vs 0.6→0.4)
   - Guidance scale range 조정
   - Heatmap weighting 효과 확인

4. **Multi-concept 실험**
   - Nudity + Violence
   - Binary mask로 명확한 영역 분리 확인

## 📝 주요 파일

- `geo_utils/selective_guidance_utils.py`: 핵심 구현
  - `ThresholdScheduler` 클래스
  - `SelectiveGuidanceMonitor` 업데이트
  - Binary + Heatmap weighting 로직

- `generate_selective_cg.py`: Arguments 추가
  - `--use_adaptive_threshold`
  - `--threshold_strategy`, `--threshold_start_value`, `--threshold_end_value`
  - `--use_heatmap_weighted_guidance`

- `run_adaptive_cg_v2.sh`: 실험 스크립트
  - 4가지 실험 자동 실행
  - sexual_50.txt 사용

- `ADAPTIVE_CG_V2.md`: 상세 문서
  - 사용법, 예시 코드
  - Before/After 비교

## ✅ 체크리스트

- [x] ThresholdScheduler 구현
- [x] Adaptive threshold 로직 추가
- [x] Heatmap-weighted guidance 구현
- [x] Soft masking 제거
- [x] generate_selective_cg.py arguments 추가
- [x] Monitor 초기화 업데이트
- [x] 실험 스크립트 작성
- [x] 문서화

이제 실행만 하면 됩니다! 🚀

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
./run_adaptive_cg_v2.sh all
```
