# Absolute GradCAM Normalization Guide

## 📌 문제점: 기존 Per-Image Normalization

### 기존 방식
```python
# 각 이미지마다 독립적으로 Min-Max normalize
normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
# 결과: 항상 max=1.0, min=0.0
```

### 문제
1. **절대적 크기 정보 손실**: 유해 신호가 약해져도 max=1.0 유지
2. **이미지 간 비교 불가**: 강한 유해 이미지와 약한 유해 이미지가 동일하게 normalize
3. **Guidance 효과 확인 어려움**: 히트맵이 감소해도 normalize 후 항상 max=1.0

---

## ✅ 해결: Training Data 기반 Absolute Normalization

### 새로운 방식
```python
# Training data에서 계산한 통계 사용
z_score = (heatmap - training_mean) / training_std
probability = gaussian_cdf(z_score)  # 0~1 확률값
```

### 장점
1. **절대적 기준**: Training data의 유해도 분포 기준
2. **이미지 간 비교 가능**: 확률값이 절대적 의미를 가짐
3. **Guidance 효과 추적**: 히트맵이 감소하면 확률도 감소

---

## 🔧 사용 방법

### Step 1: Training Data에서 통계 계산

```bash
python compute_gradcam_statistics.py \
    --data_dir ./data/harmful_training_images \
    --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth \
    --output_file ./gradcam_stats.json \
    --num_samples 1000
```

**출력 예시**:
```json
{
  "mean": 3.452183,
  "std": 2.841056,
  "min": 0.000124,
  "max": 45.234512,
  "median": 2.834512,
  "percentiles": {
    "p10": 0.523412,
    "p25": 1.234512,
    "p50": 2.834512,
    "p75": 4.523412,
    "p90": 7.234512,
    "p95": 9.523412,
    "p99": 15.234512
  },
  "num_images": 1000,
  "num_values": 4096000
}
```

---

### Step 2: Inference에서 통계 사용

```bash
python generate_always_adaptive_spatial_cg.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file prompts.txt \
    --classifier_ckpt ./work_dirs/classifier.pth \
    --gradcam_stats_file ./gradcam_stats.json \  # ← 중요!
    --guidance_scale 15.0 \
    --spatial_threshold_start 0.7 \
    --spatial_threshold_end 0.3 \
    --use_bidirectional
```

---

## 📊 Threshold 설정 가이드

### Gaussian CDF 확률 해석

```
Z-score | CDF (Probability) | 의미
--------|------------------|------------------
-2.0    | 0.023            | 매우 약한 유해 신호
-1.0    | 0.159            | 약한 유해 신호
 0.0    | 0.500            | 평균 (training data 기준)
+1.0    | 0.841            | 강한 유해 신호
+2.0    | 0.977            | 매우 강한 유해 신호
+3.0    | 0.999            | 극도로 강한 유해 신호
```

### Threshold 예시

#### 보수적 (강한 유해만 제거)
```bash
--spatial_threshold_start 0.84  # z-score > +1.0
--spatial_threshold_end 0.98    # z-score > +2.0
```
- 평균보다 1σ 이상 높은 영역만 마스킹
- FID 성능 우선, 안전성 보통

#### 균형잡힌 (권장)
```bash
--spatial_threshold_start 0.7   # z-score > +0.5
--spatial_threshold_end 0.5     # z-score > 0.0
```
- 평균 이상 영역 마스킹
- FID와 안전성 균형

#### 공격적 (약한 유해도 제거)
```bash
--spatial_threshold_start 0.5   # z-score > 0.0
--spatial_threshold_end 0.16    # z-score > -1.0
```
- 평균 미만 영역도 마스킹
- 안전성 우선, FID 저하 가능

---

## 🔬 실험: 기존 vs 새 방식

### 시나리오: 유해 이미지 생성 중 가이던스 적용

#### 기존 Per-Image Normalization
```
Step 0:
  Raw heatmap: [0.1, 5.0, 10.0, 20.0]
  Normalized:  [0.0, 0.24, 0.49, 1.0]  → max=1.0
  Threshold: 0.5 → Mask ratio: 50%

Step 49 (가이던스 적용 후):
  Raw heatmap: [0.05, 1.0, 2.0, 5.0]  ← 크게 감소!
  Normalized:  [0.0, 0.19, 0.39, 1.0]  → 여전히 max=1.0 ❌
  Threshold: 0.1 → Mask ratio: 100%  ← 오히려 증가!
```

**문제**: Raw 값은 감소했지만 normalize 때문에 효과가 보이지 않음

#### 새로운 Absolute Normalization (mean=3.45, std=2.84)
```
Step 0:
  Raw heatmap: [0.1, 5.0, 10.0, 20.0]
  Z-scores:    [-1.18, 0.55, 2.31, 5.83]
  CDF probs:   [0.12, 0.71, 0.99, 1.0]
  Threshold: 0.5 → Mask ratio: 75%

Step 49 (가이던스 적용 후):
  Raw heatmap: [0.05, 1.0, 2.0, 5.0]  ← 감소
  Z-scores:    [-1.19, -0.86, -0.51, 0.55]
  CDF probs:   [0.12, 0.19, 0.30, 0.71]  ← 확률도 감소 ✅
  Threshold: 0.3 → Mask ratio: 50%  ← 감소!
```

**효과**: Raw 값 감소가 확률값에 반영됨 → Mask ratio 감소

---

## 💡 주의사항

### 1. Training Data 선택
- **반드시 Harmful 이미지만** 사용
- Normal 이미지 포함하면 평균이 낮아져서 기준이 왜곡됨
- 충분한 샘플 (최소 500개 이상 권장)

### 2. Timestep 일관성
- 통계 계산 시 timestep = 500 (중간값) 사용
- Inference 시에도 동일한 timestep에서 비교 가능

### 3. Threshold 조정
- 처음엔 보수적으로 시작 (0.7~0.9)
- 결과 확인 후 점진적으로 낮춤
- FID와 안전성 trade-off 고려

---

## 🎯 Quick Start

### 1. 통계 계산 (1회만)
```bash
python compute_gradcam_statistics.py \
    --data_dir /path/to/harmful_images \
    --classifier_ckpt ./work_dirs/classifier.pth \
    --output_file ./gradcam_stats.json \
    --num_samples 1000
```

### 2. Inference (통계 파일 사용)
```bash
./run_always_adaptive_spatial_cg.sh
```

[run_always_adaptive_spatial_cg.sh](run_always_adaptive_spatial_cg.sh)에 다음 추가:
```bash
GRADCAM_STATS_FILE="./gradcam_stats.json"

# 명령어에 추가
python generate_always_adaptive_spatial_cg.py \
    ... \
    --gradcam_stats_file $GRADCAM_STATS_FILE \
    ...
```

---

## 📈 예상 결과

### 정상 프롬프트
```
Heatmap mean (normalized): 0.2~0.3
→ Training mean(3.45) 보다 훨씬 낮음
→ Z-score: 음수
→ CDF probability: <0.2
→ Threshold(0.5) 이하 → 마스킹 거의 없음 ✅
```

### 유해 프롬프트
```
Heatmap mean (normalized): 0.5~0.8
→ Training mean(3.45) 근처 또는 높음
→ Z-score: 양수
→ CDF probability: 0.5~0.9
→ Threshold 이상 → 강한 마스킹 ✅
```

### Guidance 효과
```
Step 0:   CDF prob = 0.8 → Mask 80%
Step 25:  CDF prob = 0.6 → Mask 60% (감소!)
Step 49:  CDF prob = 0.4 → Mask 40% (더 감소!)
```

**핵심**: 이제 Mask ratio가 실제로 감소합니다!

---

## 🔗 관련 파일

- [compute_gradcam_statistics.py](compute_gradcam_statistics.py) - 통계 계산 스크립트
- [generate_always_adaptive_spatial_cg.py](generate_always_adaptive_spatial_cg.py) - 메인 생성 스크립트 (수정됨)
- [run_always_adaptive_spatial_cg.sh](run_always_adaptive_spatial_cg.sh) - 실행 스크립트

---

## 📝 결론

**Absolute Normalization = 진짜 유해한 정도를 측정**

- ✅ Training data 기준 절대적 유해도
- ✅ 이미지 간 비교 가능
- ✅ Guidance 효과 정량화 가능
- ✅ FID 성능과 안전성 균형 조절 가능

이제 정말로 **"유해한 부분만 선택적으로 제거"**할 수 있습니다!
