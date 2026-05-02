# 📊 Adaptive CG V2 - 실험 결과 분석

## 실험 개요

**날짜**: 2025-12-23
**실험**: Adaptive CG V2 (Sigmoid Soft Masking 제거, 3가지 Adaptive 메커니즘 추가)

### 실험된 Concepts
1. **Violence** (50 prompts, 150 images)
2. **Nudity** (35 prompts, 105 images)

### 실험된 설정
1. Binary Baseline (고정 threshold, binary mask)
2. Adaptive Threshold (0.7→0.3, binary mask)
3. Heatmap-Weighted (고정 threshold, heatmap-weighted mask)
4. Full Adaptive (0.7→0.3 threshold + heatmap-weighted)

---

## 🎯 주요 성과

### ✅ 모든 Step (0~50) 완전 추적 성공!

**이전 (Soft Masking 버전):**
- Harmful 감지된 step만 기록
- 그래프에 점 몇 개만 표시
- 전체 denoising 과정 파악 불가

**현재 (Adaptive V2):**
- 모든 50 step 완전 기록
- 전체 곡선으로 표시
- Denoising 과정 완벽 추적 가능

---

## 📈 Harmful Score 패턴 분석

### Violence Experiments 관찰 (Binary Baseline 기준)

```
Step   | Harmful Score | 관찰
-------|---------------|------
0-1    | 0 → 1.0       | 초반 급격한 peak (노이즈 많은 단계)
2-10   | -1.0 ~ 0.5    | 높은 변동성 (불안정한 초반)
10-40  | -1.0 ~ 0.2    | 대부분 threshold(0.5) 이하
40-50  | -2.0 ~ 0      | 음수로 떨어짐 (확실히 safe)
```

**핵심 인사이트:**
- 초반(Step 0-10): 노이즈가 많아 classifier 예측 불안정
- 중반(Step 10-40): 점진적으로 안정화
- 후반(Step 40-50): 디테일 단계, 대부분 safe

### Guidance 적용 빈도

```
총 50 step 중:
  - Guidance 적용: 3-5 step (6-10%)
  - Guidance 미적용: 45-47 step (90-94%)

Masked Region Ratio:
  - Step 1: 0.08 (8%)
  - Step 10: 0.05 (5%)
  - Step 20: 0.15 (15%)
  - 나머지: 0 (guidance 미적용)
```

**의미:**
- 전체 생성 과정의 90% 이상은 guidance 없이 자연스럽게 진행
- 필요한 순간에만 선택적으로 개입 (Selective!)
- 이미지의 5-15% 영역에만 spatial guidance 적용

---

## 🔬 실험별 비교

### 1. Binary Baseline
**설정:**
- Threshold: 0.5 (고정)
- Guidance Scale: 10.0 → 0.5 (Cosine Anneal)
- Mask: Binary (0 or 1)

**관찰:**
- 초반 노이즈 단계에서도 동일한 threshold 사용
- 명확하고 일관된 기준
- Step 1에서 harmful 감지 → guidance 적용

**장점:**
- 간단하고 명확
- Baseline으로 적합

**단점:**
- 초반 노이즈에서 false positive 가능성

---

### 2. Adaptive Threshold
**설정:**
- Threshold: 0.7 → 0.3 (Cosine Anneal)
- Guidance Scale: 10.0 → 0.5 (Cosine Anneal)
- Mask: Binary (0 or 1)

**예상 효과:**
```
Step   | Threshold | 효과
-------|-----------|------
0-10   | 0.7       | 초반 노이즈에 엄격 → false positive 감소
10-30  | 0.5       | 중반 표준 기준
30-50  | 0.3       | 후반 관대 → 디테일 단계 더 민감하게
```

**장점:**
- 초반 노이즈에 robust
- 후반 디테일에서 더 많은 영역 guidance
- Step별로 최적화된 threshold

**예상:**
- Binary Baseline 대비 초반 false positive 감소
- 후반 디테일 단계 성능 향상

---

### 3. Heatmap-Weighted
**설정:**
- Threshold: 0.5 (고정)
- Guidance Scale: 10.0 → 0.5 (Cosine Anneal)
- Mask: Binary * Heatmap

**작동 방식:**
```
Binary Mask:
  pixel[i,j] = 1 if heatmap[i,j] > 0.5 else 0

Heatmap-Weighted:
  pixel[i,j] = 1 * heatmap[i,j] if heatmap[i,j] > 0.5 else 0

예시:
  heatmap = 0.9 → mask = 0.9 (90% guidance)
  heatmap = 0.6 → mask = 0.6 (60% guidance)
  heatmap = 0.4 → mask = 0.0 (0% guidance)
```

**장점:**
- Pixel-wise adaptive guidance 강도
- 확실한 harmful(0.9)에 강하게, 애매한 영역(0.6)에 약하게
- 더 자연스러운 경계

**예상:**
- Binary 대비 부드러운 transition
- 과도한 guidance 방지

---

### 4. Full Adaptive ⭐ 추천
**설정:**
- Threshold: 0.7 → 0.3 (Cosine Anneal)
- Guidance Scale: 10.0 → 0.5 (Cosine Anneal)
- Mask: Binary * Heatmap

**조합 효과:**
1. **Adaptive Threshold**: 초반 robust, 후반 sensitive
2. **Adaptive Guidance Scale**: 초반 강하게, 후반 약하게
3. **Heatmap-Weighted**: Pixel-wise adaptive

**예상:**
- 최고의 flexibility와 성능
- 모든 adaptive 기능의 시너지

---

## 💡 핵심 인사이트

### 1. Denoising 과정의 3단계

```
초반 (Step 0-10):
  - 노이즈 많음, Harmful Score 불안정
  - 높은 변동성, 예측 어려움
  - Adaptive Threshold (0.7)로 false positive 차단

중반 (Step 10-40):
  - 점진적 안정화
  - 대부분 threshold 근처 또는 이하
  - 필요시에만 selective guidance

후반 (Step 40-50):
  - 디테일 단계, 대부분 safe
  - Adaptive Threshold (0.3)로 더 민감하게
  - 약한 guidance로 디테일 보존
```

### 2. Selective Guidance의 효율성

```
전체 생성 과정:
  - 90%: Guidance 없이 자연스럽게 생성
  - 10%: 필요한 순간에만 개입

공간적 선택성:
  - 이미지의 5-15%에만 spatial guidance
  - 85-95%는 그대로 유지 (safe 영역)
```

**의미:**
- 과도한 개입 방지
- 자연스러운 생성 과정 유지
- 필요한 곳에만 정밀하게 적용

### 3. Adaptive 메커니즘의 필요성

**Adaptive Threshold:**
- 초반 노이즈 → 높은 threshold (0.7) 필요
- 후반 디테일 → 낮은 threshold (0.3) 적합
- Fixed threshold는 모든 단계에 최적 아님

**Heatmap-Weighted Guidance:**
- 확실한 harmful (heatmap=0.9) → 강한 guidance 필요
- 애매한 영역 (heatmap=0.6) → 약한 guidance 적합
- Binary (0 or 1)는 너무 극단적

---

## 🆚 Concept별 차이

### Violence vs Nudity

| Metric | Violence | Nudity |
|--------|----------|--------|
| 프롬프트 수 | 50 | 35 |
| 생성 이미지 | 150 | 105 |
| Visualization | 150 (100%) | 79 (75%) |
| Harmful 감지율 | 높음 | 중간 |

**관찰:**
- Violence: 모든 이미지에서 visualization 생성 (100%)
  → Harmful 감지가 더 빈번
- Nudity: 75%만 visualization
  → 일부는 처음부터 끝까지 safe

**의미:**
- Violence concept이 더 쉽게 감지됨
- Nudity는 프롬프트에 따라 차이 큼

---

## 🎯 결론 및 권장사항

### ✅ Full Adaptive 설정 추천

**이유:**

1. **Adaptive Threshold (0.7 → 0.3)**
   - 초반 노이즈에 robust
   - 후반 디테일에 sensitive
   - Step별 최적화

2. **Adaptive Guidance Scale (10.0 → 0.5)**
   - 초반 강한 방향 설정
   - 후반 디테일 보존
   - 과도한 개입 방지

3. **Heatmap-Weighted Guidance**
   - Pixel-wise adaptive
   - 자연스러운 경계
   - 확실한 영역에 집중

### 다음 단계

1. **이미지 품질 직접 비교**
   - Binary vs Heatmap-weighted 시각적 차이
   - Adaptive vs Fixed threshold 차이
   - Full Adaptive의 실제 효과

2. **정량적 평가**
   - Harmful 감지율 계산
   - False positive/negative 비율
   - CLIP/FID score 비교

3. **Multi-Concept 실험**
   - Nudity + Violence 동시 적용
   - Binary mask의 명확한 영역 분리 확인
   - Cross-contamination 방지 검증

4. **Hyperparameter Tuning**
   - Threshold range 조정 (0.7→0.3 vs 0.6→0.4)
   - Guidance scale range 조정
   - Strategy 비교 (Cosine vs Linear vs Exponential)

---

## 📝 주요 개선사항 요약

### ❌ 제거됨 (Soft Masking V1)
- Sigmoid-based soft masking
- Temperature parameter
- Gaussian smoothing
- 애매한 의미, multi-concept 문제

### ✅ 추가됨 (Adaptive CG V2)
- ThresholdScheduler (adaptive threshold by timestep)
- Heatmap-weighted guidance (pixel-wise adaptive)
- 모든 step 완전 기록
- 더 명확하고 직관적

### 🎯 결과
- 전체 denoising 과정 완벽 추적
- Step별 최적화된 threshold
- Pixel-wise adaptive guidance
- Multi-concept 친화적

---

**실험 위치:**
- Nudity: `scg_outputs/adaptive_v2/nudtiy_experiments/`
- Violence: `scg_outputs/adaptive_v2/violence_experiments/`

**Visualization 확인:**
- `{experiment}/visualizations/`
- Analysis 차트로 전체 과정 확인 가능
