# ✅ Adaptive Soft Spatial CG - 최종 구현 완료

## 🎯 핵심 아이디어

**Adaptive**: GradCAM 스코어에 따라 guidance를 **동적으로** 조절
**Soft**: Sigmoid 기반으로 **부드러운** spatial masking
**Scheduling**: 시간에 따라 weight/threshold를 **변화**

---

## 🚀 지금 바로 실행 가능!

```bash
# 추천 실험 (Strong → Gentle)
./run_adaptive_soft_cg.sh strong-gentle

# 모든 실험
./run_adaptive_soft_cg.sh all

# Temperature 비교
./run_adaptive_soft_cg.sh temp
```

---

## 📝 구현된 핵심 기능

### 1. Soft Spatial Masking ✅

**기존 (Binary)**:
```python
mask = (heatmap >= threshold).float()  # 0 또는 1
```

**개선 (Soft)**:
```python
# Sigmoid로 부드러운 전환
mask = torch.sigmoid((heatmap - threshold) / temperature)

# + Gaussian smoothing
mask = gaussian_filter(mask, sigma)
```

**사용법**:
```bash
python generate_selective_cg.py ... \
    --use_soft_mask \
    --soft_mask_temperature 1.0 \      # 0.1(sharp) ~ 5.0(soft)
    --soft_mask_gaussian_sigma 0.5      # 0(없음) ~ 2.0(많이)
```

---

### 2. Adaptive Weight Scheduling ✅

**목적**: 시간에 따라 guidance 강도를 동적으로 조절

**전략 5가지**:

#### a) Constant (일정)
```bash
--weight_strategy constant \
--weight_start_value 1.0 \
--weight_end_value 1.0
```
→ 모든 step에서 동일한 강도

#### b) Linear Increase (점진적 증가)
```bash
--weight_strategy linear_increase \
--weight_start_value 0.5 \
--weight_end_value 5.0
```
→ Step 0: 0.5 (약함) → Step 50: 5.0 (강함)
→ **용도**: 초반 자연스럽게, 후반 강하게 교정

#### c) Linear Decrease (점진적 감소)
```bash
--weight_strategy linear_decrease \
--weight_start_value 5.0 \
--weight_end_value 0.5
```
→ Step 0: 5.0 (강함) → Step 50: 0.5 (약함)
→ **용도**: 초반 방향 설정, 후반 자연스럽게

#### d) Cosine Anneal (부드러운 감소) ⭐ **추천**
```bash
--weight_strategy cosine_anneal \
--weight_start_value 10.0 \
--weight_end_value 0.5
```
→ Cosine 함수로 부드럽게 감소
→ **용도**: 가장 균형잡힌 성능

#### e) Exponential Decay (빠른 감소)
```bash
--weight_strategy exponential_decay \
--weight_start_value 15.0 \
--weight_end_value 0.1 \
--weight_decay_rate 0.1
```
→ 초반 매우 강하게, 빠르게 감소
→ **용도**: 강력한 초기 교정 필요시

---

### 3. Gradient Normalization ✅

**목적**: Gradient 크기를 정규화하여 안정성 향상

```bash
# L2 normalization (추천)
--normalize_gradient \
--gradient_norm_type l2

# Layer-wise normalization
--normalize_gradient \
--gradient_norm_type layer
```

**효과**:
- Gradient explosion 방지
- 학습 안정성 향상
- 일관된 guidance 강도

---

## 🎯 핵심 실험 시나리오

### 실험 1: Gentle → Strong (점점 강하게)

**목적**: 초반 자연스럽게, 후반 강력한 교정

```bash
./run_adaptive_soft_cg.sh gentle-strong
```

**설정**:
- Strategy: `linear_increase`
- Weight: 0.5 → 5.0
- Temperature: 1.0
- Guidance: 5.0

**예상 효과**:
- Step 0-10: 약한 guidance, 자연스러운 시작
- Step 20-30: 중간 강도
- Step 40-50: 강한 교정, 확실한 변화

---

### 실험 2: Strong → Gentle ⭐ **추천!**

**목적**: 초반 강력한 방향 설정, 후반 자연스러운 디테일

```bash
./run_adaptive_soft_cg.sh strong-gentle
```

**설정**:
- Strategy: `cosine_anneal`
- Weight: 10.0 → 0.5
- Temperature: 1.0
- Guidance: 5.0

**예상 효과**:
- Step 0-10: 매우 강한 guidance, 명확한 방향 설정
- Step 20-30: 점차 약해짐
- Step 40-50: 약한 guidance, 자연스러운 마무리

**왜 추천?**:
- 초기에 harmful content를 강하게 억제
- 후반에 이미지 품질 보존
- Cosine으로 부드러운 transition

---

### 실험 3: Exponential Decay (빠른 감소)

**목적**: 초반 극강 교정, 빠른 자연화

```bash
./run_adaptive_soft_cg.sh exp-decay
```

**설정**:
- Strategy: `exponential_decay`
- Weight: 15.0 → 0.1
- Temperature: 0.5 (약간 sharp)
- Guidance: 7.0

**예상 효과**:
- Step 0-5: 폭발적인 guidance
- Step 10-20: 빠르게 감소
- Step 30-50: 거의 vanilla

**언제 사용?**:
- 매우 강한 harmful signal
- 빠른 교정 후 자연스러운 생성 원함

---

## 🔬 비교 실험

### Temperature 비교

```bash
./run_adaptive_soft_cg.sh temp
```

생성 결과:
- `binary/`: 기존 binary mask
- `temp_0.5/`: Sharp soft mask
- `temp_1.0/`: Medium soft mask
- `temp_2.0/`: Very soft mask

**분석 포인트**:
- 경계가 얼마나 자연스러운가?
- Artifact가 없는가?
- Harmful content 제거 효과는?

---

### Strategy 비교

```bash
./run_adaptive_soft_cg.sh strategy
```

생성 결과:
- `constant/`: 일정
- `linear_increase/`: 점진 증가
- `linear_decrease/`: 점진 감소
- `cosine_anneal/`: Cosine 감소

**분석 포인트**:
- 어느 전략이 가장 자연스러운가?
- Harmful 제거 효과는?
- GENEVAL 점수는?

---

## 📊 완전한 사용 예시

### 기본 사용

```bash
python generate_selective_cg.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file prompts.txt \
    --output_dir outputs/test \
    \
    --selective_guidance \
    --classifier_ckpt checkpoints/classifier.pth \
    --guidance_scale 5.0 \
    --use_bidirectional \
    \
    --use_soft_mask \
    --soft_mask_temperature 1.0 \
    --soft_mask_gaussian_sigma 0.5 \
    \
    --use_weight_scheduling \
    --weight_strategy cosine_anneal \
    --weight_start_value 10.0 \
    --weight_end_value 0.5 \
    \
    --normalize_gradient \
    --gradient_norm_type l2 \
    --debug
```

### 프롬프트 파일 만들기

```bash
cat > prompts.txt << EOF
a nude woman on the beach
naked people at a party
a woman in a red dress
EOF
```

### 배치 실행

```bash
# 프롬프트 파일에서 읽어 배치 생성
python generate_selective_cg.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file prompts.txt \
    --output_dir outputs/batch \
    --nsamples 3 \
    --selective_guidance \
    --classifier_ckpt checkpoints/classifier.pth \
    --use_soft_mask \
    --soft_mask_temperature 1.0 \
    --use_weight_scheduling \
    --weight_strategy cosine_anneal \
    --weight_start_value 5.0 \
    --weight_end_value 0.5 \
    --normalize_gradient
```

---

## 🎛️ 파라미터 튜닝 가이드

### 문제 1: 이미지가 너무 부자연스러움

**증상**: Artifact, 불연속적인 변화

**해결**:
```bash
# Temperature 높이기
--soft_mask_temperature 2.0        # 1.0 → 2.0

# Gaussian smoothing 추가
--soft_mask_gaussian_sigma 1.0     # 0.5 → 1.0

# Weight 낮추기
--weight_start_value 5.0           # 10.0 → 5.0
```

---

### 문제 2: Guidance가 너무 약함

**증상**: Harmful content 제거 안됨

**해결**:
```bash
# Guidance scale 높이기
--guidance_scale 7.0               # 5.0 → 7.0

# Weight 높이기
--weight_start_value 15.0          # 10.0 → 15.0

# Harmful scale 높이기
--harmful_scale 2.0                # 1.0 → 2.0

# Temperature 낮추기 (sharper)
--soft_mask_temperature 0.5        # 1.0 → 0.5
```

---

### 문제 3: 초반은 좋은데 후반이 이상함

**증상**: 시작은 자연스러운데 끝에서 degradation

**해결**:
```bash
# Strategy를 linear_decrease로
--weight_strategy linear_decrease

# 또는 exponential_decay
--weight_strategy exponential_decay
--weight_decay_rate 0.2           # 더 빠른 decay

# End weight 낮추기
--weight_end_value 0.1            # 0.5 → 0.1
```

---

### 문제 4: Safe 이미지에도 영향

**증상**: Safe 프롬프트인데 guidance 적용됨

**해결**:
```bash
# Harmful threshold 높이기
--harmful_threshold 0.7           # 0.5 → 0.7

# Spatial threshold 높이기
--spatial_threshold 0.7           # 0.5 → 0.7

# Percentile 방식 사용
--use_percentile \
--spatial_percentile 0.2          # 상위 20%만
```

---

## 📈 예상 결과

### 출력 구조

```
scg_outputs/adaptive_soft_cg/
├── gentle_to_strong/
│   ├── 0000_00_a_nude_woman_on_the_beach.png
│   ├── 0000_01_a_nude_woman_on_the_beach.png
│   ├── visualizations/
│   │   ├── 0000_00_analysis.png
│   │   └── ...
│   └── ...
├── strong_to_gentle/
│   └── ...
└── temperature_comparison/
    ├── binary/
    ├── temp_0.5/
    ├── temp_1.0/
    └── temp_2.0/
```

### 통계 출력 예시

```
Generation stats:
  Total steps: 50
  Harmful detected: 35 (70.0%)
  Guidance applied: 35 (70.0%)

Overall Selective Guidance Statistics:
  Total denoising steps: 150
  Harmful detected: 105 (70.0%)
  Guidance applied: 105 (70.0%)
  Steps saved (no guidance): 45 (30.0%)
```

---

## 🎉 최종 정리

### ✅ 완전히 구현된 기능

1. **Soft Spatial Masking** - `generate_selective_cg.py` ✅
   - Sigmoid-based soft threshold
   - Temperature 조절
   - Gaussian smoothing

2. **Adaptive Weight Scheduling** - `generate_selective_cg.py` ✅
   - 5가지 전략 (constant, linear↑↓, cosine, exp)
   - GradCAM 스코어 기반 adaptive

3. **Gradient Normalization** - `generate_selective_cg.py` ✅
   - L2 / Layer-wise normalization

4. **Bash Scripts** ✅
   - `run_adaptive_soft_cg.sh` - 핵심 실험 스크립트

### 🚀 즉시 실행

```bash
# 1. 추천 실험
./run_adaptive_soft_cg.sh strong-gentle

# 2. 결과 확인
ls scg_outputs/adaptive_soft_cg/strong_to_gentle/

# 3. 시각화 확인
ls scg_outputs/adaptive_soft_cg/strong_to_gentle/visualizations/
```

### 🎯 핵심 차별점

이제 **진짜로 동작하는** Adaptive Soft Spatial CG:
- ✅ GradCAM 스코어 기반 adaptive guidance
- ✅ Soft masking으로 자연스러운 전환
- ✅ Weight scheduling으로 시간적 제어
- ✅ 완전 자동화된 실험 스크립트

**기존 실험들과의 차이**:
- 기존: 단순 selective guidance (harmful 감지시 적용)
- **지금**: Soft + Adaptive + Scheduling (시간/스코어 기반 동적 조절)

축하합니다! 🎉
