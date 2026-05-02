# 하이퍼파라미터 비교: Selective CG vs Always-On Adaptive Spatial CG

## 📊 전체 비교표

| 카테고리 | 기존 Selective CG | 새 버전 Always-On Adaptive Spatial CG | 변화 |
|---------|------------------|--------------------------------------|------|
| **총 하이퍼파라미터 수** | **26개** | **9개** | **-17개 (65% 감소)** |
| **필수 파라미터** | 12개 | 4개 | -8개 |
| **선택 파라미터** | 14개 | 5개 | -9개 |

---

## 🔴 기존 Selective CG 하이퍼파라미터 (26개)

### 1️⃣ **Harmful Detection 관련** (5개) ❌ 제거됨
```bash
--harmful_threshold          # 유해 콘텐츠 감지 임계값 (default: 0.5)
--harmful_class              # 유해 클래스 인덱스 (default: 2)
--safe_class                 # 안전 클래스 인덱스 (default: 1)
--use_adaptive_threshold     # 적응형 감지 임계값 활성화
--threshold_start_value      # 감지 임계값 시작값
--threshold_end_value        # 감지 임계값 종료값
```
**문제점**: 감지 단계 자체가 불필요하게 복잡도를 증가시킴

---

### 2️⃣ **Spatial Masking 관련** (7개) → **4개로 단순화**
```bash
# 기존 (복잡)
--spatial_threshold          # 고정 공간 임계값 (default: 0.5)
--use_percentile            # 백분위수 사용 여부 ❌ 제거
--spatial_percentile        # 백분위수 값 (default: 0.3) ❌ 제거
--threshold_strategy        # 임계값 스케줄링 전략 ✅ 유지
--spatial_threshold_start   # 공간 임계값 시작 ❌ 이름 변경
--spatial_threshold_end     # 공간 임계값 종료 ❌ 이름 변경
--gradcam_layer            # Grad-CAM 타겟 레이어 ✅ 유지

# 새 버전 (단순)
--spatial_threshold_start   # 초기 공간 임계값 (0.7) ✅
--spatial_threshold_end     # 최종 공간 임계값 (0.3) ✅
--threshold_strategy        # linear_decrease, cosine 등 ✅
--gradcam_layer            # Grad-CAM 레이어 ✅
```

---

### 3️⃣ **Guidance Scale 관련** (4개) → **2개로 단순화**
```bash
# 기존
--guidance_scale            # 가이던스 강도 ✅ 유지
--use_bidirectional        # 양방향 가이던스 ✅ 유지
--harmful_scale            # 유해 반발력 강도 ✅ 유지
--guidance_start_step      # 가이던스 시작 스텝 ✅ 유지
--guidance_end_step        # 가이던스 종료 스텝 ✅ 유지

# Weight Scheduling (제거됨)
--use_weight_scheduling    # ❌ 제거
--weight_strategy          # ❌ 제거
--weight_start_value       # ❌ 제거
--weight_end_value         # ❌ 제거
--weight_decay_rate        # ❌ 제거
```
**이유**: Adaptive spatial threshold만으로 충분, weight scheduling은 중복

---

### 4️⃣ **General CG (항상 적용 가이던스)** (5개) ❌ 완전 제거
```bash
--general_cg                      # ❌ 제거 (항상 적용이 기본이 됨)
--general_cg_scale                # ❌ 제거
--general_cg_harmful_scale        # ❌ 제거
--general_cg_use_bidirectional   # ❌ 제거
--general_cg_start_step          # ❌ 제거
--general_cg_end_step            # ❌ 제거
```
**이유**: 새 버전에서는 항상 적용이 기본 동작

---

### 5️⃣ **Adaptive CG V2 고급 기능** (5개) ❌ 제거
```bash
--use_heatmap_weighted_guidance  # ❌ 제거 (히트맵 가중 가이던스)
--normalize_gradient             # ❌ 제거 (그래디언트 정규화)
--gradient_norm_type            # ❌ 제거 (l2 or layer)
```
**이유**: 실험적 기능으로 복잡도만 증가

---

## 🟢 새 버전: Always-On Adaptive Spatial CG (9개)

### ✅ **핵심 파라미터 (4개)**
```bash
# 1. 가이던스 강도
--guidance_scale 5.0
  # 설명: 분류기 가이던스의 강도
  # 권장값: 3.0~10.0
  # 높을수록: 강한 수정 (과도하면 품질 저하)
  # 낮을수록: 약한 수정

# 2. 초기 공간 임계값 (Early steps)
--spatial_threshold_start 0.7
  # 설명: 초기 스텝의 Grad-CAM 임계값
  # 권장값: 0.6~0.8
  # 높을수록: 적은 영역 마스킹 (넓은 가이던스)
  # 낮을수록: 많은 영역 마스킹

# 3. 최종 공간 임계값 (Late steps)
--spatial_threshold_end 0.3
  # 설명: 후기 스텝의 Grad-CAM 임계값
  # 권장값: 0.2~0.4
  # 높을수록: 적은 영역 마스킹
  # 낮을수록: 많은 영역 마스킹 (세밀한 수정)

# 4. 임계값 스케줄링 전략
--threshold_strategy linear_decrease
  # 선택지:
  #   - linear_decrease: 선형 감소 (권장)
  #   - cosine_anneal: 코사인 감소 (부드러운 전환)
  #   - constant: 고정값
  #   - linear_increase: 선형 증가
```

### 📌 **선택적 파라미터 (5개)**
```bash
# 5. 양방향 가이던스 (옵션)
--use_bidirectional
  # 설명: 안전 방향으로 당기기 + 유해 방향에서 밀기
  # 권장: 활성화 (더 강력한 효과)

# 6. 유해 반발력 강도 (양방향일 때만)
--harmful_scale 1.0
  # 설명: 유해 방향 반발력 상대 강도
  # 권장값: 0.5~2.0
  # 1.0 = 균형 (pull = push)

# 7. 가이던스 시작 스텝
--guidance_start_step 0
  # 설명: 몇 번째 스텝부터 가이던스 적용
  # 권장값: 0 (처음부터)

# 8. 가이던스 종료 스텝
--guidance_end_step 50
  # 설명: 몇 번째 스텝까지 가이던스 적용
  # 권장값: 50 (끝까지)

# 9. Grad-CAM 타겟 레이어
--gradcam_layer "encoder_model.middle_block.2"
  # 설명: Grad-CAM 계산에 사용할 분류기 레이어
  # 권장: 기본값 유지
```

---

## 🎯 핵심 차이점 요약

### 기존 Selective CG의 이중 임계값 구조
```
Step 1: harmful_threshold로 감지
  ↓ (harmful detected?)
Step 2: spatial_threshold로 마스킹
  ↓
Step 3: 가이던스 적용
```

### 새 버전의 단일 임계값 구조
```
Step 1: adaptive spatial_threshold로 마스킹 (감지 없음)
  ↓
Step 2: 가이던스 항상 적용
```

---

## 💡 사용 예시

### 기본 사용 (최소 설정)
```bash
python generate_always_adaptive_spatial_cg.py \
    path/to/sd_model \
    --prompt_file prompts.txt \
    --guidance_scale 5.0 \
    --spatial_threshold_start 0.7 \
    --spatial_threshold_end 0.3 \
    --threshold_strategy linear_decrease
```

### 고급 사용 (양방향 + 커스텀 범위)
```bash
python generate_always_adaptive_spatial_cg.py \
    path/to/sd_model \
    --prompt_file prompts.txt \
    --guidance_scale 7.0 \
    --spatial_threshold_start 0.8 \
    --spatial_threshold_end 0.2 \
    --threshold_strategy cosine_anneal \
    --use_bidirectional \
    --harmful_scale 1.5 \
    --guidance_start_step 5 \
    --guidance_end_step 45
```

---

## 📈 장점 분석

| 측면 | 기존 | 새 버전 | 개선 |
|-----|------|--------|------|
| **하이퍼파라미터 수** | 26개 | 9개 | **-65%** |
| **필수 조정 파라미터** | 12개 | 4개 | **-67%** |
| **개념적 복잡도** | 높음 (감지+마스킹) | 낮음 (마스킹만) | **단순화** |
| **조정 용이성** | 어려움 | 쉬움 | **직관적** |
| **계산 오버헤드** | 높음 (감지 단계) | 낮음 (감지 없음) | **효율적** |
| **일관성** | 프롬프트마다 다름 | 모든 프롬프트 동일 | **예측 가능** |

---

## 🔬 실험 권장 사항

### 1단계: 기본 설정으로 시작
```bash
--guidance_scale 5.0
--spatial_threshold_start 0.7
--spatial_threshold_end 0.3
--threshold_strategy linear_decrease
```

### 2단계: 강도 조절 (너무 약하면)
```bash
--guidance_scale 7.0  # 5.0 → 7.0
--spatial_threshold_end 0.2  # 0.3 → 0.2 (더 공격적)
```

### 3단계: 강도 조절 (너무 강하면)
```bash
--guidance_scale 3.0  # 5.0 → 3.0
--spatial_threshold_end 0.4  # 0.3 → 0.4 (덜 공격적)
```

### 4단계: 양방향 추가 (더 강력하게)
```bash
--use_bidirectional
--harmful_scale 1.0  # 또는 1.5
```

---

## 🎓 결론

새 버전은 **"Less is More"** 철학을 따릅니다:
- ✅ 단순함: 26개 → 9개 파라미터
- ✅ 직관성: 이중 임계값 → 단일 adaptive 임계값
- ✅ 효율성: 감지 단계 제거
- ✅ 일관성: 항상 적용으로 예측 가능한 동작

**핵심**: `spatial_threshold`의 step별 adaptive 조정만으로도 충분히 효과적인 제어 가능!
