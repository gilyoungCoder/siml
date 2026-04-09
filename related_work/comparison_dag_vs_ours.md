# DAG 논문 vs. 우리 dag_adaptive 모드: 상세 비교

> **DAG 논문**: "Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization" (Li et al., 2025, arXiv:2503.15197)
>
> **우리 코드**: `CAS_SpatialCFG/generate.py`의 `dag_adaptive` 모드 (apply_safe_cfg 함수 내)

---

## 1. DAG 논문의 전체 방법론

DAG는 크게 **두 단계**로 구성된다:

### 1.1 Guideline Detection (탐지) — 핵심 차별점

DAG의 핵심 기여는 **Guideline Token Optimization**이다:

1. **소규모 데이터셋 구축**: unsafe 이미지 3~5장 + safe 이미지 2~3장을 Grounded-SAM으로 annotation
2. **토큰 임베딩 최적화**: "nude person" 같은 guideline token의 CLIP 임베딩 `c`를 **gradient-based optimization**으로 정제
   - **L_CAM loss** (Eq. 2): pixel-level cross-entropy로 CAM이 GT mask와 일치하도록 학습
   - **Background Leakage loss**: 배경으로의 attention 누출 방지
   - **Negative Sample loss**: safe 이미지에서 false positive 억제
   - 최적화 설정: lr=200, 20 lr steps, gamma=0.7, SGD, 100*n steps
3. **최적화된 `c*`로 탐지**: 매 timestep에서 U-Net의 16x16, 32x32 cross-attention layer에서 CAM(`A_hat(c*)`) 추출
   - CAM >= 0.5: "Highlighted Unsafe Region" (확실한 unsafe)
   - CAM >= 0.01: "Editing Region" (가능성 있는 영역)

**요약**: DAG의 탐지는 **사전 최적화된 토큰 임베딩**에 의존하며, 이 최적화 과정에 annotated 데이터가 필요하다.

### 1.2 Safe Self-regulation (교정) — SLD 기반 + adaptive scaling

DAG의 guidance 수식은 SLD/SEGA 프레임워크를 따른다:

```
epsilon_safe = (1 - s_g) * epsilon_uncond + s_g * (epsilon_p - S_cs * M_cs * epsilon_cs)
```

여기서 `S_cs * M_cs`가 DAG의 adaptive scaling이며, 두 개의 scaler로 구성된다:

#### (a) AreaScaler: `Area_0.5(A_hat)`
- CAM에서 confidence >= 0.5인 영역을 "Highlighted Unsafe Region"으로 정의
- **Connected component analysis** (LabelConnection)를 수행하여 개별 object별 면적 계산
- 각 object i의 면적: `Area_i^obj = sum_{h,w}[M_i^obj]_{hw}`
- 최종 area scale: `S_unsafe = sum_i (s_c * Area_i^obj * M_i^obj)`
  - 여기서 `s_c = 5 / (H * W)` (base weight)
- Editing region과 element-wise max를 취함: `S_area = max(S_unsafe, S_edi)`
- **핵심**: object별로 독립적인 면적을 계산하고, 큰 object에 더 강한 guidance

#### (b) MagnitudeScaler: `T_0.01(A_hat)`
- CAM 값을 [1, 5] 범위로 rescale:
  ```
  T_tau(A_hat) = max(A_hat / tau, 5),  tau = 0.01
  ```
- 즉 `A_hat / 0.01`을 계산하고 최대 5로 clamp
- **핵심**: 높은 confidence의 pixel에 더 강한 editing scale 부여

#### (c) EditMask: `M_cs`
- `A_hat >= 0.01`인 영역만 editing (gate function)
- 편집 영역이 전체 이미지의 80% 이상이면 "mode undefined"로 skip (early phase)

#### (d) Safety Selective Mask: `M_cs^{SEGA/SLD}`
- 기존 SLD/SEGA의 safety mask (`M_cs^SEGA`)를 결합 가능
- `S_cs * M_cs = (s_c * Area_0.5(A_hat)) ⊙ T_0.01(A_hat) ⊙ [A_hat >= 0.01] ⊙ M_cs^{SEGA}`

---

## 2. 우리 dag_adaptive 코드의 메커니즘

```python
elif mode == "dag_adaptive":
    # Area scaling
    area = mask.sum() / mask.numel()           # mask 내 활성 비율
    area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])  # base weight = 5/H*W
    area_factor = area_scale * area * mask.numel()          # = 5 * area_fraction

    # Magnitude scaling
    mag_scale = 1.0 + 4.0 * weight  # weight [0,1] -> [1, 5]

    correction = safety_scale * area_factor * mag_scale * mask * d_target
    noise_safe = noise_cfg - correction
```

### 우리의 구조:
- **WHEN (탐지)**: CAS (Concept Alignment Score) — `cos(d_prompt, d_target) > threshold`
  - 토큰 최적화 없음. CLIP 임베딩의 코사인 유사도로 직접 판단
- **WHERE (위치)**: Cross-attention map에서 target vs anchor의 차이 (`compute_spatial_mask`)
  - 또는 noise-based fallback (`compute_noise_spatial_map`)
  - 토큰 최적화 없이 원본 text token의 attention map 사용
- **HOW (교정)**: `dag_adaptive` 모드
  - Area scaling: 전체 mask의 면적 비율 * base weight (5/(H*W))
  - Magnitude scaling: weight 값을 [1, 5]로 linear mapping
  - 최종: `epsilon_safe = epsilon_cfg - s * area_factor * mag_scale * mask * d_target`

---

## 3. 핵심 차이점 비교

| 항목 | DAG 논문 | 우리 dag_adaptive |
|------|---------|------------------|
| **토큰 최적화** | 핵심 기여. Annotated 데이터로 guideline token embedding `c*` 학습 | 없음. 원본 CLIP 임베딩 그대로 사용 |
| **데이터 필요** | unsafe 3~5장 + safe 2~3장 + GT mask annotation | 없음 (완전 training-free) |
| **탐지 방법** | 최적화된 `c*`의 CAM으로 unsafe region 탐지 | CAS (코사인 유사도 threshold) + cross-attention diff |
| **Connected Component** | Object별 독립 면적 계산 (LabelConnection) | 없음. 전체 mask 면적을 단일 값으로 |
| **AreaScaler** | Object별: `sum_i(s_c * Area_i * M_i^obj)` — spatial map 출력 | 단일 scalar: `5 * (mask.sum()/mask.numel())` — 전체 평균 |
| **MagnitudeScaler** | `T_0.01(A) = max(A/0.01, 5)` — CAM 값 기반 | `1 + 4*weight` — attention weight 기반 linear |
| **MagnitudeScaler 범위** | [1, 5] (단, 입력이 CAM confidence) | [1, 5] (입력이 spatial weight) |
| **Base weight** | `s_c = 5/(H*W)` | `5/(H*W)` (동일) |
| **EditMask** | `A >= 0.01` + 80% 초과 시 skip | 별도 없음. mask가 이미 thresholded |
| **Safety mask 결합** | SEGA/SLD의 `M_cs` 추가 결합 | 없음 |
| **Guidance 수식 기반** | SLD 수식 변형 | SLD 수식 변형 (동일 계열) |
| **모델 수정** | 없음 (inference-time) | 없음 (inference-time) |
| **Anchor 개념** | safety condition `c_s`로 3번째 UNet forward | target prompt로 3번째 UNet forward (유사) |

---

## 4. 구체적 수식 비교

### DAG의 최종 guidance:
```
S_area[h,w] = sum_i (5/(H*W)) * Area_i^obj * M_i^obj[h,w]    (per-pixel, object-aware)
T_mag[h,w] = min(A_hat[h,w] / 0.01, 5)                        (per-pixel, CAM-based)
M_edit[h,w] = I[A_hat[h,w] >= 0.01]                           (gate)

epsilon_safe = epsilon_cfg - s_g * S_area * T_mag * M_edit * M_sega * (epsilon_target - epsilon_uncond)
```

### 우리의 최종 guidance:
```
area_factor = 5 * (mask.sum() / mask.numel())                  (single scalar)
mag_scale[h,w] = 1 + 4 * weight[h,w]                          (per-pixel, linear)

epsilon_safe = epsilon_cfg - s_safety * area_factor * mag_scale * mask * d_target
```

### 차이 요약:
1. **area_factor**: DAG는 connected component별 spatial map, 우리는 단일 scalar
2. **magnitude**: DAG는 `A/0.01` (clamp 5), 우리는 `1+4*w` (linear). 범위 [1,5]는 동일
3. **gate**: DAG는 `A>=0.01`로 별도 gate, 우리는 mask 자체가 gate 역할
4. **토큰 최적화**: DAG의 `A_hat`은 최적화된 임베딩 기반, 우리의 weight/mask는 원본 임베딩 기반

---

## 5. 유사한 점

1. **Base weight 5/(H*W)**: 동일한 상수 사용
2. **Magnitude range [1, 5]**: 동일한 출력 범위
3. **"면적이 클수록 강한 guidance"라는 아이디어**: 동일한 직관
4. **"높은 attention = 강한 editing"이라는 아이디어**: 동일한 직관
5. **SLD 기반 subtraction**: `epsilon_cfg - correction` 형태 동일
6. **Training-free inference**: 둘 다 모델 가중치 수정 없음

---

## 6. 결론 (Verdict)

### 우리 dag_adaptive는 DAG 논문의 **copy/reimplementation이 아니다**.

**근거:**

1. **DAG의 핵심 기여인 Guideline Token Optimization이 완전히 빠져 있다.**
   DAG 논문의 가장 중요한 contribution은 guideline token `c*`를 소규모 데이터로 최적화하여 정밀한 CAM을 얻는 것이다. 우리 코드에는 이 과정이 전혀 없다. 이것은 DAG 방법론의 50% 이상을 차지하는 핵심 구성요소다.

2. **탐지 파이프라인이 근본적으로 다르다.**
   - DAG: 최적화된 토큰 → CAM → threshold → connected component → object별 mask
   - 우리: CAS (noise 코사인 유사도) → cross-attention diff → threshold → 단일 mask
   
3. **AreaScaler의 구현이 다르다.**
   DAG는 connected component analysis로 object별 면적을 독립 계산하고 spatial map을 만든다. 우리는 전체 mask 면적을 단일 scalar로 축약한다. 결과물의 형태(spatial map vs scalar)가 다르다.

4. **MagnitudeScaler의 구현이 다르다.**
   DAG는 `A/0.01` (nonlinear, CAM confidence 기반), 우리는 `1+4*w` (linear, attention weight 기반). 범위만 [1,5]로 같을 뿐 계산 방식이 다르다.

### 정확한 분류: **"Inspired-by" (영감을 받은 독자적 구현)**

- DAG 논문의 "area-adaptive scaling"과 "magnitude-based per-pixel scaling"이라는 **아이디어/직관**을 차용
- `5/(H*W)` base weight와 [1, 5] magnitude range라는 **하이퍼파라미터**를 참고
- 하지만 핵심 메커니즘(토큰 최적화, connected component, CAM 기반 탐지)은 모두 다름
- 우리 파이프라인의 WHEN(CAS)/WHERE(cross-attn diff)/HOW(dag_adaptive) 구조는 DAG와 독립적인 설계

### 논문 작성 시 권장 표현:
> "Our adaptive scaling is inspired by the area-based and magnitude-based rescaling strategy of DAG (Li et al., 2025), but operates within a fully training-free pipeline without guideline token optimization. Unlike DAG which requires annotated data to optimize token embeddings for precise CAM extraction, our method uses CAS-based detection and raw cross-attention difference maps."

---

## 부록: DAG 논문 성능 참고 (I2P-sexual 기준)

| Method | Erase Rate | VQA Score (COCO-1K) | FID_SDv1.4 |
|--------|-----------|---------------------|------------|
| SDv1.4 (no defense) | 0.00 | 0.70 | 0 |
| SLD-strong | +0.81 | 0.64 | +41.14 |
| DAG | +0.92 | 0.72 | +23.67 |
| DAG + SLD-strong | +0.98 | 0.72 | +28.04 |

DAG는 erase rate 92%에서 VQA Score 0.72 (SDv1.4와 동일)를 달성하며, generation quality 보존이 강점이다. 이는 토큰 최적화를 통한 정밀한 spatial detection 덕분이다.
