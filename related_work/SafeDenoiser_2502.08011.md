# Training-Free Safe Denoisers for Safe Use of Diffusion Models

**arXiv**: 2502.08011  
**저자**: Mingyu Kim (UBC), Dongjun Kim (Stanford), Amman Yusuf (UBC), Stefano Ermon (Stanford), Mijung Park (UBC)  
**학회**: NeurIPS 2025 (39th Conference on Neural Information Processing Systems)  
**키워드**: Safe denoiser, training-free, concept erasing, negation set, diffusion model safety

---

## 1. 핵심 문제 정의

Diffusion model이 생성하는 unsafe content (NSFW, 저작권 침해, 개인정보 유출 등)를 **training 없이 inference 시점에** 제거하는 문제.

기존 방법의 한계:
- **Text-based 방법** (SLD, SAFREE): negative prompt이나 unsafe token 조작에 의존 -> adversarial prompt (예: MMA-Diffusion)에서 텍스트에 explicit한 nudity 단어가 없으면 무력화
- **Fine-tuning 방법** (ESD, RECE): catastrophic forgetting 위험, 재학습 비용
- **Sparse Repellency (SR)**: 유사한 접근이지만 이론적 안전 보장 없음, 이미지 품질 저하

이 논문은 **negation set** (unsafe 데이터 포인트들)을 활용하여 sampling trajectory 자체를 수정함으로써, safe distribution에서만 샘플이 나오도록 **이론적으로 보장**하는 방법을 제안.

---

## 2. 핵심 방법론

### 2.1 Safe Denoiser 정의

데이터 분포를 safe/unsafe로 분할:
- `1_safe(x)`: x가 safe이면 1, 아니면 0
- `1_unsafe(x)`: x가 unsafe이면 1, 아니면 0
- `1 = 1_safe(x) + 1_unsafe(x)` (partition of unity)

**Safe denoiser**:
```
E_safe[x|x_t] = integral( x * p_safe(x) * q_t(x_t|x) / p_{safe,t}(x_t) dx )
```

**Unsafe denoiser**:
```
E_unsafe[x|x_t] = integral( x * p_unsafe(x) * q_t(x_t|x) / p_{unsafe,t}(x_t) dx )
```

### 2.2 핵심 정리 (Theorem 3.2)

Safe denoiser와 data denoiser의 관계:

```
E_safe[x|x_t] = E_data[x|x_t] + beta*(x_t) * (E_data[x|x_t] - E_unsafe[x|x_t])    ... (Eq. 4)
```

여기서 weight:
```
beta*(x_t) = (Z_unsafe * p_{unsafe,t}(x_t)) / (Z_safe * p_{safe,t}(x_t))    ... (Eq. 5)
```

**직관**: 
- `beta*(x_t)`는 현재 x_t가 unsafe할 확률이 높을수록 증가
- Safe denoiser = data denoiser + adaptive weight * (data denoiser - unsafe denoiser)
- CFG와 구조적으로 유사하지만, unsafe 방향을 negate하는 것이 핵심

### 2.3 Unsafe Denoiser 근사

N개의 unsafe 데이터 포인트 `x^(1), ..., x^(N)`을 사용:

```
E_unsafe[x|x_t] ~= sum_{n=1}^{N} x^(n) * q_t(x_t|x^(n)) / sum_{m=1}^{N} q_t(x_t|x^(m))    ... (Eq. 6)
```

- 각 unsafe sample에 대한 가중 평균 (gaussian kernel 기반 가중치)
- 현재 noisy sample x_t와 가까운 unsafe sample이 더 큰 가중치를 받음

### 2.4 Weight 근사

`beta*(x_t)`의 정확한 계산이 불가능하므로:

```
beta*(x_t) ~= eta * beta(x_t)
```

여기서:
```
beta(x_t) = integral( p_unsafe(x) * q_t(x_t|x) dx ) ~= (1/N) * sum_{n=1}^{N} q_t(x_t|x^(n))
```

- `eta`는 controllable hyperparameter
- `beta(x_t)`는 x_t가 unsafe 영역에 가까울수록 커짐 (adaptive weight)

### 2.5 최종 Safe Denoiser 수식

```
E_safe[x|x_t] = E_data[x|x_t] + eta * beta(x_t) * (E_data[x|x_t] - E_unsafe[x|x_t])    ... (Eq. 7)
```

### 2.6 Text-to-Image 확장

SAFREE와 결합:
```
x_{0|t} = E_safe[x|x_t] + lambda * (E_data[x|x_t, c_tilde_+] - E_data[x|x_t])    ... (Eq. 8)
```

SLD와 결합:
```
x_{0|t} = E_safe[x|x_t] + lambda * (E_data[x|x_t, c_+] - E_data[x|x_t])        [CFG 부분]
           - mu * (E_data[x|x_t, c_tilde_-] - E_data[x|x_t])                     [SLD 부분]    ... (Eq. 9)
```

핵심: 기존 CFG에서 `E_data`를 `E_safe`로 교체하는 것이 전부.

### 2.7 Threshold beta_t

Text-to-image 생성 시, `beta(x_t) < beta_t`이면 safe denoiser를 적용하지 않음:
- 이미 충분히 safe한 trajectory에는 개입하지 않아 품질 보존
- `beta_t`는 timestep별로 설정 가능 (ablation에서 U-shaped 최적값 확인)

### 2.8 Algorithm 1 요약

```
Input: pretrained model, unsafe data {x^(n)}, hyperparams eta, beta_t, critical timesteps C
for t = T to 0:
    E_data[x|x_t] <- (1/alpha_t)(x_t - sigma_t * epsilon_theta(x_t, t))
    E_unsafe[x|x_t] <- weighted sum of x^(n) by gaussian kernels
    if text-to-image: compute conditional denoisers
    beta(x_t) <- (1/N) * sum q_t(x_t|x^(n)) if t in C, else 0
    if text-to-image: threshold beta(x_t) with beta_t
    compute x_{0|t} using Eq. 7, 8, or 9
    x_{t-1} = Solver(x_t, t, x_{0|t})
```

---

## 3. Training-Free 여부

**완전히 Training-Free.**
- 모델 가중치 수정 없음
- Fine-tuning 없음
- 필요한 것: pretrained diffusion model + unsafe data points (negation set) + hyperparameters
- Inference 시 추가 연산은 sub-linear 수준 (Table 5: 3000개 negative image 사용 시 이미지당 ~0.07초 추가)

---

## 4. WHEN / WHERE / HOW 프레임워크 분석

### WHEN (언제 guidance를 적용하는가)

- **Critical timesteps C에서만 적용**: 논문에서 safe denoiser는 sampling 초기 단계(t가 큰 구간)에서만 적용하고, 후반부(t가 작은, detail 생성 단계)에서는 적용하지 않음
- **이유**: 후반 단계에서 unsafe denoiser의 신호가 structural noise로 작용하여 품질 저하
- **Threshold beta_t**: `beta(x_t) > beta_t`인 경우에만 적용 (adaptive하게 unsafe한 trajectory에만 개입)
- **우리 방법(CAS)과의 비교**: CAS는 cos(d_prompt, d_target)으로 WHEN을 결정하는 반면, 이 논문은 gaussian kernel 기반 `beta(x_t)` + threshold로 결정. CAS는 텍스트 기반 semantic 판단이고, 이 방법은 이미지 공간에서의 proximity 기반.

### WHERE (어디에 적용하는가)

- **Spatial 구분 없음**: 이미지의 전체 latent에 균일하게 적용
- Cross-attention map이나 spatial mask 같은 공간적 선택 메커니즘이 **전혀 없음**
- 이는 이 방법의 가장 큰 한계 중 하나: unsafe content가 이미지의 일부분에만 있더라도 전체 denoiser를 수정
- **우리 방법(Spatial CFG)과의 비교**: 우리는 cross-attention map 기반으로 unsafe 영역에만 selective하게 guidance를 적용하므로, 이미지의 safe한 부분은 보존하면서 unsafe 부분만 수정 가능

### HOW (어떻게 개입하는가)

- **Data denoiser에서 unsafe denoiser 방향을 빼는 방식**:
  ```
  E_safe = E_data + weight * (E_data - E_unsafe)
  ```
- CFG와 구조적으로 동일한 형태이지만, "조건부 denoiser"가 아닌 "unsafe data에서 근사한 denoiser"를 사용
- **Negation set (exemplar images)** 기반: 텍스트 프롬프트가 아닌 실제 unsafe 이미지들을 참조
- Adaptive weight `beta(x_t)`: gaussian kernel로 현재 trajectory와 unsafe data의 proximity 측정
- **우리 방법과의 비교**: 우리의 HOW는 SLD-style `epsilon_safe = epsilon_cfg - s * mask * (epsilon_target - epsilon_null)` 형태. 이 논문은 data prediction space에서, 우리는 noise prediction space에서 작동하는 차이가 있지만 기본 구조는 유사 (safe = data - weight * unsafe_direction).

---

## 5. 실험 결과 요약

### 5.1 Nudity Prompt별 성능 (Table 2)

| Method | Ring-A-Bell ASR | UnlearnDiff ASR | MMA ASR | COCO FID | COCO CLIP |
|--------|----------------|-----------------|---------|----------|-----------|
| SD-v1.4 | 0.797 | 0.809 | 0.962 | 25.04 | **31.38** |
| SLD | 0.481 | 0.629 | 0.881 | 36.47 | 29.28 |
| SLD + Ours | 0.354 | 0.429 | 0.481 | 36.59 | 29.10 |
| SAFREE | 0.278 | 0.353 | 0.601 | 25.29 | 30.98 |
| **SAFREE + Ours** | **0.127** | **0.207** | **0.469** | **22.55** | 30.66 |

- SAFREE + Safe Denoiser가 가장 좋은 성능
- 특히 MMA-Diffusion에서 0.962 -> 0.469로 큰 폭 개선 (text-based 방법의 한계를 image-based로 보완)
- FID가 오히려 개선됨 (diversity 증가 효과)

### 5.2 CoPro (Multi-concept, Table 3)

- 7개 카테고리 (Harassment, Hate, Illegal Activity, Self-harm, Sexual, Shocking, Violence) 전부에서 IP 감소
- CLIP score 하락은 미미 (~0.4 정도)

### 5.3 Computation Overhead (Table 5)

- SD-v1.4 단독: 3.18 s/img
- + Ours (N=515): 3.20 s/img (+0.02s, 거의 무시 가능)
- SAFREE + Ours (N=3000): 4.29 s/img (SAFREE 단독 4.22 대비 +0.07s)

### 5.4 SD-v3 호환성 (Table 7)

- SD-v3에서도 동작 확인: ASR 0.304 -> 0.203, FID 23.15 -> 22.54

---

## 6. 장점

1. **이론적 기반**: Safe distribution에서 sampling한다는 formal guarantee (Theorem 3.2)
2. **Training-free**: 모델 수정 없이 inference 시점에 적용
3. **Plug-and-play**: SLD, SAFREE 등 기존 text-based 방법과 쉽게 결합 가능
4. **Adversarial prompt 강건성**: 텍스트가 아닌 이미지 기반이므로 MMA-Diffusion 같은 adversarial attack에 강함
5. **극도로 낮은 overhead**: 수천 개 negative image에도 이미지당 ~0.07초 추가
6. **Multi-concept 지원**: CoPro 7개 카테고리 동시 erasing 가능
7. **FID 개선 효과**: safe denoiser가 diversity를 증가시켜 FID를 오히려 개선
8. **저작권/프라이버시**: style-level IP 보호, data memorization 완화에도 적용 가능

---

## 7. 한계

1. **Spatial awareness 부재**: 이미지 전체에 균일하게 적용하므로, safe한 영역도 불필요하게 변형될 수 있음
2. **Negation set 의존**: unsafe exemplar image가 필요. 이 데이터셋의 품질과 범위가 성능에 직접 영향
3. **Hyperparameter tuning**: eta, beta_t를 concept/dataset별로 조정해야 함 (U-shaped 관계)
4. **후반부 timestep 미적용**: detail 단계에서는 적용하지 않으므로, 미세한 unsafe detail이 살아남을 가능성
5. **Standalone 성능 제한**: 단독 사용 시 text-based guidance보다 약할 수 있음 (Table 1에서 negative prompt 없이 ASR 0.633)
6. **Privacy 평가 부족**: 논문에서도 standardized metric 부재를 한계로 인정
7. **Error bar 없음**: computational resource 한계로 statistical significance 미보고

---

## 8. 우리 방법 (CAS + Spatial CFG)과의 관계 및 차이점

### 8.1 구조적 유사성

| 측면 | Safe Denoiser | CAS + Spatial CFG (ours) |
|------|--------------|--------------------------|
| Training-free | O | O |
| 핵심 수식 | E_safe = E_data + weight * (E_data - E_unsafe) | epsilon_safe = epsilon_cfg - s * mask * (epsilon_target - epsilon_null) |
| Guidance 형태 | Data denoiser에서 unsafe direction 빼기 | Noise prediction에서 unsafe direction 빼기 |
| 기본 구조 | CFG-like negation | SLD-style negation |

둘 다 본질적으로 **unsafe direction을 negate하는 CFG-variant**이라는 점에서 같은 계열.

### 8.2 핵심 차이점

| 차원 | Safe Denoiser | CAS + Spatial CFG (ours) |
|------|--------------|--------------------------|
| **WHEN 결정** | beta(x_t) > beta_t (gaussian kernel proximity) | CAS score > threshold (CLIP cosine similarity) |
| **WHEN 기반** | Image-space proximity (pixel/latent) | Text-semantic alignment (CLIP embedding) |
| **WHERE** | 전체 이미지 (spatial 구분 없음) | Cross-attention map 기반 spatial mask |
| **HOW: unsafe direction** | Exemplar images로부터 직접 계산 | Target concept text embedding에서 계산 |
| **Unsafe 정보 소스** | Negative image set (N개 이미지) | Target concept text prompt |
| **적응성** | Adaptive weight (trajectory 기반) | CAS threshold (semantic 기반) |

### 8.3 상호 보완 가능성

이 논문의 가장 큰 강점은 **adversarial prompt에 대한 강건성** (image-based이므로). 우리 방법의 강점은 **spatial selectivity** (WHERE). 두 방법은 서로 다른 축을 커버하므로 결합 가능성이 높음:

1. **Safe Denoiser의 beta(x_t)를 CAS의 추가 신호로 활용**: CAS score와 beta(x_t)를 ensemble하여 WHEN 결정의 robustness 향상
2. **Safe Denoiser에 Spatial mask 추가**: E_unsafe를 전체가 아닌 cross-attention mask 영역에만 적용
3. **Exemplar-based direction + CAS**: 우리의 v7 exemplar 방식과 유사하게, negative image에서 concept direction을 추출하되 CAS로 WHEN을 결정

### 8.4 우리 연구에 대한 시사점

1. **이론적 프레이밍 참고**: 우리 방법도 "safe distribution에서 sampling"이라는 관점으로 framing 가능. Theorem 3.2의 형태가 우리 SLD-style guidance와 본질적으로 같음을 보이면 이론적 기여 가능.
2. **Negation set 아이디어**: 우리의 exemplar prompt 대신 실제 unsafe image를 negation set으로 쓰는 것도 고려 가능. 특히 MMA-Diffusion 같은 adversarial attack 대응에 효과적.
3. **Adaptive weight**: beta(x_t)의 adaptive nature는 CAS threshold의 binary 결정보다 세밀한 조절 가능. 우리도 CAS score를 binary가 아닌 continuous weight로 사용하는 것을 고려할 수 있음.
4. **Multi-concept 실험**: CoPro 7개 카테고리 실험은 우리 Phase 2 계획과 직접적으로 관련. 이 논문의 CoPro 결과를 baseline으로 비교 가능.
5. **Baseline으로서의 가치**: NeurIPS 2025 accept이므로 반드시 비교해야 할 최신 baseline. 특히 SAFREE + Safe Denoiser 조합이 현재 SOTA.

### 8.5 경쟁력 분석

우리 방법이 이 논문보다 우위를 점할 수 있는 포인트:
- **Spatial selectivity**: safe 영역의 보존 (FP 감소에 유리)
- **Text-semantic WHEN**: CAS는 concept의 semantic alignment을 직접 측정하므로 더 정확한 개입 타이밍
- **경량성**: negative image set 불필요 (text prompt만으로 동작)
- **해석가능성**: CAS score와 spatial mask가 시각적으로 interpretable

이 논문이 우위인 포인트:
- **Adversarial robustness**: image-based이므로 text adversarial attack에 강함
- **이론적 보장**: formal safety guarantee
- **Plug-and-play 범용성**: 어떤 text-based method와도 결합 가능
- **Multi-concept**: 하나의 negation set으로 여러 concept 동시 처리

---

## 9. 관련 방법과의 비교 (논문에서 언급)

| 방법 | 유형 | 특징 | Safe Denoiser와의 관계 |
|------|------|------|----------------------|
| **SLD** | Training-free, text-based | Adaptive weight mu, unsafe prompt 기반 | Safe denoiser가 E_data를 E_safe로 대체하여 보강 |
| **SAFREE** | Training-free, text-based | Token filtering, modified prompt | Safe denoiser와 결합 시 최고 성능 |
| **Sparse Repellency** | Training-free, image-based | ReLU activation 기반 repulsion | Safe denoiser의 특수 케이스로 해석 가능, 이론적 보장 없음 |
| **ESD** | Fine-tuning | Concept erasure via gradient | Training 필요, catastrophic forgetting 위험 |
| **RECE** | Fine-tuning | Reliable concept erasure | Training 필요 |
| **Dynamic Negative Guidance** | Training-free | Markov chain 기반 | Extra training 필요 (unsafe denoiser용) |

---

## 10. 핵심 인사이트 요약

1. **"Safe denoiser = data denoiser + adaptive negation of unsafe denoiser"** 라는 깔끔한 이론적 공식화가 이 논문의 핵심 기여
2. Gaussian kernel을 이용한 unsafe denoiser 근사가 계산적으로 효율적이면서도 효과적
3. Text-based method와의 결합이 핵심 전략 (standalone보다 plug-in으로서의 가치가 더 큼)
4. **Image-based guidance**는 adversarial text attack에 대한 근본적 해법 (우리도 exemplar 방향 탐구 시 참고)
5. NeurIPS 2025 accept: 이 분야에서 가장 최신의 strong baseline으로 반드시 비교 필요
