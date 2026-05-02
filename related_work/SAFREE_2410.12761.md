# SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation

**arXiv**: 2410.12761
**저자**: Jaehong Yoon*, Shoubin Yu*, Vaidehi Patil, Huaxiu Yao, Mohit Bansal (UNC Chapel Hill)
**학회/년도**: ICLR 2025
**코드**: https://safree-safe-t2i-t2v.github.io/

---

## 1. 핵심 문제 정의

기존 training-based unlearning/editing 방법들의 세 가지 한계를 지적:
1. **즉시 적용 불가**: 새로운 concept (예: 특정 아티스트 스타일)을 제거하려면 추가 학습이 필요
2. **학습 데이터 의존**: safe generation 능력이 수집한 학습 데이터에 종속
3. **모델 가중치 변경**: 가중치를 수정하면 target concept과 무관한 content의 품질이 저하됨

기존 training-free 방법 (SLD 등)도 한계가 있음:
- Implicit/indirect한 unsafe prompt에 대한 방어 취약
- Hard filtering이 distribution shift를 유발하여 생성 품질 저하

**SAFREE의 목표**: 모델 가중치를 전혀 변경하지 않으면서, text embedding과 visual latent 양쪽 공간에서 adaptive하게 unsafe concept을 필터링하는 plug-and-play 메커니즘 제공.

---

## 2. 핵심 방법론 (상세)

SAFREE는 4개의 핵심 컴포넌트로 구성:

### 2.1 Adaptive Token Selection (Toxic Concept Subspace Proximity)

**목적**: 입력 prompt에서 unsafe content를 유발하는 trigger token을 탐지

**Toxic concept subspace 정의**:
- 사용자 정의 unsafe keyword들 (예: "Sexual Acts", "Pornography", "Nudity")의 CLIP text embedding column vector를 concatenate하여 toxic concept subspace `C = [c_0; c_1; ...; c_{K-1}]` 구성 (C는 D x K 행렬)

**Trigger token 탐지 과정**:
1. 각 i번째 token을 masking한 pooled embedding `p_{\i}` 계산
2. `p_{\i}`를 C에 projection한 뒤, C에 직교하는 잔차 벡터 `d_{\i}` 계산:
   ```
   z = (C^T C)^{-1} C^T p_{\i}
   d_{\i} = (I - P_C) p_{\i},  where P_C = C(C^T C)^{-1} C^T
   ```
3. 잔차 norm `||d_{\i}||_2`이 클수록 해당 token 제거 시 toxic subspace에서 멀어짐 = 해당 token이 toxic concept과 강하게 연관
4. Masking 결정:
   ```
   m_i = 1  if ||d_{\i}||_2 > (1 + alpha) * mean(D(p|C).delete(i))
   m_i = 0  otherwise
   ```
   여기서 alpha = 0.01 (모든 실험에서 고정)

**핵심 인사이트**: token을 제거했을 때 toxic subspace와의 거리가 평균보다 유의미하게 커지면, 그 token이 toxicity의 trigger라고 판단.

### 2.2 Concept Orthogonal Token Projection

**목적**: 탐지된 toxic token을 직접 제거하지 않고, toxic subspace에 직교하는 방향으로 projection

**왜 직접 제거하지 않는가?**
- Token을 null embedding으로 교체하면 prompt 구조가 깨지고 input space 밖으로 이탈
- 품질 저하가 심각 (ablation에서 N 방식 vs P 방식 비교)

**Projection 과정**:
1. Input space `I = [p_{\0}; p_{\1}; ...; p_{\N-1}]`에 대한 projection matrix `P_I` 계산
2. Toxic subspace에 직교하면서 input space에는 남아있도록 projection:
   ```
   p_proj = P_I (I - P_C) p
   p_safe = m * p_proj + (1 - m) * p
   ```
   - toxic token (m_i=1): projected safe embedding 사용
   - safe token (m_i=0): 원래 embedding 유지

**핵심**: orthogonal projection이므로 toxic concept 성분만 제거하고 나머지 semantic은 최대한 보존.

### 2.3 Self-Validating Filtering (Adaptive Denoising Step Control)

**목적**: safe embedding을 적용할 denoising timestep 수를 input-dependent하게 자동 결정

**관찰**: 모든 denoising step에서 safe embedding을 적용하면 과도한 필터링으로 품질 저하 발생. 서로 다른 timestep이 toxicity 생성에 불균등하게 기여.

**Self-validating threshold 계산**:
```
t' = gamma * sigmoid(1 - cos(p, p_proj))
```
여기서 gamma = 10 (논문 전체에서 고정).

**적용 규칙**:
```
p_safree = p_safe   if t <= round(t')
p_safree = p        otherwise (원래 prompt 사용)
```

**원리**: `cos(p, p_proj)`이 높을수록 (= toxic content가 많이 제거되었을수록) 더 많은 step에서 safe embedding을 적용. 반대로 benign prompt는 cos similarity가 높아 t'이 작아지므로 거의 필터링하지 않음.

### 2.4 Adaptive Latent Re-attention (Fourier Domain)

**목적**: text embedding 필터링만으로는 pixel level에서 unsafe content가 남을 수 있으므로, latent space에서도 추가 필터링

**방법**: Fourier domain에서 latent feature의 low-frequency 성분을 선택적으로 감쇠
1. 원래 prompt p와 safe prompt p_safree 각각으로 latent feature h(p), h(p_safree)에 FFT 적용:
   ```
   F(p) = b * FFT(h(p))
   F(p_safree) = b * FFT(h(p_safree))
   ```
   (b는 low-frequency 영역을 나타내는 binary mask)

2. p_safree의 low-frequency가 p보다 큰 경우 scalar s로 감쇠:
   ```
   F'_i = s * F(p_safree)_i   if F(p_safree)_i > F(p)_i
   F'_i = F(p_safree)_i       otherwise
   ```
   (s < 1)

3. IFFT로 역변환하여 refined latent feature h' 획득

**핵심 인사이트**: Low-frequency 성분이 이미지의 global structure/style/context를 담당하므로, safe prompt 조건의 low-frequency가 과도하게 amplify되면 oversmoothing이 발생. 이를 억제하면서 unsafe content의 regional 출현을 방지.

---

## 3. Training-Free 여부

**완전한 Training-Free**. 모델 가중치를 일절 수정하지 않음.
- Training/Editing time: 0
- Inference time: 9.85초/sample (A6000 기준, 100 steps)
- Model modification: 0%
- SLD-Max (9.82초)와 거의 동일한 속도, ESD (~4500초), CA (~484초) 대비 압도적으로 빠름

---

## 4. WHEN / WHERE / HOW 프레임워크 분석

### WHEN (언제 guidance를 적용하는가)

**Self-Validating Filtering으로 adaptive하게 결정**:
- t' = gamma * sigmoid(1 - cos(p, p_proj))로 threshold 계산
- t <= round(t')인 초기 denoising step에서만 safe embedding 적용
- t > round(t')인 이후 step에서는 원래 prompt를 그대로 사용

**특징**:
- Input-dependent: 각 prompt마다 다른 수의 step에 적용
- Toxic prompt일수록 cos(p, p_proj)이 낮아져 t'이 커짐 -> 더 많은 step에서 필터링
- Benign prompt일수록 cos(p, p_proj)이 높아져 t'이 작아짐 -> 거의 필터링 안 함
- **우리의 CAS와 유사한 "adaptive when" 개념이지만, 메커니즘이 다름**: CAS는 매 step마다 cos(d_prompt, d_target)으로 on/off 결정하는 반면, SAFREE는 inference 전에 총 몇 step 적용할지를 한 번에 결정

### WHERE (어디에 적용하는가)

**두 가지 공간에서 동시 적용**:
1. **Text embedding space**: toxic token만 선택적으로 projection (token-level selective)
2. **Visual latent space (Fourier domain)**: low-frequency 성분의 선택적 감쇠

**특징**:
- Text 공간: token 단위로 localize (toxic token만 projection, safe token은 유지)
- Latent 공간: frequency domain에서 global low-frequency를 감쇠 (spatial locality 없음)
- **우리의 Spatial CFG와의 차이**: 우리는 cross-attention map 기반으로 spatial (pixel) 단위 mask를 만들어 WHERE를 결정하지만, SAFREE는 spatial localization이 아닌 frequency domain에서 global하게 처리

### HOW (어떻게 개입하는가)

**두 단계 개입**:
1. **Text embedding 조작**: Orthogonal projection으로 toxic 성분 제거
   - p_safe = m * P_I(I - P_C)p + (1-m) * p
   - CFG noise prediction을 바꾸는 것이 아니라, 입력 text condition 자체를 변형

2. **Latent feature 조작**: Fourier domain re-attention
   - Denoising 중 latent feature의 FFT low-frequency 성분 감쇠

**SLD/우리 방법과의 근본적 차이**:
- SLD/우리 방법: noise prediction 수준에서 개입 (epsilon_safe = epsilon_cfg - s * mask * (epsilon_target - epsilon_null))
- SAFREE: text embedding 자체를 변형 + latent feature의 frequency 조작
- SAFREE는 noise prediction 공식을 건드리지 않음 -> classifier-free guidance 구조를 그대로 유지

---

## 5. 실험 결과 요약

### 주요 성능 (ASR - Attack Success Rate, 낮을수록 좋음)

| Method | I2P | P4D | Ring-A-Bell | MMA-Diff | UnlearnDiff |
|--------|-----|-----|-------------|----------|-------------|
| SD-v1.4 | 0.178 | 0.987 | 0.831 | 0.957 | 0.697 |
| SLD-Max | 0.115 | 0.742 | 0.570 | 0.570 | 0.479 |
| UCE | 0.103 | 0.667 | 0.331 | 0.867 | 0.430 |
| RECE | 0.064 | 0.381 | 0.134 | 0.675 | 0.655 |
| **SAFREE** | **0.034** | **0.384** | **0.114** | **0.585** | **0.282** |

### COCO 품질 (FID/CLIP/TIFA)

| Method | FID | CLIP | TIFA |
|--------|-----|------|------|
| SLD-Max | 50.51 | 28.5 | 0.720 |
| SAFREE | 36.35 | 31.1 | 0.790 |
| SD-v1.4 | - | 31.3 | 0.805 |

SAFREE가 training-free 중 최저 ASR을 달성하면서, COCO 품질도 SLD-Max보다 훨씬 우수.

### 일반화 (SDXL, SD-v3, T2V)

- SDXL: unsafe output 48% 감소
- SD-v3: unsafe output 47% 감소
- ZeroScopeT2V, CogVideoX에도 적용 가능 (T2V 확장)

---

## 6. 장점

1. **완전한 Training-Free**: 가중치 수정 없이 plug-and-play로 적용
2. **Architecture-Agnostic**: UNet (SD-v1.4, SDXL), DiT (SD-v3), T2V (ZeroScope, CogVideoX) 모두 적용 가능
3. **Adaptive Filtering**: prompt의 toxicity 정도에 따라 자동으로 필터링 강도 조절 (Self-Validating)
4. **Safe Content 보존**: orthogonal projection으로 toxic 성분만 제거, safe semantic은 유지
5. **다양한 Concept 지원**: nudity, violence, artist style 등 user-defined concept 자유롭게 설정
6. **SOTA 성능**: training-free 중 최고 성능, training-based와도 경쟁적
7. **이중 공간 필터링**: text + latent 양쪽에서 필터링하여 robust

## 7. 한계

1. **Implicit/Chain-of-Thought Toxic Prompt에 취약**: 논문 자체에서 인정 — 매우 implicit하거나 chain-of-thought 스타일의 toxic prompt에는 여전히 뚫릴 수 있음
2. **Frequency Domain의 Global 처리**: latent re-attention이 spatial locality가 없어 이미지의 특정 영역만 targeted하게 처리하기 어려움
3. **Token-Level Detection의 한계**: 단어 단위로 toxicity를 판단하므로, 개별 단어는 safe하지만 조합하면 unsafe한 경우 (compositional toxicity) 탐지가 어려울 수 있음
4. **Fixed Concept Subspace**: toxic concept keyword를 미리 정의해야 하며, 정의되지 않은 새로운 형태의 toxicity에 대응 어려움
5. **Oversmoothing 리스크**: low-frequency 감쇠가 과도하면 이미지 detail 손실 가능 (s 파라미터 의존)
6. **Adversarial Attack에 대한 완전한 방어 아님**: 논문에서도 "perfect safeguarding"은 open problem이라고 명시

---

## 8. 우리 방법 (CAS + Spatial CFG)과의 관계/차이점

### 공통점

| 측면 | SAFREE | CAS + Spatial CFG (Ours) |
|------|--------|--------------------------|
| Training-Free | O | O |
| Adaptive Filtering | O (Self-Validating) | O (CAS threshold) |
| 모델 가중치 수정 | X | X |
| 주요 모델 | SD-v1.4 | SD-v1.4 |
| 평가 데이터셋 | I2P, P4D, Ring-A-Bell, MMA, UnlearnDiff | Ring-A-Bell, MMA, P4D, UnlearnDiff |

### 핵심 차이점

#### (1) WHEN 메커니즘의 차이
- **SAFREE**: inference 시작 전에 t' = gamma * sigmoid(1 - cos(p, p_proj))로 총 적용 step 수를 한 번에 결정. 초기 step들에서만 적용.
- **Ours (CAS)**: 매 denoising step마다 cos(d_prompt, d_target)을 실시간 계산하여 해당 step에서 guidance on/off를 동적 결정.
- **차이의 의미**: SAFREE는 "처음 N step"이라는 연속 블록으로 적용하지만, CAS는 중간 step에서도 on/off가 가능한 더 세밀한 temporal control. 실제로 diffusion에서 unsafe content가 특정 중간 timestep에서 emerge할 수 있으므로, CAS의 per-step adaptive 방식이 더 유연할 수 있음.

#### (2) WHERE 메커니즘의 차이
- **SAFREE**: text token 단위 선택 (embedding space) + Fourier low-frequency 감쇠 (latent space). Spatial locality 없음.
- **Ours (Spatial CFG)**: cross-attention map 기반 spatial mask 생성. 이미지의 어느 영역에 unsafe concept이 집중되는지 파악하여 해당 영역에만 guidance 적용.
- **차이의 의미**: SAFREE는 "어떤 token이 toxic인가"를 WHERE로 보지만, 우리는 "이미지의 어떤 pixel 영역이 toxic인가"를 WHERE로 봄. 근본적으로 다른 공간에서의 localization.

#### (3) HOW 개입 방식의 차이
- **SAFREE**: text embedding 자체를 orthogonal projection으로 변형 -> 변형된 embedding으로 일반 CFG 수행
- **Ours**: noise prediction 수준에서 개입. epsilon_safe = epsilon_cfg - s * mask * (epsilon_target - epsilon_null). SLD 스타일의 safe guidance.
- **차이의 의미**: SAFREE는 input (text condition)을 바꾸고, 우리는 output (noise prediction)을 바꿈. SAFREE 방식은 모델 내부 연산에 개입하지 않아 architecture-agnostic하지만, 우리 방식은 denoising 과정의 직접적 control이 가능.

#### (4) Concept Direction 정의 방식
- **SAFREE**: CLIP text embedding의 toxic keyword column vector들로 subspace 구성. Projection distance로 proximity 측정.
- **Ours (CAS)**: exemplar prompt 기반 concept direction. d_target = E[epsilon(z_t, anchor)] - epsilon(z_t, null)로, 실제 diffusion process 내에서 concept direction을 정의.
- **차이의 의미**: SAFREE는 text space에서 static하게 정의, 우리는 diffusion noise space에서 dynamic하게 정의. 우리 방식이 diffusion model의 internal representation을 더 직접적으로 반영.

### 우리 방법에 대한 시사점

1. **Text Embedding Filtering의 보완 가능성**: SAFREE의 orthogonal projection은 우리 파이프라인의 전처리로 추가 가능. Text를 먼저 정제한 뒤 CAS + Spatial CFG를 적용하면 이중 방어 가능.

2. **Self-Validating vs CAS**: SAFREE의 cos(p, p_proj) 기반 threshold와 우리의 cos(d_prompt, d_target) 기반 CAS는 상호 보완적. SAFREE는 text space에서, CAS는 noise space에서 toxicity를 측정하므로 앙상블 가능.

3. **Spatial Locality가 우리의 강점**: SAFREE의 가장 큰 약점은 pixel-level spatial control이 없다는 점. 우리의 cross-attention 기반 spatial mask가 이를 보완하는 차별적 강점.

4. **Architecture Agnostic은 SAFREE의 강점**: 우리 방법은 UNet의 cross-attention에 의존하므로 DiT 등으로 확장이 쉽지 않을 수 있음. SAFREE는 text embedding만 조작하므로 어떤 backbone에도 적용 가능.

5. **벤치마크 비교 기회**: SAFREE가 ICLR 2025에 발표되어 강력한 baseline. 우리 NeurIPS 제출 시 SAFREE와의 직접 비교가 필수적. 특히 Ring-A-Bell (SAFREE: 0.114)과 MMA (SAFREE: 0.585)에서의 ASR 비교가 핵심.

6. **Fourier Re-attention 아이디어 참고**: Latent space에서의 frequency domain 처리는 우리의 spatial mask와 orthogonal한 접근. 두 가지를 결합하면 spatial + frequency 양쪽에서 필터링 가능.

---

## 9. 논문에서 참고할 핵심 수식 정리

| 수식 | 내용 |
|------|------|
| Eq.1 | CFG: epsilon_t = (1+w) * epsilon(z_t, p) - w * epsilon(z_t, null) |
| Eq.2 | Projection coordinate: z = (C^T C)^{-1} C^T p_{\i} |
| Eq.3 | Residual (orthogonal to C): d_{\i} = (I - P_C) p_{\i} |
| Eq.4 | Token masking: m_i = 1 if ||d_{\i}|| > (1+alpha) * mean |
| Eq.5 | Safe projection: p_safe = m * P_I(I-P_C)p + (1-m) * p |
| Eq.6 | Self-validating: t' = gamma * sigmoid(1 - cos(p, p_proj)) |
| Eq.7-8 | Fourier re-attention: FFT -> selective low-freq attenuation -> IFFT |

---

## 10. 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| alpha | 0.01 | Token detection sensitivity (모든 실험 고정) |
| gamma | 10 | Self-validating scale factor |
| s | < 1 | Fourier low-frequency attenuation scalar |
| Toxic keywords | User-defined | 예: "Sexual Acts", "Pornography" for nudity |
