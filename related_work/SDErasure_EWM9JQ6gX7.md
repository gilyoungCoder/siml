# SDErasure: Concept-Specific Trajectory Shifting for Concept Erasure via Adaptive Diffusion Classifier

## 기본 정보
- **논문 제목**: SDErasure: Concept-Specific Trajectory Shifting for Concept Erasure via Adaptive Diffusion Classifier
- **저자**: Fengyuan Miao, Shancheng Fang*, Lingyun Yu*, Yadong Qu, Yuhao Sun, Xiaorui Wang, Hongtao Xie
- **소속**: USTC, Shenzhen University, Metastone Technology
- **학회/년도**: **ICLR 2026** (published as conference paper)
- **OpenReview**: https://openreview.net/forum?id=EWM9JQ6gX7

---

## 1. 핵심 문제 정의

기존 concept erasure 방법들의 **과도한 모델 침습성(model intrusiveness)** 문제를 두 차원에서 지적:

1. **Related-concept quality 훼손**: 지워진 concept으로 생성한 이미지가 시각적으로 비정상적 (예: "cat" 지웠는데 이상한 artifact)
2. **Unrelated-concept fidelity 훼손**: 무관한 concept의 생성 품질까지 저하 (FID 급증)

**근본 원인 진단**: 기존 방법들이 **모든 concept에 동일한 erasure 전략을 적용**하며, concept마다 생성 과정에서 중요한 timestep이 다르다는 점을 무시.

**핵심 발견**: 각 concept의 생성은 좁은 범위의 **critical timesteps**에 집중됨
- 구조적 concept (airplane, church): **높은 noise 단계** (early steps)에서 결정
- 세밀한 의미적 concept (얼굴, 스타일): **낮은 noise 단계** (late steps)에서 결정

---

## 2. 핵심 방법론

### Training-free 여부: **Training-based** (fine-tuning 필요)

SDErasure는 3개 핵심 컴포넌트로 구성:

### 2.1 Step Selection Algorithm (SSScore)

**목표**: 각 concept에 대해 fine-tuning할 최적 timestep을 자동으로 찾기

**방법**: Diffusion model을 classifier로 활용 (Li et al., 2023의 diffusion classifier)

1. Target concept `c_t`와 anchor concept `c_a`에 대해 각 timestep에서 noise prediction error 계산:
```
L_t^(c) = ||ε_θ(x_t, t, c_t) - ε||²
L_t^(a) = ||ε_θ(x_t, t, c_a) - ε||²
```

2. **SSScore (Step Separability Score)**: Bayes' theorem 기반 instantaneous posterior probability
```
S_t = exp(-L_t^(c)) / (exp(-L_t^(c)) + exp(-L_t^(a)))
```

3. **Threshold-based selection**: `S_t > λ`인 timestep만 erasure fine-tuning 대상
   - λ ∈ [0.5, 0.8]이 최적, λ=0.8이 best
   - SSScore 계산은 **1회 전처리** (학습/추론 중 반복 비용 없음)

**직관**: S_t가 높으면 해당 timestep에서 target과 anchor의 denoising trajectory가 크게 분기 → 이 지점에서 개입하면 최소한의 부작용으로 erasure 가능

### 2.2 Score Rematching Loss (핵심 학습 목표)

Target concept의 noise prediction을 anchor concept 방향으로 rematch:

**기본 alignment loss**:
```
L_a = ||ε_θ(x_t, c_t, t) - ε_θ*(x_t, c_a, t)||²
```

**Negative guidance term** (원본 모델의 concept 방향 차이):
```
σ(x_t, c_t, c_a, t) = ε_θ*(x_t, c_t, t) - ε_θ*(x_t, c_a, t)
```

**최종 Score Rematching loss**:
```
L_e = ||ε_θ(x_t, c_t, t) - [ε_θ*(x_t, c_a, t) - η·σ(x_t, c_t, c_a, t)]||²
```

- `η`: erasure 강도 제어
- `c_a = empty`로 설정하면 anchor-free erasure (ESD와 유사)
- Self-contrastive scheme: 원본 모델(frozen)이 fine-tuned 모델을 supervise

### 2.3 Quality Regulation

**Early-preserve loss** (구조 보존):
```
L_p = ||ε_θ(x_t*, c_t, t*) - ε_θ*(x_t*, c_t, t*)||²
```
- `t*`는 early denoising steps (e.g., 45 < t < 50)
- 초기 단계에서 concept들이 수렴하므로, 이 구간의 trajectory를 보존

**Concept-retain loss** (무관 concept 보존):
```
L_r = ||ε_θ(x_t, c_r, t) - ε_θ*(x_t, c_r, t)||²
```
- `c_r`: 보호할 무관 concept

**최종 목적함수**:
```
L_o = L_e + β₁·L_r + β₂·L_p
```

---

## 3. WHEN/WHERE/HOW 프레임워크 분석

### WHEN (언제 guidance를 적용하는가)
- **Concept-specific adaptive timestep selection**: SSScore가 threshold λ를 넘는 timestep에서만 erasure fine-tuning
- 모든 timestep에 균일하게 적용하지 않음 → 이것이 핵심 차별점
- Structural concept → early steps, Fine-grained concept → late steps (자동 탐지)

### WHERE (어디에 적용하는가)
- **전체 latent space**: spatial mask 사용 없음
- Noise prediction 전체에 대해 score rematching 수행
- 공간적 선택성은 없고, **시간적 선택성(temporal selectivity)**에 집중

### HOW (어떻게 개입하는가)
- **Model weight fine-tuning**: U-Net 파라미터를 직접 수정
- Score rematching: target concept의 noise prediction → anchor direction으로 shift
- ESD의 negative guidance loss를 기반으로 발전시킨 형태
- Quality regulation으로 과도한 수정 방지

---

## 4. 실험 결과 (핵심)

### 4.1 Object Erasure (CIFAR-10)
| Method | Avg Acc_e↓ | Avg Acc_s↑ | Avg H₀↑ |
|--------|-----------|-----------|---------|
| ESD-u | 15.65% | 88.53% | 84.53 |
| MACE | 7.03% | 96.72% | 92.78 |
| ANT | 14.70% | 98.58% | 88.87 |
| **SDErasure** | **2.71%** | **98.00%** | **95.33** |

### 4.2 Celebrity Erasure (FID가 핵심)
| Method | FID↓ (Elon Musk) | FID↓ (Taylor Swift) |
|--------|-----------------|-------------------|
| ESD | 13.50 | 13.55 |
| UCE | 12.60 | 11.85 |
| ANT | 12.56 | 12.61 |
| **SDErasure** | **7.60** | **6.49** |

→ FID를 9.51 → 6.74로 줄임 (SOTA)

### 4.3 Explicit Content Erasure (Nudity, I2P)
| Method | NudeNet Total↓ | FID↓ | CLIP↑ |
|--------|---------------|------|-------|
| ANT | **23** | 41.25 | 29.23 |
| **SDErasure** | 49 | **16.92** | **30.84** |
| RECE | 69 | 19.16 | 30.60 |
| ESD-u | 121 | 21.10 | 30.07 |

→ ANT보다 nudity 탐지가 약간 많지만, FID(16.92 vs 41.25)와 CLIP(30.84 vs 29.23)에서 압도적 우위

---

## 5. 장점과 한계

### 장점
1. **Concept-specific adaptivity**: 각 concept마다 최적 timestep 자동 탐지 → 정밀한 개입
2. **Generation quality 보존**: FID 대폭 개선 (9.51 → 6.74), related/unrelated concept 모두 보존
3. **유연한 구조**: anchor-based altering + anchor-free erasing 모두 지원
4. **SSScore의 효율성**: 1회 전처리로 끝나므로 학습/추론에 추가 비용 없음
5. **Multi-concept erasure 지원**: 20명 celebrity 동시 erasure 실험

### 한계
1. **Training 필요**: Fine-tuning이 필수이므로 training-free 방법 대비 비용 높음
2. **Source code 공개 시 bypass 가능**: Training-based이므로 모델 가중치를 공개해야 방어 가능
3. **Spatial selectivity 없음**: 전체 latent에 대해 작용하므로, 이미지 내 특정 영역만 선택적 처리 불가
4. **Nudity erasure에서 ANT 대비 약간 열세**: NudeNet 49 vs 23 (하지만 FID trade-off 훨씬 우수)
5. **Anchor 설계 의존**: Anchor concept 선택이 성능에 영향 (heuristic rule 사용)

---

## 6. 우리 방법(CAS + Spatial CFG)과의 관계/차이점

### 근본적 차이: Training-based vs Training-free
| 항목 | SDErasure | 우리 방법 (CAS + Spatial CFG) |
|------|-----------|------------------------------|
| **학습 필요** | Fine-tuning 필수 | 완전 training-free |
| **모델 수정** | U-Net 가중치 직접 수정 | 모델 가중치 변경 없음 |
| **WHEN** | SSScore로 critical timestep 선택 (concept-specific) | CAS threshold로 harmful prompt 탐지 |
| **WHERE** | 전체 latent (spatial 선택 없음) | Spatial mask (cross-attn/noise 기반) |
| **HOW** | Score rematching fine-tuning | SLD-style noise subtraction |
| **적용 대상** | Object, Celebrity, Style, Nudity | 주로 Nudity (multi-concept 확장 중) |
| **Temporal selectivity** | 핵심 기여 (concept별 adaptive timestep) | 제한적 (CAS가 timestep별 gating) |
| **Spatial selectivity** | 없음 | 핵심 기여 (WHERE guidance) |
| **Bypass 저항** | 모델 가중치에 내재 → 소스코드만으로 bypass 어려움 | Inference-time → 소스코드 공개 시 bypass 가능 |

### 상호 보완 가능성
1. **SSScore의 temporal selectivity 아이디어**: CAS guidance를 특정 timestep에만 적용하는 전략으로 차용 가능
   - 현재 우리 방법은 모든 timestep에 CAS를 계산하고 threshold로 gating
   - SDErasure처럼 concept-specific critical timestep만 선택하면 더 정밀할 수 있음
2. **Quality Regulation 아이디어**: Early timestep에서 guidance를 억제하는 전략은 우리의 COCO FP 문제 해결에 참고 가능
3. **Anchor concept 활용**: SDErasure의 anchor-based score rematching과 우리의 anchor_shift 모드가 유사한 철학

### Related Work 관점에서의 위치
- SDErasure는 **training-based methods** 카테고리에서 SOTA
- 우리 방법은 **training-free methods** 카테고리
- 논문에서 SLD를 training-free baseline으로 언급하며, 효과는 약하지만 품질 보존이 좋다고 평가
- 우리 방법이 SLD를 넘어서는 training-free 성능을 보여줄 수 있다면 positioning 명확

---

## 7. 핵심 인사이트 요약

1. **"언제 개입할지"가 "어떻게 개입할지"만큼 중요하다** — concept마다 생성에 관여하는 timestep이 다름
2. **Diffusion model as classifier**: SSScore는 추가 모델 없이 U-Net 자체로 concept 분류 가능하다는 것을 활용
3. **Fine-tuning의 precision**: 모든 timestep 대신 5-10개 critical timestep만 fine-tuning해도 충분
4. **Quality regulation의 중요성**: Early-preserve + concept-retain으로 generation quality 보존
