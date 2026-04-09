# Method Evolution: v3 → v13 + 신규 제안 v14-v19

## 아키텍처 프레임워크

모든 버전은 동일한 3-Question 프레임워크를 공유:

```
Input: prompt P, model SD v1.4 (safety_checker=None)

For each denoising step t:
  1. WHEN: CAS(d_prompt, d_target) > tau?     → 이 step에서 개입할지?
  2. WHERE: spatial_mask M ∈ [0,1]^{H×W}      → 어디에 개입할지?
  3. HOW: eps_safe = f(eps_cfg, eps_target,    → 어떻게 수정할지?
          eps_anchor, M)

Output: safe image (nudity removed, content preserved)
```

## 기존 버전 요약표

| Version | WHEN | WHERE | HOW | UNet calls | NudeNet% | SR% | 상태 |
|---------|------|-------|-----|-----------|----------|-----|------|
| **v3/dag** | CAS text (tau=0.3) | CrossAttn maps | **dag_adaptive** (area+magnitude) | 3-4 | **0.95%** | **90.5%** | **HOW 최강 — 우리 독자 방법** |
| **v4** | CAS text (tau=0.3) | Noise Spatial CAS (3×3 unfold) | 5 modes | 3-4 | ~24% | ~96.5% | 발표된 벤치마크 |
| **v5** | CAS text | Anchor Projection axis | proj_subtract | 4 | - | - | 실험적 |
| **v6** | CAS text (tau=0.6) | **CrossAttn Probe** (text keys) | 5 modes | 3-4 | 15.5% | 91.4% | WHERE 최고 |
| **v7** | CAS text/exemplar | Noise Spatial CAS | Hybrid + exemplar anchor | 3 | **5.06%** | 54.1% | NN 최고 |
| **v8** | CAS text | Noise Spatial CAS | Exemplar Projection | 3 | 42-70% | - | **실패** |
| **v9** | CAS text | Noise Spatial CAS | Direct Exemplar | 3 | 72-76% | - | **실패** |
| **v10** | CAS text/exemplar | Noise Spatial CAS | proj_anchor (clamped) | 3 | 6.65% | 41.7% | 중간 |
| **v11** | CAS text | Noise Spatial CAS | Stochastic Ensemble | 3 | 6.65% | 41.1% | 중간 |
| **v12** | CAS text | CrossAttn + Noise hybrid | proj_anchor | 3-4 | 9.5% | 57.9% | 중간 |
| **v13** | CAS text | **CLIP Image Exemplar** Probe | 5 modes | 3-4 | 8.86% | - | Grid 미완 |

## 핵심 교훈 (v8-v13에서 배운 것)

### 1. Exemplar-based HOW는 실패 (v8, v9)
- **원인**: Pre-computed noise-space exemplar direction은 현재 이미지의 specific nudity 패턴과 불일치
- **교훈**: HOW에서는 online target (현재 이미지에서 직접 계산)이 필수. Exemplar은 anchor/WHERE에만 유효

### 2. v7의 SR 실패 원인: mask over-coverage
- v7 avg mask area: **0.88** (거의 전체 이미지)
- v6 avg mask area: **0.31** (focused)
- 같은 `apply_guidance()` 사용 → 차이는 오직 WHERE mask
- **교훈**: noise-based spatial CAS는 nudity prompt에 대해 globally correlated → 공간 선택성 부재

### 3. v13의 실패 원인: representation collapse
- CLIP CLS token (1개 768-dim 벡터) → 4번 반복 → cross-attention key
- 모든 key가 동일 → softmax attention이 uniform → 공간 선택성 없음
- **교훈**: 이미지 정보를 넣을 때 spatial diversity 보존 필수

### 4. v6의 nudity leak 원인: text probe의 한계
- `target_words = ["nude", "naked", "nudity", "nsfw", "bare", "body"]` — 너무 일반적
- Ring-A-Bell의 간접적 nudity 표현 (e.g., "sultry temptress") 에서 token match 실패
- **교훈**: 이미지 기반 probe가 텍스트 기반보다 잠재적으로 우수 (시각적 패턴 직접 매칭)

---

## 상세 버전 분석

### v4: Spatial CAS + Soft Anchor Inpainting (기준 모델)

**파일**: `CAS_SpatialCFG/generate_v4.py`

**WHEN**: Global CAS
```python
d_prompt = eps_prompt - eps_null      # prompt direction
d_target = eps_target - eps_null      # target concept direction
cas = cos_sim(d_prompt, d_target)     # [-1, 1]
guided = cas > tau                    # tau=0.3 (v4), 0.6 (v6+)
```

**WHERE**: Noise Spatial CAS (3×3 Neighborhood)
```python
# Per-pixel 36-dim feature vectors (3×3 unfold × 4 latent channels)
patches_p = unfold(d_prompt, kernel=3)  # [B, 36, H*W]
patches_t = unfold(d_target, kernel=3)  # [B, 36, H*W]
spatial_cas = cos_sim(patches_p, patches_t)  # per-pixel
mask = sigmoid(alpha * (spatial_cas - threshold))  # soft mask
mask = gaussian_blur(mask, sigma=1.0)
```

**HOW**: 5가지 모드
- `anchor_inpaint`: `eps = eps_cfg * (1-sM) + eps_anchor_cfg * sM`
- `sld`: `eps = eps_cfg - s * M * (eps_target - eps_null)`
- `hybrid`: `eps = eps_cfg - t_s*M*(eps_target - eps_null) + a_s*M*(eps_anchor - eps_null)`
- `hybrid_proj`: Projection + anchor blend
- `projection`: `d_safe = d_prompt - s * proj(d_prompt, d_target)`

### v6: Cross-Attention Probing WHERE

**파일**: `CAS_SpatialCFG/generate_v6.py` + `attention_probe.py`

**WHERE 혁신**: UNet의 cross-attention layer에서 직접 semantic spatial information 추출

```
[Offline 1회] K_target = to_k(text_embed("nude person"))  # per-layer pre-compute

[매 step] 기존 eps_prompt forward pass 중:
  Q_cond = to_q(hidden_states)              # UNet 내부 query
  probe_attn = softmax(Q_cond × K_target^T / √d)  # 추가 matmul (≈무시할 cost)
  → spatial attention map (어디가 "nude"에 attend 하는지)

[Aggregation]
  - Per-head average → per-layer average
  - Multi-resolution (16×16, 32×32) → upsample → 64×64 average
  - Normalize to [0,1] → sigmoid soft mask
```

**장점**: 
- Zero extra UNet calls (기존 forward pass에 piggyback)
- Semantically meaningful (cross-attention = "어떤 픽셀이 어떤 단어에 주목하는지")
- Focused mask (avg 0.31 area) → 높은 SR

**단점**:
- Text keyword matching 의존 → 간접적 nudity expression 놓침
- target_words 확장 시 COCO FP 증가 위험 ("body" → "body of water" 등)

### v7: GLASS Exemplar-Based Directions (최고 NN)

**파일**: `CAS_SpatialCFG/generate_v7.py` + `prepare_concept_subspace.py`

**Exemplar 방식**:
```
[Offline] 16 nude + 16 clothed exemplar latents 생성 (DDIM/GLASS)
  For each timestep t:
    z_t = alpha_t * z0 + sigma_t * noise
    d_target_t = UNet(z_t, t, "nude") - UNet(z_t, t, "")
    d_anchor_t = UNet(z_t, t, "clothed") - UNet(z_t, t, "")
  → concept_directions.pt: per-timestep target/anchor directions
```

**3가지 exemplar 모드**:
1. `exemplar`: Online target + pre-computed anchor (3 UNet calls, 25% 절약)
2. `text`: v4와 동일 (4 UNet calls)
3. `hybrid_exemplar`: weighted blend of online + exemplar directions

### v13: CLIP Image Exemplar Probe (현재 최신)

**파일**: `CAS_SpatialCFG/generate_v13.py` + `prepare_clip_exemplar.py`

**CLIP 이미지 임베딩 injection 과정**:
```
[Offline]
  1. CLIP ViT-L/14로 16개 nudity exemplar 이미지에서 feature 추출
     → get_image_features(): pooled CLS token [1, 768]
  2. 16개 이미지의 CLS token 평균: avg_feature [768]
  3. 768-dim 벡터를 4번 반복 → [1, 77, 768] pseudo-text 시퀀스 구성
     → [BOS, concept, concept, concept, concept, EOS, PAD, ...]
  4. (Optional) Text encoder transformer layers 통과

[Online] v6와 동일한 cross-attention probe, 단 K_target이 CLIP image 기반
```

**4가지 probe source**:
- `clip_exemplar`: CLIP 이미지 임베딩만 사용
- `text`: v6 fallback (텍스트만)
- `both`: CLIP mask ∪ noise CAS mask (union)
- `diff`: CLIP target mask - noise anchor mask

**한계**: CLS token 1개만 사용 (256개 patch token 버림) → spatial 정보 완전 손실

---

## 신규 제안: v14-v19

### v14: Hybrid WHERE Fusion ⭐ (최우선)

**아이디어**: v6의 focused cross-attention mask × v7의 noise CAS 확인 신호

```
mask_attn = cross_attention_probe(Q, K_text_target)   # focused, ~0.31 area
mask_cas  = spatial_cas(d_prompt, d_target)            # broad, ~0.88 area
mask_final = mask_attn × sigmoid(alpha × (mask_cas - threshold))
→ 교차점: body region에서 실제로 nudity 생성 중인 곳만 (~0.15-0.25 area)
```

**기대 효과**: SR ~75-85% (v6 수준 mask 집중도), NN ~5-8% (v7 수준 nudity 제거)
**구현**: v6 기반, v7의 `compute_spatial_cas()` 추가 → 기존 코드 조합
**시간**: 1-2시간

### v15: CLIP Patch Token Probe ⭐ (핵심 이미지 임베딩 개선)

**아이디어**: CLS token 대신 CLIP ViT-L/14의 256개 patch token 전체 사용

```
[Offline]
  CLIP ViT-L/14 vision_model(image).last_hidden_state → [1, 257, 1024]
  patch_tokens = features[:, 1:, :]  # skip CLS → [1, 256, 768] after projection
  
  For each exemplar image:
    patches[i] = extract_256_patches(image_i)
  
  # 가장 discriminative한 K개 패치 선택:
  # nude patches와 clothed patches 간 cosine distance가 큰 것
  selected_patches = top_K_discriminative(nude_patches, clothed_patches)
  
[Online] Cross-attention probe with K patch tokens as keys
  → 각 패치가 exemplar 이미지의 서로 다른 body 영역 정보를 담음
  → spatial selectivity 보존
```

**v13 대비 개선**: 256개 diverse token (vs 1개 반복 CLS) → cross-attention에서 진정한 spatial probing 가능
**위험**: CLIP 224px patches와 SD 64×64 latent 공간 간 alignment 불확실
**시간**: 4시간

### v16: Contrastive Image Direction

**아이디어**: CLIP space에서 nude-clothed 차이 벡터를 concept direction으로 사용

```
d_concept = normalize(mean(CLIP(nude_images)) - mean(CLIP(clothed_images)))
→ nudity-specific features만 isolation (포즈, 배경 등 공유 요소 제거)
→ CAS target direction 및 cross-attention probe key로 활용
```

**v13 대비 개선**: 단순 nude 평균이 아니라 (nude - clothed) 차이 → concept specificity 향상
**위험**: 차이 벡터의 magnitude가 작을 수 있음 (비슷한 이미지 쌍이면)
**시간**: 4시간

### v17: IP-Adapter Image Projection

**아이디어**: Pre-trained IP-Adapter의 Resampler module로 CLIP → cross-attention space 매핑

```
IP-Adapter Resampler: [1, 257, 768] → [1, K, cross_attn_dim]
  - K=4 or 16 tokens
  - 수백만 image-text 쌍으로 학습된 projection
  - UNet의 cross-attention과 호환되도록 설계됨

→ naive CLS 반복보다 압도적으로 좋은 cross-attention key 생성
```

**장점**: 사전 학습된 최적 projection, SD1.4/1.5와 호환
**위험**: "사전학습 weights 사용"이 training-free 주장과 충돌 가능 (CLIP도 마찬가지이긴 함)
**시간**: 1일

### v18: Timestep-Adaptive Mask Sharpening

**아이디어**: Denoising 단계에 따라 mask/guidance 강도 조절

```python
def adaptive_params(step_idx, total_steps):
    frac = step_idx / total_steps  # 0.0 (시작) → 1.0 (끝)
    # 초반: 넓고 강한 guidance (layout 단계에서 nudity 방지)
    # 후반: 좁고 약한 guidance (fine detail 보존)
    safety_scale = s_high * (1 - frac) + s_low * frac
    spatial_threshold = thresh_low * (1 - frac) + thresh_high * frac
    return safety_scale, spatial_threshold
```

**장점**: 어떤 WHERE 방법과도 결합 가능 (v14/v15와 조합)
**시간**: 30분

### v19: Multi-Exemplar Diverse Probe Ensemble

**아이디어**: 16개 exemplar를 평균하지 않고, 각각 개별 probe → union mask

```
For each exemplar image i = 1..16:
  K_target_i = precompute_keys(CLIP_embed(image_i))
  probe_mask_i = cross_attention_probe(Q, K_target_i)

mask_final = max(probe_mask_1, ..., probe_mask_16)  # union
→ 다양한 nudity 패턴 (front/back/side, male/female) 모두 포착
```

**장점**: 다양한 nudity 구성을 개별적으로 탐지 (averaging으로 인한 정보 손실 방지)
**위험**: mask union이 너무 넓어질 수 있음 → top-K discriminative exemplars만 사용
**시간**: 4시간

---

## 추천 실행 조합

**Primary Method (논문 메인)**: v15 (CLIP Patch Token) + dag_adaptive HOW + v18 (Timestep Adaptive)
- Image exemplar spatial probe + pixel-wise adaptive guidance + timestep scheduling
- "Image+Text Example-based Selective Guidance" 논문 narrative와 일치
- dag_adaptive가 HOW의 핵심: v3에서 이미 SR 90.5% / NN 0.95% 달성

**Ablation Study**:
| Row | Description | Purpose |
|-----|-------------|---------|
| SD baseline | No guidance | Floor |
| SLD | Text-only HOW (sld mode) | Previous work |
| SAFREE | Text subspace projection | Comparison |
| v3/dag | CAS + CrossAttn + dag_adaptive | **Ours: HOW 최강 (ablation 기준점)** |
| v4 (CAS+SpatialCFG) | Text-only WHEN+WHERE + hybrid HOW | Text-only baseline |
| v6 (CrossAttn Probe) | Focused WHERE + hybrid HOW | WHERE ablation |
| v14 (Hybrid WHERE) | Dual mask fusion + dag_adaptive | WHERE+HOW 조합 |
| v15 (CLIP Patch) | Image exemplar WHERE + dag_adaptive | **Image contribution 증명** |
| v15+v18 | Full method (Image WHERE + dag_adaptive + timestep) | **Main result** |
| v16 (Contrastive) | Difference direction | Direction ablation |

**HOW Mode Ablation** (핵심 — dag_adaptive가 왜 최강인지):
| HOW Mode | 방식 | NN% | SR% | 비고 |
|----------|------|-----|-----|------|
| sld | fixed-scale subtraction | - | - | SLD 원본 |
| hybrid | target subtract + anchor add | 5.06 | 54.1 | v7 결과 |
| dag_adaptive | area+magnitude adaptive scaling | **0.95** | **90.5** | v3 결과, **최강** |
| dual | SLD + anchor 결합 | - | - | 중간 |

**Comparison Table Format (논문 Table 1)**:
| Method | Type | Ring-A-Bell ||| MMA ||| COCO ||
|--------|------|NN%|SR%|VQA|NN%|SR%|VQA|FP%|FID|
