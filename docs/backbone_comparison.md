# SafeGen Cross-Backbone 기술 비교서

> SafeGen의 When-Where-How 프레임워크가 4개 diffusion backbone에서 어떻게 적응되는지에 대한 기술 문서.
> 작성일: 2026-04-16

---

## 1. Architecture Overview

### 1.1 SD v1.4 (CompVis/stable-diffusion-v1-4)

- **Denoiser**: UNet (encoder-decoder with skip connections)
- **Text Encoder**: CLIP ViT-L/14 (단일)
- **Scheduler**: DDIM (noise prediction, epsilon)
- **Latent Shape**: `[B, 4, 64, 64]` (512x512 이미지 기준)
- **Attention**: Cross-attention layers (`attn2`) — Q는 image features, K/V는 text embeddings
- **CFG**: Traditional — `eps_cfg = eps_null + scale * (eps_prompt - eps_null)`, negative/positive 2-pass
- **VAE**: 4-channel, `scaling_factor` 적용

**구현 파일**:
- `CAS_SpatialCFG/generate_v27.py` — v27 dual-probe 버전
- `SafeGen/safegen/generate_family.py` — family-grouped 버전

### 1.2 SD3 (stabilityai/stable-diffusion-3-medium-diffusers)

- **Denoiser**: MMDiT (Multi-Modal Diffusion Transformer, joint attention)
- **Text Encoder**: Triple — CLIP-L + CLIP-G + T5-XXL
- **Scheduler**: Flow Match Euler Discrete (velocity prediction)
- **Latent Shape**: `[B, 16, 128, 128]` (1024x1024 기준, 16-channel VAE)
- **Attention**: Joint self-attention — image tokens과 text tokens이 concat되어 함께 attention
  - Q/K/V: `attn.to_q/k/v` (image), `attn.add_q_proj/k_proj/v_proj` (text)
- **CFG**: Traditional — velocity space에서 `v_cfg = v_null + scale * (v_prompt - v_null)`
- **VAE**: 16-channel, `scaling_factor` + `shift_factor` 적용

**구현 파일**:
- `scripts/sd3/generate_sd3_safegen.py`
- `scripts/sd3/attention_probe_sd3.py`

### 1.3 FLUX.1-dev (black-forest-labs/FLUX.1-dev, 12B)

- **Denoiser**: DiT (Diffusion Transformer)
- **Text Encoder**: Dual — CLIP + T5-XXL
- **Scheduler**: Flow Match Euler Discrete (flow prediction)
- **Latent Shape**: `[B, 16, H, W]` -> packed `[B, (H/2)*(W/2), 64]` (2x2 patch packing)
- **Attention**: DiT attention with `img_ids`/`txt_ids` positional encoding
- **CFG**: **Embedded guidance** — `guidance_scale`이 tensor로 transformer에 직접 전달됨. Negative pass 불필요
  - `transformer.config.guidance_embeds = True`
  - 단일 forward pass가 이미 guided prediction
- **VAE**: `scaling_factor` + `shift_factor`, `_unpack_latents(latents, h, w, vae_scale_factor)`

**구현 파일**: `CAS_SpatialCFG/generate_flux1_v1.py`

### 1.4 FLUX.2-klein (black-forest-labs/FLUX.2-klein-4B)

- **Denoiser**: DiT (Diffusion Transformer, 4B params — FLUX.1보다 작음)
- **Text Encoder**: Mistral (단일, FLUX.1과 다름)
- **Scheduler**: Flow Match Euler Discrete
- **Latent Shape**: packed `[B, seq_len, C]` (FLUX.1과 동일한 packing scheme)
- **Attention**: DiT attention, `img_ids`/`txt_ids` 사용, `guidance=None` (embedded guidance 미사용)
- **CFG**: **Traditional** — SD v1.4와 동일하게 neg/pos 2-pass
  - `eps_cfg = en + cfg_scale * (ep - en)`
- **VAE**: Batch normalization 포함 — `bn.running_mean`/`bn.running_var` denormalization 필요

**구현 파일**: `CAS_SpatialCFG/generate_flux2klein_v1.py`

---

## 2. Key Structural Differences Table

| 항목 | SD v1.4 | SD3 | FLUX.1-dev | FLUX.2-klein |
|------|---------|-----|-----------|-------------|
| **Denoiser** | UNet | MMDiT | DiT (12B) | DiT (4B) |
| **Prediction type** | Epsilon (noise) | Velocity | Flow | Flow |
| **Attention** | Cross-attention (`attn2`) | Joint self-attention | DiT attention | DiT attention |
| **Text encoder** | CLIP ViT-L/14 | CLIP-L + CLIP-G + T5-XXL | CLIP + T5-XXL | Mistral |
| **Text embed shape** | `[1, 77, 768]` | `[1, N, 4096]` (triple concat) | `[1, N, 4096]` (dual) | `[1, N, D]` |
| **Latent channels** | 4 | 16 | 16 (64 packed) | C (packed) |
| **Latent resolution** | `64x64` (512px) | `128x128` (1024px) | packed sequence | packed sequence |
| **Scheduler** | DDIM | FlowMatchEuler | FlowMatchEuler | FlowMatchEuler |
| **CFG mechanism** | Traditional (2-pass) | Traditional (2-pass) | **Embedded** (1-pass) | Traditional (2-pass) |
| **VAE decode** | `/ scaling_factor` | `/ scaling_factor + shift_factor` | `/ scaling_factor + shift_factor` | BN denorm + unpatchify |
| **Pooled projections** | 없음 | `pooled_projections` 필요 | `pooled_projections` 필요 | 없음 |
| **Default steps** | 50 | 28 | 28 | 50 |
| **Default CFG/guidance** | 7.5 | 7.0 | 3.5 (embedded) | 4.0 |

---

## 3. SafeGen Adaptation per Backbone

### 3.1 WHEN: Global CAS (Concept Alignment Score)

#### 핵심 원리
CAS는 prompt noise direction과 target concept noise direction 간의 cosine similarity를 측정한다:

```
CAS = cos(d_prompt, d_target)
    = cos(prediction_prompt - prediction_null, prediction_target - prediction_null)
```

Cosine similarity는 **방향 기반**이므로, prediction type이 epsilon이든 velocity이든 flow이든 동일하게 작동한다. 이것이 SafeGen이 training-free로 backbone 간 이식 가능한 핵심 이유이다.

#### 구현 비교

**SD v1.4** (`generate_v27.py:51-63`):
```python
class GlobalCAS:
    def compute(self, ep, en, et):
        dp = (ep - en).reshape(1, -1).float()  # epsilon space
        dt = (et - en).reshape(1, -1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
```

**SD3** (`generate_sd3_safegen.py:50-77`):
```python
class GlobalCAS:
    def compute(self, v_prompt, v_null, v_target):
        dp = (v_prompt - v_null).reshape(1, -1).float()  # velocity space
        dt = (v_target - v_null).reshape(1, -1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
```

**FLUX.1-dev** (`generate_flux1_v1.py:50-70`) / **FLUX.2-klein** (`generate_flux2klein_v1.py:43-63`):
```python
class GlobalCAS:
    def compute(self, ep, en, et):
        dp = (ep - en).reshape(1, -1).float()  # flow space (packed sequence)
        dt = (et - en).reshape(1, -1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
```

#### 차이점 분석

| 항목 | SD v1.4 | SD3 | FLUX.1-dev | FLUX.2-klein |
|------|---------|-----|-----------|-------------|
| Prediction space | Epsilon | Velocity | Flow | Flow |
| Input tensor shape | `[1,4,64,64]` | `[1,16,128,128]` | `[1,seq,64]` (packed) | `[1,seq,C]` (packed) |
| `reshape(1,-1)` 후 | `[1, 16384]` | `[1, 262144]` | `[1, seq*64]` | `[1, seq*C]` |
| Default threshold | 0.6 | 0.4 | 0.6 | 0.6 |
| Sticky mode | Yes | Yes | Yes | Yes |

**핵심 인사이트**: `reshape(1, -1).float()` 후 cosine similarity를 계산하므로, tensor의 원래 shape이나 prediction type과 무관하게 동일한 공식이 적용된다. SD3에서 default threshold가 0.4로 낮은 이유는 velocity space의 direction 분포가 epsilon space와 다르기 때문이다.

**FLUX.1-dev 특이사항**: Embedded guidance 모델임에도 CAS 계산을 위해 별도의 null pass (`en`)를 수행한다. 이는 `ep` (guidance-embedded prediction)에서 `en`을 빼야 "prompt의 고유 방향"을 추출할 수 있기 때문이다.
- `generate_flux1_v1.py:425-435` — null pass에도 `guidance=guidance_tensor`를 전달 (transformer API 요구사항)

---

### 3.2 WHERE: Spatial Probe (Attention-based Mask)

WHERE 컴포넌트는 backbone 간 가장 큰 차이를 보인다. 이는 각 아키텍처의 attention 구조가 근본적으로 다르기 때문이다.

#### SD v1.4: UNet Cross-Attention Probe

**구조**: UNet의 `attn2` (cross-attention) 레이어에서 Q=image features, K=text embeddings로 attention이 계산된다. Probe는 이 구조를 활용하여 pre-computed target concept key로 추가 matmul을 수행한다.

**구현** (`attention_probe.py:67-93`):
```python
class ProbeCrossAttnProcessor:
    def __init__(self, store, layer_name, target_key):
        self.target_key = target_key  # K_target = to_k(target_embeds), [1, 77, dim]

    def __call__(self, attn, hidden_states, encoder_hidden_states, ...):
        # 1. Normal cross-attention (output unchanged)
        # 2. Probe: Q_img @ K_target^T → attention map
        #    Shape: [H, spatial_seq, num_target_tokens]
```

**핵심 특성**:
- Cross-attention 레이어별로 별도 hooking
- Resolution별 (`[16, 32]`) selective probing
- Token-level spatial map: 특정 target token이 어느 spatial position에 attend하는지 직접 추출
- Text probe + Image probe 동시 사용 가능 (dual probe, `generate_v27.py:337-389`)
- Image probe: CLIP image features를 text encoder baseline에 삽입하여 probe key 생성 (`generate_v27.py:185-207`)
- **추가 UNet call 불필요** — 기존 forward pass의 hook만으로 동작

**Mask 생성 파이프라인** (`generate_v27.py:440-490`):
1. Attention map 추출 (img_probe / txt_probe)
2. Token indices 기반 spatial aggregation
3. Sigmoid thresholding: `sigmoid(alpha * (attn - threshold))`
4. Gaussian blur smoothing
5. Dual probe fusion: `union` / `soft_union` / `mean`
6. Optional: Noise CAS gate (conservative false positive 제거)

#### SD3: Joint-Attention Probe

**구조**: MMDiT는 cross-attention 대신 joint self-attention을 사용한다. Image tokens과 text tokens이 concat된 후 함께 attention을 수행한다.

```
Q = [Q_img; Q_txt],  K = [K_img; K_txt]
attn = softmax(Q @ K^T / sqrt(d))
```

Cross-attention에 해당하는 부분은 이 joint attention matrix의 **image-to-text sub-matrix**이다:
```
cross_attn_like = attn[:, :num_img_tokens, num_img_tokens:]
```

**구현** (`attention_probe_sd3.py:44-153`):
```python
class JointAttentionProbeProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states, ...):
        num_img_tokens = hidden_states.shape[1]

        # Image projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Text projections (별도 projection layers)
        enc_q = attn.add_q_proj(encoder_hidden_states)
        enc_k = attn.add_k_proj(encoder_hidden_states)

        # Concatenate: [image, text]
        query = torch.cat([query, enc_q], dim=2)
        key = torch.cat([key, enc_k], dim=2)

        if self.store.active:
            # Explicit attention computation (no SDPA)
            attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))
            attn_probs = attn_weights.softmax(dim=-1)

            # Extract image→text sub-matrix
            b_idx = min(1, batch_size - 1)  # Conditional batch (CFG idx=1)
            cross_attn = attn_probs[b_idx, :, :num_img_tokens, num_img_tokens:]
            # Shape: [H, num_img, num_txt]
```

**핵심 차이점 vs SD v1.4**:

| 항목 | SD v1.4 | SD3 |
|------|---------|-----|
| Attention type | Separate cross-attention | Joint self-attention의 sub-matrix |
| Image Q projection | `attn.to_q` | `attn.to_q` (image-only) |
| Text K projection | `attn.to_k` (text via `encoder_hidden_states`) | `attn.add_k_proj` (별도 text projection) |
| RMSNorm | 없음 | `attn.norm_q`, `attn.norm_k`, `attn.norm_added_q/k` |
| Hook target | `attn2` processors | `transformer_blocks[i].attn` processors |
| Block selection | Resolution-based (`[16, 32]`) | Index-based (default: middle third) |
| Probe key precomputation | Pre-cached `K_target` | 실시간 joint attention에서 추출 |
| SDPA fallback | N/A | Probing 비활성 시 SDPA 사용 (속도 최적화) |

**제약사항**:
- SD3에서는 joint attention 때문에 explicit attention 계산이 필요 (SDPA는 attention weights를 반환하지 않음)
- CPU offload 환경에서 hook 등록/해제를 매 step마다 수행해야 함 (`generate_sd3_safegen.py:362-379`)
- Image probe (CLIP image features) 미구현 — text probe만 사용

**Spatial mask 계산** (`attention_probe_sd3.py:211-276`):
```python
def compute_sd3_spatial_mask(store, token_indices, latent_h, latent_w):
    for block_name, attn_map in maps.items():
        avg_map = attn_map.mean(dim=0)       # [num_img, num_txt] — head 평균
        spatial_map = token_map.max(dim=-1)[0] # [num_img] — token union
        # num_img = (H/patch)*(W/patch), reshape to 2D
        spatial_2d = spatial_map.view(1, 1, ph, pw)
        # Interpolate to target resolution
        spatial_2d = F.interpolate(spatial_2d, size=(latent_h, latent_w), ...)
```

SD3에서는 image tokens이 `(resolution / patch_size)^2` 개이므로 (1024px, patch=2 -> 64x64 = 4096 tokens), 이를 2D로 reshape한 후 latent resolution으로 interpolate한다.

#### FLUX.1-dev & FLUX.2-klein: WHERE 미구현 (Global Mask)

현재 FLUX 구현에서는 spatial probe가 구현되지 않았다. CAS가 trigger되면 **global mask** (전체 latent에 균일 적용)를 사용한다.

**FLUX.2-klein** (`generate_flux2klein_v1.py:67-91`):
```python
def apply_guidance(eps_cfg, eps_null, eps_target, eps_anchor,
                   mask, how, safety_scale, cfg_scale):
    m = mask  # scalar 1.0 (global)
```

**FLUX.1-dev** (`generate_flux1_v1.py:74-102`):
```python
def apply_guidance(eps_cfg, eps_null, eps_target, eps_anchor,
                   mask, how, safety_scale):
    m = mask  # scalar 1.0 (global)
```

**FLUX에서 spatial probe가 어려운 이유**:
1. **Packed latent format**: `[B, (H/2)*(W/2), C*4]` 형태로, image tokens이 1D sequence로 pack됨
2. **DiT attention 구조**: UNet cross-attention이나 MMDiT joint attention과 달리, FLUX DiT의 attention block 구조가 다름
3. **img_ids/txt_ids 기반 positional encoding**: Spatial 정보가 positional encoding에만 존재
4. 기술적으로 가능하지만 (packed sequence를 2D로 unpack → attention map 추출 → 다시 pack), v1 구현에서는 global guidance로 시작

**향후 구현 가능 방향**: FLUX DiT의 attention layer를 hook하여 image-to-text attention sub-matrix를 추출하고, `img_ids`의 spatial 좌표를 이용해 2D mask로 복원하는 방식.

---

### 3.3 HOW: Guided Denoising

#### 핵심 공식 (anchor_inpaint mode)

모든 backbone에서 기본 guidance mode는 `anchor_inpaint`이다:

```
prediction_guided = prediction_cfg * (1 - s*M) + prediction_anchor_cfg * (s*M)
```

여기서:
- `prediction_cfg`: CFG-guided prediction (또는 embedded-guided prediction)
- `prediction_anchor_cfg`: anchor concept의 CFG-guided prediction
- `M`: spatial mask (1.0 = unsafe 영역, 0.0 = safe 영역)
- `s`: safety_scale

#### Backbone별 HOW 구현 비교

**SD v1.4** (`generate_v27.py:90-155`) — 5가지 guidance mode 지원:

```python
# anchor_inpaint
ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
blend = (safety_scale * m).clamp(max=1.0)
out = eps_cfg * (1 - blend) + ea_cfg * blend

# hybrid
out = eps_cfg - ts * m * (eps_target - eps_null) + as_ * m * (eps_anchor - eps_null)

# hybrid_proj (projection removal + anchor blend)
# proj_replace (nudity component swap)
# target_sub
```

SD v1.4가 가장 많은 guidance mode를 지원하며, `hybrid_proj`와 `proj_replace`는 SD v1.4 전용이다.

**SD3** (`generate_sd3_safegen.py:106-129`) — velocity space에서 동일 공식:

```python
# anchor_inpaint (velocity space)
va_cfg = v_null + cfg_scale * (v_anchor - v_null)
blend = (s * m).clamp(max=1.0)
out = v_cfg * (1 - blend) + va_cfg * blend
```

수학적으로 SD v1.4와 동일하다. Epsilon을 velocity로 치환했을 뿐이다.

**FLUX.2-klein** (`generate_flux2klein_v1.py:67-91`) — traditional CFG:

```python
# anchor_inpaint (traditional CFG와 동일)
ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
blend = min(s * m, 1.0)  # m은 scalar 1.0
out = eps_cfg * (1 - blend) + ea_cfg * blend
```

SD v1.4와 동일한 공식. Mask가 scalar이므로 `.clamp` 대신 `min()` 사용.

**FLUX.1-dev** (`generate_flux1_v1.py:74-102`) — **Embedded guidance 핵심 차이**:

```python
# anchor_inpaint — NO cfg_scale in anchor formula!
blend = min(s * m, 1.0)
out = eps_cfg * (1 - blend) + eps_anchor * blend  # eps_anchor 자체가 이미 guided
```

FLUX.1-dev에서 `eps_cfg = ep` (single-pass output이 이미 guided prediction). Anchor prediction `eps_anchor`도 마찬가지로 이미 embedded guidance가 적용된 상태이므로, `eps_null + cfg_scale * (eps_anchor - eps_null)` 공식이 아닌 **직접 blend**를 사용한다.

#### HOW 비교표

| 항목 | SD v1.4 | SD3 | FLUX.1-dev | FLUX.2-klein |
|------|---------|-----|-----------|-------------|
| Prediction space | Epsilon | Velocity | Flow (embedded) | Flow |
| Anchor CFG formula | `en + cfg * (ea - en)` | `vn + cfg * (va - vn)` | `ea` 직접 사용 | `en + cfg * (ea - en)` |
| Mask type | Spatial (probe) | Spatial (probe) | Global (scalar) | Global (scalar) |
| Guidance modes | 5 (ainp/hyb/hyb_proj/proj_rep/tsub) | 3 (ainp/hyb/tsub) | 3 (ainp/hyb/tsub) | 3 (ainp/hyb/tsub) |
| Family guidance | Per-family spatial mask | 미지원 | Per-family global | Per-family global |
| UNet/Transformer calls per guided step | 4 (null+prompt+target+anchor) | 4 | 4 (prompt+null+target+anchor) | 4 |

#### FLUX.1-dev의 UNet Call 구조 (Embedded Guidance)

FLUX.1-dev는 traditional CFG가 없으므로 baseline에서는 **1-pass**만 필요하다. 하지만 SafeGen 적용 시:

```
Baseline:      1 call (ep = guided prediction)
SafeGen WHEN:  3 calls (ep + en + et) — CAS 계산을 위해 en 필요
SafeGen HOW:   4 calls (ep + en + et + ea) — anchor prediction 추가
```

이는 traditional CFG 모델 (SD v1.4, SD3, FLUX.2-klein)의 baseline 2-pass와 비교하면, SafeGen이 FLUX.1-dev의 inference 효율성 이점을 일부 상쇄한다.

---

## 4. Summary Comparison Table

| Component | SD v1.4 | SD3 | FLUX.1-dev | FLUX.2-klein |
|-----------|---------|-----|-----------|-------------|
| **Architecture** | UNet | MMDiT | DiT (12B) | DiT (4B) |
| **WHEN: CAS** | Epsilon cos-sim | Velocity cos-sim | Flow cos-sim | Flow cos-sim |
| **WHEN: Formula** | `cos(ep-en, et-en)` | `cos(vp-vn, vt-vn)` | `cos(ep-en, et-en)` | `cos(ep-en, et-en)` |
| **WHEN: Threshold** | 0.6 | 0.4 | 0.6 | 0.6 |
| **WHERE: Probe** | Dual (text+image) cross-attn | Joint-attn text probe | Global (미구현) | Global (미구현) |
| **WHERE: Resolution** | `[16, 32]` layers | Middle-third blocks | N/A | N/A |
| **WHERE: Cost** | 0 extra calls | 0 extra calls | N/A | N/A |
| **HOW: Guidance** | Spatial masked blend | Spatial masked blend | Global blend | Global blend |
| **HOW: anchor_inpaint** | `ecfg*(1-sM) + ea_cfg*(sM)` | `vcfg*(1-sM) + va_cfg*(sM)` | `ep*(1-s) + ea*(s)` | `ecfg*(1-s) + ea_cfg*(s)` |
| **HOW: Modes** | 5 | 3 | 3 | 3 |
| **Family grouped** | Spatial per-family | 미지원 | Global per-family | Global per-family |
| **Baseline calls** | 2 (CFG) | 2 (CFG) | 1 (embedded) | 2 (CFG) |
| **SafeGen calls** | 3-4 | 3-4 | 3-4 | 3-4 |
| **Text encoding** | `pipe.text_encoder(ids)` | `pipe.encode_prompt(p, p2, p3)` | `pipe.encode_prompt(p, p2)` | `pipe.encode_prompt(p)` |
| **Pooled projections** | 불필요 | 필요 (`pooled_projections`) | 필요 (`pooled_projections`) | 불필요 |
| **VAE decode** | `lat / sf` | `lat / sf + shift` | unpack + `lat / sf + shift` | BN denorm + unpatchify |
| **CPU offload** | 불필요 | 필요 (24GB GPU) | 필요 | 필요 |

---

## 5. Implications for the Paper

### 5.1 Cross-Backbone Compatibility가 입증하는 것

1. **Method-agnostic design**: SafeGen의 When-Where-How 프레임워크는 특정 아키텍처에 종속되지 않는다. CAS의 cosine similarity는 prediction type (epsilon/velocity/flow)에 무관하게 방향 기반으로 동작하며, guidance 공식도 prediction space에서의 선형 연산이므로 backbone 간 직접 이식 가능하다.

2. **Training-free의 실질적 의미**: Fine-tuning 기반 방법 (ESD, UCE, SA 등)은 새로운 backbone마다 재학습이 필요하다. SafeGen은 코드 적응만으로 즉시 적용 가능하며, 이는 빠르게 진화하는 diffusion model 생태계에서 실용적 이점이다.

3. **UNet -> Transformer 전환 대응**: SD v1.4 (UNet) -> SD3/FLUX (Transformer) 아키텍처 전환에도 WHEN/HOW 컴포넌트는 코드 수정 없이 동작한다. WHERE (probe)만 attention 구조에 따른 적응이 필요하다.

### 5.2 Backbone별 제약사항

| Backbone | 제약사항 | 영향 |
|----------|---------|------|
| **SD v1.4** | 없음 (가장 완전한 구현) | Full spatial + dual probe + 5 guidance modes |
| **SD3** | Image probe 미구현, CPU offload 필수 | Text probe만 사용, 매 step hook 등록/해제 오버헤드 |
| **FLUX.1-dev** | Spatial probe 미구현, embedded guidance로 인한 추가 pass | Global guidance만 가능, baseline 대비 3-4x call 증가 |
| **FLUX.2-klein** | Spatial probe 미구현 | Global guidance만 가능 |

### 5.3 Training-Free 접근의 장점

1. **즉시 배포 가능**: 새로운 모델 출시 시 (FLUX.2, SD3.5 등) 별도 학습 없이 adaptation 코드만 작성하면 됨
2. **원본 모델 보존**: 모델 weights를 변경하지 않으므로 benign 이미지 품질에 영향 없음
3. **Concept 확장 용이**: Target/anchor concept text만 변경하면 nudity 외 다른 concept (violence, hate 등)에도 적용 가능
4. **Computational cost**: Training-based 방법의 수 시간 학습 vs SafeGen의 inference-time 추가 2-3 forward pass

### 5.4 향후 개선 방향

1. **FLUX spatial probe**: DiT attention layer hooking + packed sequence unpack을 통한 2D mask 복원
2. **SD3 image probe**: MMDiT joint attention에 CLIP image features 주입
3. **Unified probe interface**: backbone-agnostic probe abstraction layer
4. **CAS threshold auto-tuning**: backbone별 optimal threshold를 calibration set으로 자동 결정

---

## 6. ML Research Perspective & Insights

이 섹션은 SafeGen의 cross-backbone 호환성을 ML 연구 관점에서 해석한다. 왜 training-free transfer가 가능한지, 어떤 구조적 병목이 WHERE 컴포넌트를 제약하는지, embedded guidance가 가져오는 수학적 tension은 무엇인지, 그리고 이 방법론의 일반화 한계와 다음 연구 방향을 정리한다.

### 6.1 Why Training-Free Transfer Works

#### Parameterization-invariant direction

CAS는 `cos(pred_prompt - pred_null, pred_target - pred_null)` 형태로 정의된다. 이 식의 핵심은 **두 벡터의 차이(difference)가 어떤 parameterization이든 "denoising trajectory의 변화 방향"을 가리킨다는 점**이다.

- **Epsilon (DDPM/DDIM)**: `eps_theta(x_t, t, c)` — noise residual
- **Velocity (v-parameterization, SD3)**: `v = alpha_t * eps - sigma_t * x_0` — noise와 clean signal의 선형 결합
- **Flow matching (FLUX)**: `u_theta(x_t, t, c)` — `x_1 - x_0` 방향의 velocity field (straight-line interpolation의 시간 미분)

세 parameterization은 `x_t`의 동일 지점에서 "다음 step이 어디로 갈지"를 예측한다. **차이 벡터 `pred_c1 - pred_c2`는 parameterization의 선형 변환에 대해 불변(invariant up to scaling)이다**. Cosine similarity는 scale invariant이므로, 서로 다른 parameterization에서 계산된 CAS 값이 (일반적으로 calibrate된 threshold 하에서) 동일한 의미를 갖는다. 이것이 SD3에서 threshold를 0.4로 내리면 되는 이유 — 분포 scale은 다르지만 방향성의 topology는 보존된다.

이 관찰은 Song & Kingma의 score-based 해석과 일맥상통한다. DDPM, v-pred, flow matching은 모두 **learned score `nabla_x log p_t(x|c)`의 변형**으로 해석 가능하며, conditional score의 차이 `score(x|c1) - score(x|c2)`는 **tangent space 상의 동일한 vector field**를 가리킨다 (cf. Ho & Salimans 2022, classifier-free guidance의 원 formulation).

#### CFG literature와의 연결

Classifier-free guidance (Ho & Salimans, 2022)의 표준 공식 `eps_cfg = eps_null + w * (eps_c - eps_null)`은 본질적으로 `(eps_c - eps_null)`을 "concept direction"으로 보고 그 방향으로 extrapolation하는 연산이다. SafeGen의 CAS는 **두 concept direction의 alignment**를 측정하는 것이고, HOW 컴포넌트의 anchor_inpaint는 **target direction을 anchor direction으로 교체하는 local rewrite**이다. 두 연산 모두 CFG가 발견한 "difference-as-direction"이라는 원칙 위에서 동작하므로, CFG가 동작하는 모든 backbone에서 자동으로 동작한다.

### 6.2 Attention Probe Challenges Across Architectures

WHERE 컴포넌트가 backbone별로 가장 큰 구현 편차를 보이는 이유는 attention 구조가 근본적으로 다르기 때문이다. 이를 "token → spatial 매핑의 recoverability"라는 관점에서 본다.

#### UNet cross-attention: clean token → spatial

SD v1.4의 `attn2`는 `Q = W_q @ x_img`, `K = W_k @ x_txt`로 명확히 **image query ↔ text key** 구조이다. Attention map `softmax(QK^T/sqrt(d))`의 shape `[H, S_img, S_txt]`는 그 자체로 "spatial position이 어떤 token을 attend하는지"에 대한 직접적 해석을 제공한다. Target token index만 알면 바로 2D mask를 얻을 수 있다 — 이것이 probe 구현이 가장 cheap한 이유.

#### MMDiT joint attention: sub-matrix extraction이 핵심

MMDiT는 image tokens과 text tokens을 concat한 후 single self-attention을 수행한다 (Esser et al., 2024, "Scaling Rectified Flow Transformers..."). Full attention matrix는 4개 block으로 분해된다:

```
attn = | A_ii   A_it |
       | A_ti   A_tt |
```

여기서 cross-attention에 해당하는 정보는 `A_it` (image queries attending to text keys)에만 있다. `A_ii`는 image self-attention이므로 text concept 정보를 담지 않는다. 따라서 probe는 반드시 `attn[:, :S_img, S_img:]` **sub-matrix를 명시적으로 slicing**해야 한다 (`attention_probe_sd3.py:224`).

추가 복잡성: (1) image/text에 별도 projection이 사용된다 (`to_q/k/v` vs `add_q_proj/k_proj/v_proj`), (2) RMSNorm이 Q/K에 적용된다, (3) SDPA kernel은 attention weights를 반환하지 않으므로 probe 활성 시 explicit `softmax(QK^T)`로 fallback해야 한다 (속도 손실 수반).

#### FLUX DiT: spatial recoverability가 가장 어려움

FLUX는 **packed latent**를 사용한다: `[B, 16, H, W]`이 2×2 patchify로 `[B, (H/2)*(W/2), 64]`가 된다. Attention 계산 시 image는 이미 1D sequence로 flatten되어 있으며, spatial 정보는 오직 `img_ids` (positional encoding tensor)에만 담긴다.

FLUX.1-dev는 **double-stream + single-stream** 이중 아키텍처를 사용한다 (Black Forest Labs, FLUX 기술 블로그 기준). Double-stream block은 SD3처럼 image/text가 별도 projection 후 joint attention이지만, single-stream block은 image와 text를 concat한 후 **공유 QKV projection**을 적용한다. 후자의 경우 `A_it` sub-matrix를 추출하더라도 image/text가 같은 embedding space로 projection된 후이므로, text token-level 의미 해석이 흐려진다.

FLUX에서 spatial probe를 구현하려면:
1. Double-stream block에서만 hook (semantic 해석이 가능한 layer)
2. `A_it = attn[:, :S_img, S_img:]` 추출
3. `S_img = (H/2)*(W/2)` 이므로 `img_ids`의 `(row, col)` 좌표로 2D grid 재구성
4. `F.interpolate`로 latent resolution에 맞추기
5. Packed sequence 내 spatial mask를 다시 packing 형식에 맞춰 broadcast

이론적으로 구현 가능하나, single-stream block이 전체 layer의 상당 부분을 차지하므로 (FLUX.1-dev는 19 double + 38 single), SD3 대비 probe signal-to-noise ratio가 낮을 가능성이 있다.

### 6.3 Embedded Guidance Paradox (FLUX.1-dev)

FLUX.1-dev는 CFG distillation을 학습 시 내재화했다 (guidance-distilled model). `guidance_scale`이 tensor로 transformer에 전달되며 (`transformer.config.guidance_embeds = True`), 단일 forward pass가 이미 guided prediction을 반환한다. 이로 인해 **true `eps_null`이 존재하지 않는다**는 수학적 tension이 발생한다.

#### Null prediction의 의미 왜곡

Traditional CFG 모델에서 `eps_null = eps_theta(x_t, t, c=empty)`는 unconditional denoising direction이다. FLUX.1-dev에서 `guidance=w` tensor를 전달한 채 `c=empty`로 호출하면, 반환되는 것은 "unconditional at guidance w"이다 — 이는 distillation target이 `eps_null + w * (eps_c - eps_null)`이었을 때의 `eps_null`과 **수학적으로 동일하지 않다**. 실질적으로는 `w=0` 또는 `w=1`로 설정해야 "순수 unconditional"에 가까워진다.

우리 구현 (`generate_flux1_v1.py:425-435`)은 null pass에서도 같은 `guidance_tensor`를 유지한다 — 이는 CAS 계산에 쓰이는 `(ep - en)`이 **"prompt 고유 direction"이 아니라 "prompt-vs-empty direction at fixed guidance"를 측정**함을 의미한다. 실험적으로 threshold 0.6에서 동작한다는 사실은 이 변형된 direction도 여전히 discriminative하다는 증거지만, SD v1.4와 이론적 등가성을 가지지는 않는다.

#### Anchor_inpaint formula의 간소화

이 때문에 FLUX.1-dev의 anchor_inpaint는:

```
out = ep * (1 - s*m) + ea * (s*m)    # ea는 이미 guided
```

전통적 backbone의 `eps_null + cfg * (eps_anchor - eps_null)` blending을 쓰면 **이중으로 guidance가 누적**된다 — `ea` 자체가 이미 `cfg` scale로 amplified되어 있기 때문이다. Trade-off:

- **장점**: 공식이 단순하고 anchor prediction pass가 `cfg` amplification 없이 그대로 사용 가능
- **단점**: `cfg_scale`로 anchor 강도를 fine-tune할 수 없음 — `safety_scale`만이 유일한 lever이므로 튜닝 자유도가 감소
- **결과**: FLUX.1-dev에서 SafeGen의 hyperparameter search space가 축소되며, 같은 SR을 달성하는 sweet spot이 좁을 수 있음

### 6.4 Implications for Generalization

#### Cross-backbone 성공의 해석

4개 backbone에서 동작한다는 사실이 "방법이 fundamental하다"를 의미하는가? 엄밀히는 **"현재 mainstream T2I family의 구조적 공통분모 위에서 동작한다"**가 더 정확한 주장이다. 공통분모:

1. Text-conditional iterative denoising (prompt-conditioned 반복 refinement)
2. CFG-compatible prediction (`pred_c - pred_null`이 concept direction을 의미)
3. Text encoder가 concept-discriminative embedding을 제공 (CLIP/T5/Mistral 모두)
4. Cross-modal attention이 존재 (형태는 달라도 text→image 정보 흐름이 있음)

#### SafeGen을 깨뜨릴 수 있는 backbone (가설)

- **Fully autoregressive T2I** (예: Parti-style 토큰 autoregressive): 반복적 denoising이 없으므로 `pred_c - pred_null`의 direction 해석 자체가 불가능. CAS 재정의 필요.
- **Consistency models / single-step distilled**: `x_t`에서 바로 `x_0`를 예측 — concept direction이 trajectory 차이가 아니라 endpoint 차이가 되어 cosine similarity의 의미가 달라짐. Threshold 재calibration은 필요하나, 잠재적으로 노이즈 영역에서 direction이 불안정할 수 있음.
- **Mixture-of-Experts routing**: Prompt별로 다른 expert가 활성화되면, `pred_c`와 `pred_null`이 다른 sub-network를 통과하여 `pred_c - pred_null` 차이가 concept direction이 아니라 routing artifact를 반영할 가능성.
- **Guidance-free trained models** (guidance distillation 극단 버전): `pred_null`과 `pred_c`의 차이가 학습 시 최소화되었다면 direction signal 자체가 약해짐. CAS가 거의 0에 가까워져 trigger되지 않을 수 있음.

반대로, **video diffusion**이나 **3D diffusion**은 공간 차원이 추가될 뿐 위 4가지 공통분모를 유지하므로, CAS는 그대로 동작하고 WHERE만 재설계하면 된다.

### 6.5 Future Directions

#### FLUX spatial probe (short-term)

가장 즉각적 확장. 6.2의 구현 경로를 따라 double-stream block에서 `A_it` sub-matrix를 추출하고 `img_ids`의 `(row, col)` coordinates로 2D reshape. 예상 결과: FLUX.1/2의 WHERE cost를 global → spatial로 upgrade하여 benign FP rate를 낮추고 harmful 영역의 localization을 개선.

#### Video diffusion으로의 확장

Stable Video Diffusion, Open-Sora 등은 **(T, C, H, W) latent**를 사용하며 temporal attention을 추가로 갖는다. SafeGen 확장 설계:
- **CAS**: temporal CAS (per-frame) + global spatiotemporal CAS (전체 sequence)
- **WHERE**: temporal-spatial mask — 어느 frame의 어느 영역이 unsafe인지
- **HOW**: temporal consistency를 깨지 않도록 anchor prediction도 동일 temporal context에서 sampling

핵심 난점은 **frame 간 mask consistency** — frame별 독립적 masking은 flickering을 야기하므로 optical flow 기반 mask propagation 또는 3D Gaussian smoothing이 필요.

#### Fine-tuning 방법과의 hybrid

SafeGen (training-free) + ESD/UCE/SA (fine-tuning) 조합은 각각의 약점을 상쇄할 수 있다:

- **Fine-tuning의 약점**: New backbone에 재학습 필요, benign quality 손상, 특정 concept에 과적합
- **SafeGen의 약점**: Global guidance만 가능한 backbone (FLUX)에서 FP rate 상대적으로 높음, hyperparameter sensitivity

**Hybrid 제안**: (1) Fine-tuning으로 model weights에 **coarse-level** unlearning을 주입하여 baseline harmful generation 확률을 낮추고, (2) SafeGen을 inference-time **safety net**으로 얹어 잔존 harmful case를 spatial masking으로 처리. 이 경우 fine-tuning 강도를 약하게 할 수 있어 benign quality 보존이 개선될 가능성.

#### 자동 calibration

CAS threshold, safety_scale, probe block index 등의 hyperparameter는 backbone별로 수동 tuning되어 왔다. **Calibration set 기반 auto-tuning** (e.g., FPR ≤ 5%를 constraint로 두고 TPR을 최대화하는 threshold를 binary search)은 새 backbone 이식 시 adaptation cost를 추가로 낮출 수 있으며, 논문 reproducibility 관점에서도 기여.

---

**정리**: SafeGen의 cross-backbone 성공은 "방법이 universal하다"가 아니라 "현재 T2I 생태계의 공통 구조 위에서 직교적으로 설계되었다"로 읽어야 한다. WHEN/HOW는 CFG 수학의 선형성에 기반하여 자동 이식되고, WHERE만이 attention 구조의 이질성으로 인해 per-backbone 엔지니어링을 요구한다. 다음 연구 단계는 (1) FLUX spatial probe, (2) video로의 확장, (3) fine-tuning과의 hybrid, (4) 자동 calibration이다.
