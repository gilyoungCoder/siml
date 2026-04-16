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
