# Bidirectional Classifier Guidance

## 🎯 개념

### 기존 방식 (Unidirectional)
```python
# Pull toward SAFE only
grad_safe = ∂(safe_logit) / ∂(latent)
latent += guidance_scale * grad_safe
```

### 새로운 방식 (Bidirectional) ⭐
```python
# Pull toward SAFE + Push away from HARMFUL
grad_safe = ∂(safe_logit) / ∂(latent)
grad_harmful = ∂(harmful_logit) / ∂(latent)

grad_combined = grad_safe - harmful_scale * grad_harmful
latent += guidance_scale * grad_combined
```

---

## 🔬 원리

### 1. Safe Direction (Pull)
```python
safe_logit = logits[:, 1]  # Clothed class
grad_safe = ∂(safe_logit) / ∂(latent)
```

**의미**: "Latent를 어떻게 바꿔야 clothed로 분류되는가?"

### 2. Harmful Direction (Push)
```python
harmful_logit = logits[:, 2]  # Nude class
grad_harmful = ∂(harmful_logit) / ∂(latent)
```

**의미**: "Latent를 어떻게 바꿔야 nude로 분류되는가?"

### 3. Combined Gradient
```python
grad = grad_safe - harmful_scale * grad_harmful
```

**해석**:
- `grad_safe`: Clothed 방향으로 당기기 ➕
- `- harmful_scale * grad_harmful`: Nude 방향에서 밀어내기 ➖
- **합**: "Clothed로 가면서 + Nude에서 멀어지기"

---

## 💡 수학적 직관

### Vector Space 관점

```
Latent Space (3-class):

  [Not People]
       |
       |
  [Clothed] ←------ Current Latent -----→ [Nude]
   (safe)                                  (harmful)
```

**Unidirectional**:
```
grad_safe: Current → Clothed
Result: 한쪽 방향으로만 이동
```

**Bidirectional**:
```
grad_safe: Current → Clothed (pull)
-grad_harmful: Current ← Nude (push away)

Result: 양쪽에서 힘이 가해짐
→ 더 강력하고 명확한 방향성
```

---

## 📊 기대 효과

### 1. **더 강한 Suppression**

**Unidirectional**:
```
grad_magnitude ≈ ||grad_safe||
```

**Bidirectional**:
```
grad_magnitude ≈ ||grad_safe - grad_harmful||
                ≥ ||grad_safe||  (일반적으로)
```

**결과**: 더 강력한 guidance 효과

### 2. **더 명확한 Decision Boundary**

```
Unidirectional:
  - Safe 방향만 강조
  - Decision boundary가 모호할 수 있음

Bidirectional:
  - Safe 방향 강조 + Harmful 방향 회피
  - Decision boundary가 명확
  - Ambiguous region 감소
```

### 3. **Robust to Initialization**

```
Unidirectional:
  - Latent가 애매한 위치에 있으면 효과 약함

Bidirectional:
  - Push + Pull 양방향 힘
  - 더 robust한 guidance
```

---

## ⚙️ 사용 방법

### 1. Shell Script 설정

```bash
# run_selective_cg.sh

# Enable bidirectional guidance
USE_BIDIRECTIONAL=true

# Harmful repulsion scale (relative to guidance_scale)
HARMFUL_SCALE=1.0
```

**HARMFUL_SCALE 파라미터**:
- `0.0`: Bidirectional 끄기 (unidirectional과 동일)
- `0.5`: 약한 repulsion (safe pull이 주도)
- `1.0`: **Equal weight (권장)** ⭐
- `2.0`: 강한 repulsion (harmful push가 주도)

### 2. Python 직접 실행

```bash
python generate_selective_cg.py CompVis/stable-diffusion-v1-4 \
  --prompt_file prompts/harmful.txt \
  --selective_guidance \
  --use_bidirectional \
  --guidance_scale 5.0 \
  --harmful_scale 1.0
```

### 3. 코드 레벨

```python
# Initialize guidance module
guidance_module = SpatiallyMaskedGuidance(
    classifier_model=classifier,
    safe_class=1,       # Clothed
    harmful_class=2,    # Nude
    use_bidirectional=True  # Enable bidirectional
)

# Apply guidance
guided_latent = guidance_module.apply_guidance(
    latent=latent,
    timestep=timestep,
    spatial_mask=mask,
    guidance_scale=5.0,
    harmful_scale=1.0   # Repulsion strength
)
```

---

## 🧪 실험 권장 사항

### Baseline Comparison

```bash
# 1. Unidirectional (baseline)
USE_BIDIRECTIONAL=false
GUIDANCE_SCALE=5.0

# 2. Bidirectional - Equal weight
USE_BIDIRECTIONAL=true
GUIDANCE_SCALE=5.0
HARMFUL_SCALE=1.0

# 3. Bidirectional - Strong repulsion
USE_BIDIRECTIONAL=true
GUIDANCE_SCALE=5.0
HARMFUL_SCALE=2.0

# Compare:
# - NSFW detection rate (safety)
# - GENEVAL score (quality)
# - Visual inspection
```

### Parameter Sweep

| HARMFUL_SCALE | Expected Effect | Use Case |
|---------------|----------------|----------|
| 0.0 | = Unidirectional | Baseline |
| 0.5 | Gentle repulsion | Conservative |
| **1.0** | **Equal weight** | **Balanced (권장)** |
| 1.5 | Strong repulsion | Aggressive |
| 2.0 | Very strong | Maximum safety |

---

## 🔍 내부 동작

### compute_masked_gradient()

```python
def compute_masked_gradient(self, latent, timestep, spatial_mask,
                           guidance_scale=5.0, harmful_scale=1.0):
    with torch.enable_grad():
        logits = classifier(latent, timestep)  # [B, 3]

        if self.use_bidirectional:
            # 1. Gradient toward SAFE
            safe_logit = logits[:, 1].sum()
            grad_safe = torch.autograd.grad(
                safe_logit, latent, retain_graph=True
            )[0]

            # 2. Gradient toward HARMFUL
            harmful_logit = logits[:, 2].sum()
            grad_harmful = torch.autograd.grad(
                harmful_logit, latent
            )[0]

            # 3. Combine: pull + push
            grad = grad_safe - harmful_scale * grad_harmful
        else:
            # Unidirectional (backward compatibility)
            safe_logit = logits[:, 1].sum()
            grad = torch.autograd.grad(safe_logit, latent)[0]

    # Apply spatial mask
    masked_grad = grad * spatial_mask.unsqueeze(1)

    # Scale
    masked_grad = masked_grad * guidance_scale

    return masked_grad
```

---

## 📈 예상 결과

### Harmful Prompt: "nude person"

**Unidirectional**:
```
Step 10:
  grad_safe magnitude: 0.05
  → latent += 5.0 * 0.05 = 0.25 shift

NSFW rate: 10%
```

**Bidirectional (harmful_scale=1.0)**:
```
Step 10:
  grad_safe magnitude: 0.05
  grad_harmful magnitude: 0.03
  combined: 0.05 - 1.0*0.03 = 0.08
  → latent += 5.0 * 0.08 = 0.40 shift

NSFW rate: 5% (↓ 50% reduction)
```

### Benign Prompt: "person reading a book"

**두 방식 모두 동일**:
```
harmful_score < threshold
→ Guidance 건너뛰기
→ No difference

GENEVAL: High (both)
```

---

## ⚠️ 주의사항

### 1. Computational Cost

**Bidirectional**은 **2배 backward pass** 필요:
```python
grad_safe = torch.autograd.grad(safe_logit, latent, retain_graph=True)
grad_harmful = torch.autograd.grad(harmful_logit, latent)
```

**Impact**:
- Unidirectional: 1 backward per step
- Bidirectional: 2 backwards per step
- **Total cost: ~1.5x** (gradient 계산만 증가, forward는 동일)

### 2. Gradient Conflict

만약 `grad_safe`와 `grad_harmful`이 반대 방향이 아니라면:
```python
# Bad case:
grad_safe = [+1, +2]
grad_harmful = [+0.5, +1]  # 같은 방향!

combined = [+1, +2] - [+0.5, +1] = [+0.5, +1]
→ 효과 감소
```

**해결**: 일반적으로 3-class classifier에서는 문제 없음 (서로 반대 방향)

### 3. Hyperparameter Sensitivity

`harmful_scale`이 너무 크면:
```
harmful_scale = 10.0
→ grad_harmful 지배
→ Safe로 가는 대신 Harmful에서 도망만 감
→ Not People class로 갈 수 있음 ⚠️
```

**권장**: `harmful_scale ∈ [0.5, 2.0]`

---

## 🎯 핵심 요약

### Bidirectional Guidance란?

**"Safe로 당기면서 + Harmful에서 밀어내는" 양방향 guidance**

```python
grad = grad_safe - harmful_scale * grad_harmful
```

### 장점
1. ✅ **더 강한 suppression**: 양방향 힘
2. ✅ **명확한 decision boundary**: Safe/Harmful 명확히 구분
3. ✅ **Robust guidance**: Ambiguous cases에서도 효과적

### 단점
1. ⚠️ **Computational cost**: 1.5x (2 backward passes)
2. ⚠️ **Hyperparameter tuning**: `harmful_scale` 조정 필요

### 권장 설정
```bash
USE_BIDIRECTIONAL=true
HARMFUL_SCALE=1.0  # Equal weight
```

---

## 🚀 다음 단계

### 1. Quick Test
```bash
# Bidirectional vs Unidirectional 비교
./test_selective_cg.sh  # 수정: USE_BIDIRECTIONAL=true
```

### 2. Full Experiment
```bash
# Harmful prompts에서 테스트
./run_selective_cg.sh
```

### 3. Evaluation
- **NSFW Detection**: Nudenet으로 측정
- **Quality**: GENEVAL score
- **Comparison**: Unidirectional baseline 대비

---

**구현 완료!** ✅

이제 양방향 guidance로 더 강력한 suppression이 가능합니다! 🎉
