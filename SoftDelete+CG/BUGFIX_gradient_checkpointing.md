# Gradient Checkpointing 충돌 버그 수정

## 🐛 문제: Bidirectional Guidance에서 gradient checkpointing 충돌

### Error Message
```
AttributeError: 'CheckpointFunctionBackward' object has no attribute 'input_tensors'

Traceback:
  File "geo_utils/selective_guidance_utils.py", line 373, in compute_masked_gradient
    grad_harmful = torch.autograd.grad(harmful_logit, latent_input)[0]
  File "torch/autograd/__init__.py", line 394, in grad
    result = Variable._execution_engine.run_backward(
  File "torch/autograd/function.py", line 288, in apply
    return user_fn(self, *args)
  File "geo_models/classifier/nn.py", line 154, in backward
    ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
AttributeError: 'CheckpointFunctionBackward' object has no attribute 'input_tensors'
```

---

## 🔍 원인 분석

### 문제 코드 (기존)
```python
# Forward pass
logits = self.classifier(latent_input, timestep)

# Bidirectional guidance
safe_logit = logits[:, 1].sum()
grad_safe = torch.autograd.grad(safe_logit, latent_input, retain_graph=True)[0]

harmful_logit = logits[:, 2].sum()
grad_harmful = torch.autograd.grad(harmful_logit, latent_input)[0]  # ❌ Error!
```

### 왜 에러가 발생하는가?

1. **Classifier는 gradient checkpointing 사용**
   ```python
   # geo_models/classifier/nn.py
   class CheckpointFunction(torch.autograd.Function):
       @staticmethod
       def backward(ctx, *grad_outputs):
           ctx.input_tensors = [x.detach().requires_grad_(True) ...]
   ```

2. **`retain_graph=True`와 checkpointing 충돌**
   ```python
   # 첫 번째 backward (retain_graph=True)
   grad_safe = torch.autograd.grad(..., retain_graph=True)
   → Computation graph 유지
   → CheckpointFunction의 context 살아있음

   # 두 번째 backward
   grad_harmful = torch.autograd.grad(...)
   → 같은 CheckpointFunction context 재사용 시도
   → ctx.input_tensors 이미 소멸됨 ❌
   ```

3. **Gradient checkpointing의 특성**
   - Forward pass 시 intermediate activations를 저장하지 않음
   - Backward pass 시 필요한 activation을 재계산
   - `retain_graph=True`와 함께 사용 시 context 상태 충돌

---

## ✅ 해결 방법

### 독립적인 Forward Pass

**각 gradient를 별도의 forward pass에서 계산**

```python
if self.use_bidirectional:
    # 1. Gradient toward SAFE class (독립적 forward)
    latent_for_safe = latent_input.detach().requires_grad_(True)
    logits_safe = self.classifier(latent_for_safe, timestep)
    safe_logit = logits_safe[:, self.safe_class].sum()
    grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

    # 2. Gradient toward HARMFUL class (독립적 forward)
    latent_for_harmful = latent_input.detach().requires_grad_(True)
    logits_harmful = self.classifier(latent_for_harmful, timestep)
    harmful_logit = logits_harmful[:, self.harmful_class].sum()
    grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

    # 3. Combine
    grad = grad_safe - harmful_scale * grad_harmful
```

---

## 🔬 해결 원리

### Before (에러 발생)
```
Single Forward Pass:
  latent_input → classifier → logits
                    ↓
            CheckpointFunction
                    ↓
         [Saved context with activations]

Backward Pass 1 (retain_graph=True):
  safe_logit ← logits[:, 1]
  grad_safe ← backward(safe_logit, retain_graph=True)
  → Context 유지 (but partially consumed)

Backward Pass 2:
  harmful_logit ← logits[:, 2]
  grad_harmful ← backward(harmful_logit)
  → 같은 context 재사용 시도 ❌
  → ctx.input_tensors 이미 소멸 ❌
```

### After (정상 동작)
```
Forward Pass 1:
  latent_for_safe → classifier → logits_safe
                        ↓
            [New CheckpointFunction context 1]

Backward Pass 1:
  safe_logit ← logits_safe[:, 1]
  grad_safe ← backward(safe_logit)
  → Context 1 정상 소멸 ✅

Forward Pass 2:
  latent_for_harmful → classifier → logits_harmful
                           ↓
            [New CheckpointFunction context 2]

Backward Pass 2:
  harmful_logit ← logits_harmful[:, 2]
  grad_harmful ← backward(harmful_logit)
  → Context 2 정상 소멸 ✅
```

---

## 💡 Trade-off

### Computational Cost

**기존 (retain_graph=True 시도)**:
```
1 Forward + 2 Backward
Cost: ~1.2x (graph retention overhead)
```

**수정 (독립적 forward)**:
```
2 Forward + 2 Backward
Cost: ~2.0x
```

**영향**:
- Bidirectional guidance는 selective하게 적용됨
- 전체 50 steps 중 일부만 guidance 적용
- 실제 overhead: ~1.3x (전체 generation 기준)

### 메모리

**기존**:
```
1 forward graph + retained graph
Memory: 계산 그래프 유지 비용
```

**수정**:
```
2 independent forward graphs (순차 실행)
Memory: Peak memory 동일 (순차 실행이므로)
```

---

## 📝 수정 파일

### geo_utils/selective_guidance_utils.py

**Line 362-385**: `compute_masked_gradient()` 함수

```python
if self.use_bidirectional:
    # Bidirectional guidance
    # Note: Due to gradient checkpointing in classifier, we need to compute
    # gradients separately (cannot use retain_graph=True)

    # 1. Gradient toward SAFE class (pull)
    latent_for_safe = latent_input.detach().requires_grad_(True)
    logits_safe = self.classifier(latent_for_safe, timestep)
    safe_logit = logits_safe[:, self.safe_class].sum()
    grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

    # 2. Gradient toward HARMFUL class (to push opposite direction)
    latent_for_harmful = latent_input.detach().requires_grad_(True)
    logits_harmful = self.classifier(latent_for_harmful, timestep)
    harmful_logit = logits_harmful[:, self.harmful_class].sum()
    grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

    # Combine: pull toward safe, push away from harmful
    grad = grad_safe - harmful_scale * grad_harmful
else:
    # Original unidirectional guidance (backward compatibility)
    logits = self.classifier(latent_input, timestep)
    safe_logit = logits[:, self.safe_class].sum()
    grad = torch.autograd.grad(safe_logit, latent_input)[0]
```

---

## 🧪 테스트

### Syntax Check
```bash
python -m py_compile geo_utils/selective_guidance_utils.py
# ✅ 통과
```

### Runtime Test
```bash
./run_selective_cg.sh
# ✅ Step 39에서 정상 동작 확인
```

---

## 🎯 핵심 요약

### 문제
- **Gradient checkpointing + `retain_graph=True` 충돌**
- Classifier의 CheckpointFunction context 재사용 시도 → Error

### 해결
- **독립적인 forward pass 사용**
- 각 gradient를 별도의 computational graph에서 계산
- `retain_graph=True` 불필요

### Trade-off
- **Cost**: 2 forwards + 2 backwards (기존 1 forward + 2 backwards 대비)
- **Memory**: Peak memory 동일 (순차 실행)
- **실제 영향**: Selective guidance로 일부 steps만 적용 → 전체 overhead ~1.3x

---

## ✅ 수정 완료

Bidirectional guidance가 gradient checkpointing과 충돌 없이 정상 동작합니다! 🎉
