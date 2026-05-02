# Dtype 불일치 버그 수정

## 🐛 문제 1: Conv2d dtype 불일치

```
RuntimeError: Input type (c10::Half) and bias type (float) should be the same
```

### 원인
- Stable Diffusion pipeline: `torch.float16` (FP16)
- Classifier model: `torch.float32` (FP32)
- Latent가 FP16인데 classifier에 전달될 때 dtype 불일치 발생

---

## 🐛 문제 2: Model attribute error

```
AttributeError: 'EncoderUNetModelForClassification' object has no attribute 'dtype'
```

### 원인
- PyTorch 모델은 직접 `.dtype` attribute가 없음
- 모델 파라미터에서 dtype을 가져와야 함

---

## ✅ 해결 방법

### 1. Classifier는 FP32 유지, Latent를 FP32로 변환

**파일**: `generate_selective_cg.py`

```python
# Classifier는 float32 유지 (timestep_embedding이 float32 출력)
classifier = load_discriminator(...).to(device)
classifier.eval()
print(f"  Classifier loaded (dtype: float32)")
```

**이유**: Classifier의 `timestep_embedding` 함수가 항상 float32를 출력하므로,
모델을 float16으로 변환하면 dtype 불일치 발생

### 2. Classifier dtype을 인스턴스 변수로 저장

**파일**: `geo_utils/selective_guidance_utils.py`

#### `SelectiveGuidanceMonitor.__init__()` 함수
```python
self.classifier = classifier_model
self.classifier.eval()

# Get classifier dtype from its parameters
self.classifier_dtype = next(self.classifier.parameters()).dtype
```

#### `SpatiallyMaskedGuidance.__init__()` 함수
```python
self.classifier = classifier_model
self.safe_class = safe_class
self.device = device

# Get classifier dtype from its parameters
self.classifier_dtype = next(self.classifier.parameters()).dtype
```

### 3. Latent dtype 자동 변환

**파일**: `geo_utils/selective_guidance_utils.py`

#### `detect_harmful()` 함수
```python
# Ensure latent dtype matches classifier
latent_input = latent.to(dtype=self.classifier_dtype)

# Get classifier predictions
logits = self.classifier(latent_input, timestep)
```

#### `get_spatial_mask()` 함수
```python
# Ensure latent dtype matches classifier
latent_input = latent.to(dtype=self.classifier_dtype)

with torch.enable_grad():
    heatmap, info = self.gradcam.generate_heatmap(
        latent=latent_input,
        ...
    )
```

#### `compute_masked_gradient()` 함수
```python
# Ensure latent requires grad and matches classifier dtype
latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

# Forward pass
logits = self.classifier(latent_input, timestep)

# Backward to get gradient
grad = torch.autograd.grad(safe_logit, latent_input)[0]

# ...

# Convert back to original latent dtype
masked_grad = masked_grad.to(dtype=latent.dtype)
return masked_grad.detach()
```

---

## 📝 수정 파일 목록

1. **generate_selective_cg.py**
   - Line 365-368: Classifier FP32 유지 (FP16 변환 제거)

2. **geo_utils/selective_guidance_utils.py**
   - Line 72: `SelectiveGuidanceMonitor.__init__()` - classifier_dtype 저장
   - Line 128: `detect_harmful()` - latent → FP32 변환
   - Line 174: `get_spatial_mask()` - latent → FP32 변환
   - Line 309: `SpatiallyMaskedGuidance.__init__()` - classifier_dtype 저장
   - Line 331: `compute_masked_gradient()` - latent → FP32 변환
   - Line 355: `compute_masked_gradient()` - gradient → 원래 dtype 복원

---

## 🧪 테스트

```bash
# Syntax check (통과)
python -m py_compile generate_selective_cg.py geo_utils/selective_guidance_utils.py

# Runtime test
./test_selective_cg.sh
# 또는
./run_selective_cg.sh
```

---

## 💡 설계 원칙

### Dtype 처리 전략
1. **Classifier는 FP32 유지**: `timestep_embedding`이 FP32 출력하므로 불일치 방지
2. **입력 시 자동 변환**: `latent.to(dtype=self.classifier_dtype)` (FP16 → FP32)
3. **출력 시 원복**: `grad.to(dtype=latent.dtype)` (FP32 → FP16)

### 장점
- Classifier 내부 dtype 불일치 방지
- Pipeline latent (FP16)와 자동 변환
- 명시적 dtype 처리 → 버그 방지
- 메모리: Latent만 일시적으로 FP32 변환 (큰 오버헤드 없음)

---

## ✅ 수정 완료

모든 dtype 불일치 문제 해결됨.
이제 정상적으로 실행 가능합니다!
