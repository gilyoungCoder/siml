# Classifier Interpretation Guide

이 가이드는 nudity classifier가 어떤 latent 영역을 기반으로 unsafe(nude) 판단을 내리는지 해석하는 방법을 설명합니다.

## 🎯 목표

- **Grad-CAM**: Classifier가 nude를 판단할 때 주목하는 latent 영역을 heatmap으로 시각화
- **Layer-wise Analysis**: U-Net의 각 layer에서 어떤 feature가 활성화되는지 분석
- **Integrated Gradients**: 입력의 어떤 부분이 prediction에 기여하는지 정량적 측정
- **Generation Analysis**: 생성 과정 중 어느 시점에 nude feature가 나타나는지 추적

## 📁 파일 구조

```
SoftDelete+CG/
├── geo_utils/
│   └── classifier_interpretability.py    # 해석 도구 모듈
├── interpret_classifier.py               # 메인 실행 스크립트
├── examples/
│   └── quick_interpret.py               # 간단한 예제
└── INTERPRETATION_README.md             # 이 파일
```

## 🚀 Quick Start

### 1. 단일 이미지 분석

이미 생성된 이미지를 분석하여 classifier가 어디를 보는지 확인:

```bash
python interpret_classifier.py \
    --mode image \
    --image_path ./test_images/sample.png \
    --output_dir ./interpretations/sample \
    --timestep 500
```

**출력:**
- `gradcam.png`: Grad-CAM heatmap overlay
- `layers.png`: Layer-wise activation statistics
- `integrated_gradients.png`: IG attribution per channel
- `summary.json`: 수치 결과 요약

### 2. 디렉토리 전체 분석

여러 이미지를 배치 처리:

```bash
python interpret_classifier.py \
    --mode image \
    --image_dir ./generated_images \
    --output_dir ./interpretations/batch
```

### 3. 생성 과정 분석

이미지 생성 중 step-by-step으로 classifier attention 추적:

```bash
python interpret_classifier.py \
    --mode generation \
    --prompt "a person at the beach" \
    --output_dir ./interpretations/generation \
    --num_steps 50
```

**출력:**
- `step_000.png`, `step_010.png`, ...: 각 step의 heatmap
- `heatmap_evolution.gif`: Heatmap 변화 애니메이션
- `probability_evolution.png`: 각 class 확률 변화 그래프
- `prediction_trajectory.json`: Step별 prediction 데이터

### 4. 빠른 예제 실행

간단한 테스트용:

```bash
cd examples
python quick_interpret.py
```

## 📊 해석 방법

### Grad-CAM Heatmap 읽기

생성된 `gradcam.png`를 열면:

```
┌─────────────────────────────────────┐
│  Latent Ch 0  │  Latent Ch 1  │ ... │  원본 latent channels
├─────────────────────────────────────┤
│  🔥 Heatmap   │  Overlay      │ Bar │  해석 결과
└─────────────────────────────────────┘
```

**색상 의미:**
- 🔴 **빨강/밝음**: 이 영역이 nude 판단에 **중요**
- 🔵 **파랑/어두움**: 이 영역은 nude 판단에 **덜 중요**

**예시 해석:**
```python
# summary.json에서
{
  "gradcam": {
    "max_attention": 0.987,      # 최대 attention 강도
    "mean_attention": 0.234,     # 평균 attention
    "top_10_percent_mean": 0.876 # 상위 10% 영역의 평균
  }
}
```

→ `max_attention`이 높고 특정 영역에 집중되어 있으면, classifier가 그 부분에서 nude feature를 강하게 감지

### Layer-wise Activation

`layers.png`는 U-Net의 각 layer별 activation을 보여줍니다:

```
input_blocks.0  →  초기 low-level feature (edges, textures)
input_blocks.3  →  중간 feature (parts, shapes)
input_blocks.6  →  후기 feature (semantic parts)
───────────────────────────────────────────────────
output_blocks.0 →  초기 upsampling (reconstruction start)
output_blocks.2 →  🎯 High-level semantic (nude concepts)
output_blocks.5 →  최종 refinement
```

**해석:**
- `output_blocks.2`에서 특정 channel의 activation이 높다 → 해당 channel이 nude 판단에 핵심 역할

### Integrated Gradients

`integrated_gradients.png`는 각 latent channel의 기여도를 보여줍니다:

```
Channel 0: ████████████ (중요도 높음)
Channel 1: ████         (중간)
Channel 2: ██████████   (높음)
Channel 3: ██           (낮음)
```

**활용:**
- 어떤 channel이 nude detection에 가장 중요한지 파악
- Unlearning 시 특정 channel에 집중할 수 있음

### Generation Analysis

`probability_evolution.png` 그래프:

```
확률
1.0 ┤
    │          ╱─────  Nude (target)
0.5 ┤      ╱───
    │  ───╯
0.0 ┤──────────────── Not People
    └────────────────→ Step
    0   10   20   30
```

**해석 포인트:**
- **Nude 확률이 급격히 증가하는 step**: 이 시점에 nude concept이 latent에 나타남
- **Heatmap 변화**: 어떤 영역에서 nude feature가 발현하는지 시각적으로 확인

## 🔬 심화 사용법

### 1. 다른 Layer에서 Grad-CAM 수행

```python
from geo_utils.classifier_interpretability import ClassifierGradCAM

# Early layer (low-level features)
gradcam_early = ClassifierGradCAM(classifier, target_layer_name="input_blocks.3")

# Mid layer
gradcam_mid = ClassifierGradCAM(classifier, target_layer_name="output_blocks.0")

# Late layer (high-level semantic)
gradcam_late = ClassifierGradCAM(classifier, target_layer_name="output_blocks.2")
```

**추천:**
- `output_blocks.2`: 전반적인 semantic 이해 (기본값)
- `output_blocks.0`: 초기 reconstruction 단계
- `input_blocks.6`: 인코딩 마지막 단계

### 2. Custom Timestep 분석

Diffusion의 다른 시점에서 분석:

```bash
# Early denoising (큰 noise, high-level structure)
python interpret_classifier.py --mode image --image_path ./img.png --timestep 900

# Mid denoising (중간 detail)
python interpret_classifier.py --mode image --image_path ./img.png --timestep 500

# Late denoising (세밀한 detail)
python interpret_classifier.py --mode image --image_path ./img.png --timestep 100
```

### 3. Attention Manipulation과 연결

Generation 중 attention suppression이 잘 작동하는지 확인:

```bash
# 1. Baseline: Suppression 없이 생성
python generate_adaptive.py --prompt "person" --tau 999 --output baseline.png

# 2. Interpretation
python interpret_classifier.py --mode image --image_path baseline.png --output_dir ./interp_baseline

# 3. With suppression
python generate_adaptive.py --prompt "person" --tau 0.35 --output suppressed.png

# 4. Compare interpretation
python interpret_classifier.py --mode image --image_path suppressed.png --output_dir ./interp_suppressed
```

**비교 포인트:**
- Suppression 후 Grad-CAM heatmap이 약해졌는가?
- Nude 확률이 감소했는가?
- 어떤 latent 영역이 변화했는가?

### 4. 프로그래밍 방식 사용

```python
import torch
from geo_utils.classifier_interpretability import (
    ClassifierGradCAM,
    load_classifier_for_interpretation
)

# Load
classifier = load_classifier_for_interpretation(
    "./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
)

# Prepare inputs
latent = ...  # [1, 4, 64, 64]
timestep = torch.tensor([500])

# Run Grad-CAM
gradcam = ClassifierGradCAM(classifier)
heatmap, info = gradcam.generate_heatmap(
    latent, timestep, target_class=2  # 2 = nude
)

# Access results
nude_prob = info['probs'][0, 2].item()
attention_map = heatmap[0].cpu().numpy()

print(f"Nude probability: {nude_prob:.3f}")
print(f"Max attention region: {attention_map.max():.3f}")

# Find top attention locations
top_indices = torch.topk(heatmap[0].flatten(), k=10).indices
top_coords = [(idx // 64, idx % 64) for idx in top_indices]
print(f"Top attention locations: {top_coords}")
```

## 📈 실험 워크플로우 예시

### 실험 1: Suppression 효과 검증

```bash
# 1. Baseline generation (no suppression)
python generate_adaptive.py \
    --prompt_file harmful_prompts.txt \
    --tau 999 \
    --output_dir ./exp1_baseline

# 2. Interpret baseline
python interpret_classifier.py \
    --mode image \
    --image_dir ./exp1_baseline \
    --output_dir ./exp1_baseline_interp

# 3. With suppression
python generate_adaptive.py \
    --prompt_file harmful_prompts.txt \
    --tau 0.35 \
    --gamma 5.0 \
    --output_dir ./exp1_suppressed

# 4. Interpret suppressed
python interpret_classifier.py \
    --mode image \
    --image_dir ./exp1_suppressed \
    --output_dir ./exp1_suppressed_interp

# 5. Compare results
python -c "
import json
from pathlib import Path

baseline = list(Path('./exp1_baseline_interp').glob('*/summary.json'))
suppressed = list(Path('./exp1_suppressed_interp').glob('*/summary.json'))

print('Baseline Nude Probs:')
for f in baseline:
    data = json.load(open(f))
    print(f'  {data[\"predictions\"][\"probs\"][2]:.3f}')

print('\\nSuppressed Nude Probs:')
for f in suppressed:
    data = json.load(open(f))
    print(f'  {data[\"predictions\"][\"probs\"][2]:.3f}')
"
```

### 실험 2: Critical Step 찾기

언제 nude concept이 생성되는지 찾기:

```bash
python interpret_classifier.py \
    --mode generation \
    --prompt "a person" \
    --num_steps 50 \
    --output_dir ./critical_step_analysis

# Check prediction_trajectory.json
python -c "
import json
data = json.load(open('./critical_step_analysis/prediction_trajectory.json'))

nude_probs = [(d['step'], d['probs'][2]) for d in data]
sorted_by_change = sorted(enumerate(nude_probs[1:], 1),
                          key=lambda x: x[1][1] - nude_probs[x[0]-1][1],
                          reverse=True)

print('Top 5 steps with largest nude probability increase:')
for idx, (step, prob) in sorted_by_change[:5]:
    prev_prob = nude_probs[idx-1][1]
    print(f'  Step {step}: {prev_prob:.3f} → {prob:.3f} (Δ {prob-prev_prob:.3f})')
"
```

## 🔧 트러블슈팅

### Issue: "Layer not found"

```
⚠ Warning: Layer 'output_blocks.2' not found!
```

**해결:**
```python
# 사용 가능한 layer 확인
from geo_models.classifier.classifier import load_discriminator

classifier = load_discriminator(...)
for name, _ in classifier.named_modules():
    if 'output_blocks' in name or 'input_blocks' in name:
        print(name)
```

### Issue: CUDA out of memory

**해결:**
```bash
# 이미지 크기 조절 또는 CPU 사용
python interpret_classifier.py --device cpu ...

# 또는 batch size 1로 제한 (이미 기본값)
```

### Issue: Hooks not capturing gradients

**원인:** `requires_grad=False`인 latent

**해결:** `classifier_interpretability.py`에서 자동으로 처리됨:
```python
latent = latent.clone().requires_grad_(True)
```

## 📚 참고 자료

### Grad-CAM 논문
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017

### Integrated Gradients
- Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017

### 관련 코드
- Classifier 정의: [geo_models/classifier/encoder_unet_model.py](geo_models/classifier/encoder_unet_model.py)
- Gradient guidance: [geo_utils/gradient_model_utils.py](geo_utils/gradient_model_utils.py)
- Attention suppression: [generate_adaptive.py](generate_adaptive.py)

## 🎯 다음 단계

### 1. Token Attention과 연결

Cross-attention과 classifier attention을 비교:

```python
# TODO: 구현 예정
# - Cross-attention map 추출 (from U-Net)
# - Classifier attention map과 correlation 분석
# - 어떤 text token이 nude region과 연관되는지 확인
```

### 2. Temporal Analysis

여러 timestep에서의 attention 변화 추적:

```python
for t in [100, 300, 500, 700, 900]:
    heatmap, info = gradcam.generate_heatmap(latent, t, target_class=2)
    # Analyze evolution
```

### 3. Concept Vectors

Harmful concept vector와 classifier attention의 관계:

```python
# Load harm vector from generate_adaptive.py
harm_vector = build_harm_vector(pipe, harmful_concepts)

# Compare with Grad-CAM attention
# Where do they align?
```

## 💡 유용한 팁

1. **여러 layer를 비교**: 다른 layer에서 heatmap을 생성하여 어느 level에서 nude feature가 잡히는지 확인
2. **Timestep sweep**: 다양한 timestep에서 분석하여 언제 concept이 형성되는지 파악
3. **Batch processing**: 많은 이미지를 분석하여 통계적 패턴 찾기
4. **시각화 customize**: `VisualizationUtils` 클래스를 수정하여 원하는 형태로 시각화

## 📧 문의

이슈나 질문이 있으면 코드 주석을 참고하거나 실험 결과를 공유해주세요!
