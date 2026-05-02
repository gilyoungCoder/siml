# 🚀 Classifier Interpretation - Quick Start Guide

## 📋 필요한 입력 (Inputs)

### ✅ 이미 준비된 것들:
- **Classifier 체크포인트**: `./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth` ✓
- **SD 모델**: CompVis/stable-diffusion-v1-4 (Hugging Face에서 자동 다운로드) ✓
- **테스트 이미지들**:
  - `./img/CNBWON/*.png` (50개 이미지) ✓
  - `./img/country nude body, fully clothed/*.png` ✓

### 🎯 필요한 것:
- 분석할 이미지 경로 (이미 있음!)
- 또는 생성할 프롬프트 (텍스트만 입력)

---

## 🎬 사용법 (3가지 모드)

### Mode 1️⃣: 단일 이미지 분석

**가장 간단한 시작! 이미지 하나 분석하기**

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
./interpret_single_image.sh
```

**설정 변경 (스크립트 내부):**
```bash
# 다른 이미지로 바꾸려면:
vim interpret_single_image.sh

# 이 줄을 수정:
IMAGE_PATH="./img/CNBWON/000000.png"

# 다른 timestep으로 바꾸려면:
TIMESTEP=500  # 100 (late), 500 (mid), 900 (early)
```

**출력 결과:**
```
./interpretation_results/single_000000/
├── 000000_gradcam.png              ← 🔥 Grad-CAM 히트맵
├── 000000_layers.png               ← Layer별 activation
├── 000000_integrated_gradients.png ← IG attribution
└── 000000_summary.json             ← 수치 결과
```

---

### Mode 2️⃣: 배치 이미지 분석

**여러 이미지를 한 번에 분석**

```bash
./interpret_batch_images.sh
```

**설정 변경:**
```bash
vim interpret_batch_images.sh

# 이 줄을 수정:
IMAGE_DIR="./img/CNBWON"  # 50개 이미지 분석
# 또는
IMAGE_DIR="./img/country nude body, fully clothed"
```

**출력 결과:**
```
./interpretation_results/batch_CNBWON/
├── 000000/
│   ├── 000000_gradcam.png
│   ├── 000000_layers.png
│   ├── 000000_integrated_gradients.png
│   └── 000000_summary.json
├── 000001/
│   └── ...
└── ...

+ 자동 통계 출력:
  - Average nude probability
  - Predicted class distribution
```

---

### Mode 3️⃣: 생성 과정 분석 ⭐

**이미지 생성 중 step-by-step 분석 (가장 흥미로운 모드!)**

```bash
./interpret_generation.sh
```

**설정 변경:**
```bash
vim interpret_generation.sh

# 이 줄을 수정:
PROMPT="a person at the beach"
# 또는
PROMPT="nude body"
PROMPT="portrait of a person"
```

**출력 결과:**
```
./interpretation_results/generation_a_person_at_the_beach/
├── final_image.png                  ← 생성된 최종 이미지
├── step_000.png                     ← Step 0에서의 히트맵
├── step_010.png
├── step_020.png
├── ...
├── heatmap_evolution.gif            ← 🎬 히트맵 변화 애니메이션
├── probability_evolution.png        ← 📊 확률 변화 그래프
└── prediction_trajectory.json       ← 전체 step 데이터

+ 자동 분석:
  - Critical step: nude 확률이 급증하는 시점
  - Final nude probability
```

---

## 🎯 실제 예시

### 예시 1: 기본 이미지 분석
```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# 그냥 실행!
./interpret_single_image.sh

# 결과 보기
eog ./interpretation_results/single_000000/*.png
```

### 예시 2: 다른 이미지로 변경
```bash
# 스크립트 수정
vim interpret_single_image.sh
# IMAGE_PATH="./img/CNBWON/000010.png"  ← 이렇게 변경

# 실행
./interpret_single_image.sh

# 결과 JSON 확인
cat ./interpretation_results/single_000010/000010_summary.json | jq
```

### 예시 3: 배치 분석
```bash
# 50개 이미지 한 번에 분석
./interpret_batch_images.sh

# 모든 히트맵 보기
find ./interpretation_results/batch_CNBWON -name "*_gradcam.png" | xargs eog
```

### 예시 4: 생성 과정 분석 (추천!)
```bash
# 프롬프트 수정
vim interpret_generation.sh
# PROMPT="a nude person"  ← 이렇게 변경

# 실행 (약 2분 소요)
./interpret_generation.sh

# 애니메이션 보기
eog ./interpretation_results/generation_*/heatmap_evolution.gif

# 그래프 보기
eog ./interpretation_results/generation_*/probability_evolution.png
```

---

## 📊 결과 해석

### Grad-CAM 히트맵 읽기

```
🔴 빨간색/밝은 영역 = Classifier가 nude 판단에 중요하게 본 부분
🔵 파란색/어두운 영역 = 덜 중요한 부분
```

### JSON 결과 읽기

```json
{
  "predictions": {
    "probs": [0.05, 0.23, 0.72],  // [not_people, clothed, nude]
    "predicted_class": 2,          // 2 = nude
    "predicted_class_name": "Nude"
  },
  "gradcam": {
    "max_attention": 0.987,        // 최대 attention 강도
    "mean_attention": 0.234,       // 평균
    "top_10_percent_mean": 0.876   // 상위 10% 영역의 평균
  },
  "integrated_gradients": {
    "channel_importance": {
      "channel_0": 245.3,          // 이 channel이 가장 중요!
      "channel_1": 87.2,
      "channel_2": 156.7,
      "channel_3": 34.1
    }
  }
}
```

---

## 🔧 트러블슈팅

### ❌ CUDA out of memory
```bash
# CPU로 실행
vim interpret_single_image.sh
# 스크립트 마지막의 --device cuda를 --device cpu로 변경
```

### ❌ Image not found
```bash
# 사용 가능한 이미지 확인
ls ./img/CNBWON/*.png | head
ls ./img/country\ nude\ body,\ fully\ clothed/*.png | head

# 스크립트에서 IMAGE_PATH 수정
```

### ❌ Model download 오류
```bash
# Hugging Face 로그인 (처음 한 번만)
huggingface-cli login
# 또는
export HF_TOKEN="your_token"
```

---

## 💡 유용한 팁

### 1. 여러 timestep 비교
```bash
# 3개의 다른 timestep에서 분석
for t in 100 500 900; do
    sed -i "s/TIMESTEP=.*/TIMESTEP=$t/" interpret_single_image.sh
    ./interpret_single_image.sh
done
```

### 2. Suppression 전후 비교
```bash
# 1. Baseline 생성 (suppression 없음)
vim generate_adaptive.sh
# HARM_SUPPRESS=false로 설정
./generate_adaptive.sh

# 2. Baseline 분석
IMAGE_PATH="./output_dir/image.png"  # 생성된 이미지 경로
./interpret_single_image.sh

# 3. Suppression 활성화
vim generate_adaptive.sh
# HARM_SUPPRESS=true로 설정
./generate_adaptive.sh

# 4. 비교
./interpret_single_image.sh
# → 히트맵이 약해졌는가?
```

### 3. Critical step 찾기
```bash
# Generation mode로 실행
./interpret_generation.sh

# JSON에서 nude 확률 추출
cat ./interpretation_results/generation_*/prediction_trajectory.json | \
  jq '.[] | "\(.step): \(.probs[2])"'

# → 어느 step에서 nude 확률이 급증하는지 확인
```

---

## 📚 다음 단계

### 실험 아이디어:

1. **Suppression 효과 검증**
   - Baseline vs Suppressed 이미지 비교
   - 히트맵 차이 분석

2. **Critical Step 연구**
   - 여러 프롬프트로 generation mode 실행
   - nude concept이 발현하는 시점 패턴 찾기

3. **Layer 분석**
   - 어떤 layer에서 nude feature가 활성화되는지
   - Unlearning 시 특정 layer에 집중할 수 있을까?

4. **Channel Importance**
   - Integrated Gradients로 중요한 channel 찾기
   - 특정 channel을 조작하면 nude 판단이 바뀌는가?

---

## 📞 도움말

- **상세 가이드**: [INTERPRETATION_README.md](INTERPRETATION_README.md)
- **코드 문서**: [geo_utils/classifier_interpretability.py](geo_utils/classifier_interpretability.py)
- **예제 코드**: [examples/quick_interpret.py](examples/quick_interpret.py)

---

## ⚡ 빠른 체크리스트

- [ ] Classifier 체크포인트 확인: `ls work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth`
- [ ] 테스트 이미지 확인: `ls img/CNBWON/*.png | head`
- [ ] GPU 사용 가능 확인: `nvidia-smi`
- [ ] 단일 이미지 분석 실행: `./interpret_single_image.sh`
- [ ] 결과 확인: `ls interpretation_results/single_*/`

**첫 실행 추천**: `./interpret_single_image.sh` (가장 간단!)
