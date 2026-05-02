# Machine Unlearning: Nudity Removal with FK Steering

**목적**: Sexual/nude content가 포함된 프롬프트를 입력으로 받아, **nudity가 제거된 (clothed people)** 이미지를 생성합니다.

이는 FK Steering을 사용한 **inference-time machine unlearning**입니다 - 모델 재학습 없이, 생성 시점에 원하지 않는 콘텐츠를 제거합니다.

## 🎯 동작 방식

```
Input:  "a photo of nude people at the beach"
        ↓
    FK Steering
    (Target: Clothed People - Class 1)
        ↓
Output: Photo of CLOTHED people at the beach
```

### 어떻게 작동하나?

1. **Multiple particles (k개)**: k개의 이미지를 동시에 생성
2. **Classifier reward**: 각 diffusion step마다 nudity classifier로 평가
   - Reward = `logit[clothed] - logit[nude]` (옷 입은 사람 선호)
3. **Resampling**: 높은 reward (더 옷 입은) particle은 유지, 낮은 것은 제거
4. **결과**: Nude content가 자동으로 clothed로 변환됨

## 📁 파일 구조

```
SoftDelete+CG/
├── unlearn_nudity_fk.py              # 메인 스크립트
├── run_unlearn_nudity.sh             # 간단 실행 스크립트
├── compare_unlearning_methods.sh     # 여러 방법 비교
├── fk_steering.py                    # FK steering 구현
└── prompts/
    └── sexual_50.txt                 # Nude/sexual 프롬프트 50개
```

## 🚀 빠른 시작

### 1. 기본 실행 (sexual_50.txt 전체 처리)

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# 간단한 실행
bash run_unlearn_nudity.sh
```

결과:
- `./unlearned_outputs/sexual_fk_steering/` 에 이미지 생성됨
- 각 프롬프트마다:
  - `prompt000_sample0_best.png` - 가장 높은 reward (가장 옷 입은)
  - `prompt000_sample0_baseline.png` - 비교를 위한 baseline
  - `prompt000_sample0_particle*.png` - 모든 particle들
- `results.json` - 상세한 결과 데이터

### 2. 커스텀 프롬프트 파일 사용

```bash
python unlearn_nudity_fk.py \
    --prompt_file your_prompts.txt \
    --output_dir ./your_output \
    --num_particles 4 \
    --lambda_scale 15.0
```

### 3. 여러 방법 비교

```bash
# Baseline vs Best-of-N vs FK Steering 비교
bash compare_unlearning_methods.sh
```

이 스크립트는 다음을 비교합니다:
- ✅ Baseline (unlearning 없음)
- ✅ Best-of-N (k=4)
- ✅ FK Steering (k=2, λ=5)
- ✅ FK Steering (k=4, λ=10)
- ✅ FK Steering (k=4, λ=15) ← 권장
- ✅ FK Steering (k=8, λ=15)

## ⚙️ 주요 파라미터

### `--num_particles` (k)
- **의미**: 동시에 생성할 이미지 개수
- **기본값**: 4
- **추천**:
  - `k=2`: 빠르지만 효과적 (논문에서도 좋은 결과)
  - `k=4`: 성능/속도 균형 ✅ 권장
  - `k=8`: 더 나은 품질, 느림

### `--lambda_scale` (λ)
- **의미**: Nudity removal 강도
- **기본값**: 15.0
- **추천**:
  - `λ=5.0`: 약한 steering (미묘한 변화)
  - `λ=10.0`: 중간 steering
  - `λ=15.0`: 강한 steering ✅ 권장
  - `λ=20.0`: 매우 강한 steering (과도할 수 있음)

### `--potential_type`
- **의미**: Particle 평가 방식
- **옵션**:
  - `max`: 최고 reward particle 선호 ✅ 기본값
  - `difference`: Reward 증가하는 particle 선호
  - `sum`: 누적 reward 높은 particle 선호

### `--resampling_interval`
- **의미**: 몇 step마다 resampling 할지
- **기본값**: 10
- **추천**:
  - `5`: 더 자주 resample (강한 steering)
  - `10`: 균형 ✅ 권장
  - `20`: 덜 자주 resample (더 탐색적)

## 📊 결과 분석

### results.json 구조

```json
{
  "config": { ... },
  "prompts": [
    {
      "prompt": "original prompt with nude content",
      "prompt_idx": 0,
      "samples": [
        {
          "best_reward": 3.456,        // 가장 높은 reward
          "mean_reward": 2.891,         // 평균 reward
          "baseline_reward": 0.234,     // Baseline reward
          "improvement": 3.222,         // FK - Baseline
          "all_rewards": [2.1, 3.4, 2.9, 2.8]
        }
      ]
    }
  ]
}
```

### Reward 해석

```python
# Classifier 출력: [logit_not_people, logit_clothed, logit_nude]

# Target Class 1 (clothed people)의 reward:
reward = logit[1] - logit[2]
```

- **높은 reward** (예: 3.0+): 강하게 clothed people
- **중간 reward** (예: 1.0~3.0): 적당히 clothed
- **낮은 reward** (예: < 0): Nude가 남아있음

### 성공 기준

✅ **성공적인 unlearning**:
- FK reward >> Baseline reward
- Final images에 nude content 없음
- 여전히 prompt의 다른 요소들은 유지 (장소, 스타일 등)

## 📝 예시

### Example 1: 기본 설정

```bash
python unlearn_nudity_fk.py \
    --prompt_file ./prompts/sexual_50.txt \
    --output_dir ./outputs/default \
    --num_particles 4 \
    --lambda_scale 15.0 \
    --generate_baseline
```

**예상 결과**:
```
Best reward: 3.7891
Baseline reward: -0.5432
Improvement: 4.3323  ← 큰 개선!
```

### Example 2: 강한 steering

```bash
python unlearn_nudity_fk.py \
    --prompt_file ./prompts/sexual_50.txt \
    --output_dir ./outputs/strong \
    --num_particles 8 \
    --lambda_scale 20.0 \
    --resampling_interval 5
```

더 강력한 nudity removal, 더 많은 compute 필요

### Example 3: 빠른 실행

```bash
python unlearn_nudity_fk.py \
    --prompt_file ./prompts/sexual_50.txt \
    --output_dir ./outputs/fast \
    --num_particles 2 \
    --lambda_scale 10.0 \
    --num_inference_steps 25  # 더 적은 step
```

빠르지만 여전히 효과적

## 🔬 실험 아이디어

### 1. Lambda Scaling 실험

```bash
for LAMBDA in 5 10 15 20; do
    python unlearn_nudity_fk.py \
        --prompt_file ./prompts/sexual_50.txt \
        --output_dir ./experiments/lambda_${LAMBDA} \
        --lambda_scale ${LAMBDA} \
        --num_particles 4
done
```

### 2. Particle Scaling 실험

```bash
for K in 2 4 8 16; do
    python unlearn_nudity_fk.py \
        --prompt_file ./prompts/sexual_50.txt \
        --output_dir ./experiments/k_${K} \
        --num_particles ${K} \
        --lambda_scale 15.0
done
```

### 3. Potential 비교

```bash
for POT in max difference sum; do
    python unlearn_nudity_fk.py \
        --prompt_file ./prompts/sexual_50.txt \
        --output_dir ./experiments/potential_${POT} \
        --potential_type ${POT} \
        --num_particles 4
done
```

## 🎨 출력 예시

생성되는 파일들:

```
outputs/
├── prompt000_sample0_baseline.png      # 원본 (nude 가능)
├── prompt000_sample0_best.png          # FK steering 결과 (clothed)
├── prompt000_sample0_particle0.png     # Particle 0
├── prompt000_sample0_particle1.png     # Particle 1
├── prompt000_sample0_particle2.png     # Particle 2
├── prompt000_sample0_particle3.png     # Particle 3
└── results.json                        # 상세 결과
```

## ⚡ 성능 최적화

### GPU 메모리 부족시:

```bash
# Particle 수 줄이기
--num_particles 2

# 또는 float32 대신 float16 사용 (코드에서 자동)
```

### 더 빠른 생성:

```bash
# Step 수 줄이기
--num_inference_steps 25

# Resampling 덜 하기
--resampling_interval 20

# Particle 수 줄이기
--num_particles 2
```

### 더 나은 품질:

```bash
# 더 많은 particles
--num_particles 8

# 더 많은 steps
--num_inference_steps 100

# 더 강한 steering
--lambda_scale 20.0
```

## 📈 기대 결과 (논문 기반)

FK Steering의 장점:
- ✅ **No training needed**: 모델 재학습 불필요
- ✅ **Better than fine-tuning**: Fine-tuned 모델보다 좋은 결과
- ✅ **Efficient**: Best-of-N보다 같은 k로 더 나은 결과
- ✅ **Scalable**: k=2만으로도 효과적

예상 성능:
- **k=2**: Baseline보다 큰 개선
- **k=4**: 최적의 성능/속도 균형
- **k=8**: Marginal improvement

## 🔍 디버깅

### 문제: Reward가 여전히 낮음

```bash
# Lambda 증가
--lambda_scale 20.0

# 더 자주 resample
--resampling_interval 5

# 더 많은 particles
--num_particles 8
```

### 문제: 이미지가 너무 왜곡됨

```bash
# Lambda 감소
--lambda_scale 10.0

# 덜 자주 resample
--resampling_interval 15
```

### 문제: 너무 느림

```bash
# Step 수 감소
--num_inference_steps 25

# Particle 수 감소
--num_particles 2

# Baseline 생성 안 함
# (--generate_baseline 플래그 제거)
```

## 📚 추가 정보

- **논문**: "A General Framework for Inference-time Scaling and Steering of Diffusion Models" (Singhal et al., 2025)
- **arXiv**: https://arxiv.org/abs/2501.06848
- **FK Steering 상세 설명**: `FK_STEERING_README.md` 참고

## 🎯 Use Cases

1. **Content Moderation**: Sexual content 자동 필터링
2. **Safety Testing**: Red-teaming을 위한 edge case 생성
3. **Dataset Cleaning**: Inappropriate content 제거
4. **Research**: Machine unlearning 연구

## ⚠️ 주의사항

- 이 도구는 **연구 및 안전 목적**으로 만들어졌습니다
- Classifier가 완벽하지 않을 수 있습니다
- 생성된 이미지는 여전히 검토가 필요할 수 있습니다
- Lambda가 너무 높으면 이미지 품질이 저하될 수 있습니다

## 💡 팁

1. **처음 시작**: `bash run_unlearn_nudity.sh`로 기본 설정 테스트
2. **비교 필요**: `bash compare_unlearning_methods.sh`로 여러 방법 비교
3. **Fine-tuning**: Lambda와 k를 조절해서 최적 값 찾기
4. **Baseline 생성**: `--generate_baseline`으로 개선 정도 확인

---

**Happy Unlearning! 🚀**
