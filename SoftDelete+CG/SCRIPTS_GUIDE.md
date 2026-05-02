# 🚀 Bash Scripts 사용 가이드

## 📋 스크립트 목록

### 1. 메인 실행 스크립트

| 스크립트 | 용도 | 사용법 |
|---------|------|--------|
| `run_complete_workflow.sh` | 전체 워크플로우 통합 실행 | `./run_complete_workflow.sh [quick\|presets\|full]` |
| `run_soft_cg_experiment.sh` | Preset 및 비교 실험 | `./run_soft_cg_experiment.sh [strong\|all-presets]` |
| `quick_test.sh` | 빠른 단일 테스트 | `./quick_test.sh [default\|temp\|schedule]` |
| `analyze_results.sh` | 결과 분석 및 시각화 | `./analyze_results.sh [all\|visualize\|grid]` |

---

## 🎯 빠른 시작 (Quick Start)

### 1️⃣ 환경 설정

```bash
# Classifier 경로 설정 (중요!)
export CLASSIFIER_PATH="checkpoints/nude_classifier_best.pth"

# 또는 스크립트 내에서 직접 수정:
# vim run_soft_cg_experiment.sh
# CLASSIFIER_PATH="your/path/here"
```

### 2️⃣ 가장 빠른 테스트

```bash
# 추천 설정으로 1개 이미지 생성 (1-2분)
./quick_test.sh default
```

### 3️⃣ Preset 실험

```bash
# 추천 preset (Strong Decay) 실행
./run_soft_cg_experiment.sh strong

# 모든 preset 실행
./run_soft_cg_experiment.sh all-presets
```

### 4️⃣ 전체 워크플로우

```bash
# 모든 실험 + 분석 + 리포트 (시간 소요)
./run_complete_workflow.sh full
```

---

## 📚 상세 사용법

### `run_complete_workflow.sh` - 통합 워크플로우

**전체 실험 파이프라인을 한번에 실행**

```bash
# 옵션:
./run_complete_workflow.sh quick      # 빠른 테스트만
./run_complete_workflow.sh presets    # Preset 실험
./run_complete_workflow.sh full       # 전체 (추천!)
./run_complete_workflow.sh analyze    # 분석만
./run_complete_workflow.sh custom     # 단계 선택
./run_complete_workflow.sh menu       # 대화형 메뉴
```

**워크플로우 단계:**
1. 환경 확인 (Python, GPU, 파일)
2. Weight scheduling 시각화
3. 빠른 기능 테스트
4. Preset 4개 실험
5. 비교 실험 (Temperature, Strategy, Scale)
6. 결과 분석 및 리포트 생성

**출력:**
- `EXPERIMENT_REPORT.md`: 실험 리포트
- `experiment_statistics.json`: 통계 데이터
- `visualizations/`: 시각화 그래프
- `outputs/soft_cg_experiments/`: 생성 이미지

---

### `run_soft_cg_experiment.sh` - Preset 실험

**다양한 preset 설정으로 실험**

```bash
# 개별 Preset 실행
./run_soft_cg_experiment.sh gentle       # Preset 1: Gentle Increase
./run_soft_cg_experiment.sh strong       # Preset 2: Strong Decay ⭐
./run_soft_cg_experiment.sh constant     # Preset 3: Constant Soft
./run_soft_cg_experiment.sh aggressive   # Preset 4: Aggressive Decay

# 그룹 실행
./run_soft_cg_experiment.sh all-presets       # 모든 preset
./run_soft_cg_experiment.sh all-comparisons   # 모든 비교 실험
./run_soft_cg_experiment.sh all               # 전부

# 비교 실험
./run_soft_cg_experiment.sh temp         # Temperature 비교
./run_soft_cg_experiment.sh strategy     # Strategy 비교
./run_soft_cg_experiment.sh scale        # Guidance Scale 비교
```

**Preset 상세:**

#### Preset 1: Gentle Increase (부드럽게 → 강하게)
```bash
./run_soft_cg_experiment.sh gentle
```
- Strategy: Linear Increase
- Weight: 0.5 → 2.0
- Temperature: 2.0 (매우 soft)
- 용도: 자연스러운 시작, 후반 강화

#### Preset 2: Strong Decay ⭐ **추천!**
```bash
./run_soft_cg_experiment.sh strong
```
- Strategy: Cosine Anneal
- Weight: 5.0 → 0.5
- Temperature: 1.0 (적당)
- 용도: 초반 강력한 방향 설정, 자연스러운 마무리

#### Preset 3: Constant Soft (일정)
```bash
./run_soft_cg_experiment.sh constant
```
- Strategy: Constant
- Weight: 1.0 (고정)
- Temperature: 1.0
- 용도: 균일한 guidance

#### Preset 4: Aggressive Decay (강력)
```bash
./run_soft_cg_experiment.sh aggressive
```
- Strategy: Exponential Decay
- Weight: 10.0 → 0.1 (빠른 감소)
- Temperature: 0.5 (sharp)
- 용도: 매우 강력한 초기 교정

---

### `quick_test.sh` - 빠른 테스트

**빠르게 단일 이미지 생성하여 효과 확인**

```bash
# 기본 테스트 (추천 설정)
./quick_test.sh default

# Soft vs Binary 마스크 비교
./quick_test.sh soft-vs-binary

# Temperature 효과 (0.1, 1.0, 5.0)
./quick_test.sh temp

# Scheduling 전략 비교
./quick_test.sh schedule

# 커스텀 파라미터 (대화형)
./quick_test.sh custom

# Safe vs Unsafe 프롬프트 비교
./quick_test.sh safe-vs-unsafe
```

**예시 출력:**
```
outputs/quick_test/
├── default/          # 기본 설정
├── temp_0.1/         # Sharp mask
├── temp_1.0/         # Medium mask
├── temp_5.0/         # Soft mask
└── statistics.json   # 통계
```

---

### `analyze_results.sh` - 결과 분석

**실험 결과 분석 및 시각화**

```bash
# 모든 분석 실행
./analyze_results.sh all

# 개별 분석
./analyze_results.sh visualize    # Weight schedule 그래프
./analyze_results.sh stats        # 통계 수집
./analyze_results.sh grid         # 이미지 비교 그리드
./analyze_results.sh report       # Markdown 리포트
```

**생성되는 파일:**
```
visualizations/
├── weight_schedules.png                # Scheduling 전략 비교
├── temperature_effect.png              # Temperature 효과
├── temperature_comparison_grid.png     # Temperature 이미지 비교
├── strategy_comparison_grid.png        # Strategy 이미지 비교
└── scale_comparison_grid.png           # Scale 이미지 비교

experiment_statistics.json              # 모든 실험 통계
EXPERIMENT_REPORT.md                    # 종합 리포트
```

---

## 🎨 실험 시나리오

### 시나리오 1: 처음 사용자 (빠른 검증)

```bash
# 1. 빠른 기본 테스트
./quick_test.sh default

# 2. 결과 확인
ls outputs/quick_test/default/

# 3. 추천 preset 실행
./run_soft_cg_experiment.sh strong

# 4. 결과 확인
ls outputs/soft_cg_experiments/strong_decay/
```

### 시나리오 2: 파라미터 튜닝

```bash
# 1. Temperature 효과 확인
./quick_test.sh temp

# 2. 최적 temperature로 다른 파라미터 테스트
./quick_test.sh custom
# (대화형으로 파라미터 입력)

# 3. 전체 비교 실험
./run_soft_cg_experiment.sh all-comparisons

# 4. 결과 분석
./analyze_results.sh all
```

### 시나리오 3: 프로덕션 배포

```bash
# 1. 전체 워크플로우 실행
./run_complete_workflow.sh full

# 2. 리포트 확인
cat EXPERIMENT_REPORT.md

# 3. 시각화 확인
ls visualizations/

# 4. 최적 설정 선택 후 프로덕션 적용
```

### 시나리오 4: 연구/논문용

```bash
# 1. 모든 preset 실험
./run_soft_cg_experiment.sh all-presets

# 2. 모든 비교 실험
./run_soft_cg_experiment.sh all-comparisons

# 3. 상세 분석
./analyze_results.sh all

# 4. 통계 및 그래프 사용
# - experiment_statistics.json
# - visualizations/*.png
# - EXPERIMENT_REPORT.md
```

---

## ⚙️ 설정 변경

### Classifier 경로 변경

**방법 1: 환경 변수**
```bash
export CLASSIFIER_PATH="/path/to/your/classifier.pth"
./run_soft_cg_experiment.sh strong
```

**방법 2: 스크립트 직접 수정**
```bash
vim run_soft_cg_experiment.sh

# 다음 라인 수정:
CLASSIFIER_PATH="your/custom/path.pth"
```

### 출력 디렉토리 변경

```bash
vim run_soft_cg_experiment.sh

# 다음 라인 수정:
OUTPUT_BASE_DIR="outputs/my_custom_experiments"
```

### 이미지 생성 수 변경

```bash
vim run_soft_cg_experiment.sh

# 다음 라인 수정:
NUM_IMAGES=10  # 프롬프트당 10개 생성
```

### 커스텀 프롬프트 추가

```bash
vim run_soft_cg_experiment.sh

# run_strong_decay() 함수에서 --prompts 수정:
--prompts "your custom prompt 1" "your custom prompt 2" \
```

---

## 🐛 트러블슈팅

### 1. Classifier를 찾을 수 없음

```bash
⚠️  경고: Classifier 체크포인트를 찾을 수 없습니다
```

**해결:**
```bash
# 경로 확인
ls checkpoints/

# 환경 변수 설정
export CLASSIFIER_PATH="올바른/경로/classifier.pth"
```

### 2. GPU 메모리 부족

```bash
RuntimeError: CUDA out of memory
```

**해결:**
```bash
# 스크립트에서 batch size 줄이기
vim run_soft_cg_experiment.sh

# BATCH_SIZE=1 확인 (이미 1이면 이미지 수 줄이기)
NUM_IMAGES=1  # 한번에 1개만 생성
```

### 3. 스크립트 실행 권한 오류

```bash
Permission denied
```

**해결:**
```bash
chmod +x *.sh
```

### 4. Python 패키지 없음

```bash
ModuleNotFoundError: No module named 'diffusers'
```

**해결:**
```bash
pip install diffusers accelerate transformers torch scipy
```

---

## 📊 결과 이해하기

### Statistics.json 해석

```json
{
  "total_steps": 50,           // 전체 denoising 스텝
  "harmful_steps": 35,          // Harmful 감지된 스텝
  "guidance_applied": 35,       // Guidance 적용된 스텝
  "harmful_ratio": 0.7,         // 70% 스텝에서 harmful 감지
  "guidance_ratio": 0.7         // 70% 스텝에서 guidance 적용
}
```

**좋은 결과:**
- **Unsafe 프롬프트**: `guidance_ratio` > 0.5 (guidance 잘 적용됨)
- **Safe 프롬프트**: `guidance_ratio` ≈ 0 (guidance 거의 안됨)

### 시각화 그래프 해석

**weight_schedules.png:**
- X축: Denoising step (0~50)
- Y축: Weight multiplier
- 각 선: 다른 scheduling 전략
- **확인**: 원하는 패턴으로 weight가 변하는지

**temperature_effect.png:**
- X축: Heatmap value (0~1)
- Y축: Mask value (0~1)
- 다른 색: 다른 temperature
- **확인**: Threshold 0.5 근처에서 얼마나 부드럽게 전환되는지

---

## 💡 팁 & 트릭

### 1. 빠른 반복 테스트

```bash
# 이미지 1개만 빠르게 생성
NUM_IMAGES=1 ./quick_test.sh default
```

### 2. 특정 seed로 재현

```bash
# quick_test.sh에서 SEED 변경
vim quick_test.sh
SEED=12345  # 원하는 seed
```

### 3. 로그 저장

```bash
# 실행 로그 저장
./run_complete_workflow.sh full 2>&1 | tee experiment.log
```

### 4. 백그라운드 실행 (긴 실험)

```bash
# 백그라운드에서 실행
nohup ./run_soft_cg_experiment.sh all > experiment.log 2>&1 &

# 진행 상황 확인
tail -f experiment.log
```

### 5. 디스크 공간 관리

```bash
# 오래된 실험 결과 삭제
rm -rf outputs/soft_cg_experiments/old_experiment

# 통계만 백업
cp -r outputs/soft_cg_experiments/**/statistics.json backup/
```

---

## 🎯 권장 워크플로우

### 초보자
```bash
1. ./quick_test.sh default
2. ./run_soft_cg_experiment.sh strong
3. ./analyze_results.sh visualize
```

### 중급자
```bash
1. ./quick_test.sh temp
2. ./quick_test.sh schedule
3. ./run_soft_cg_experiment.sh all-presets
4. ./analyze_results.sh all
```

### 전문가
```bash
1. ./run_complete_workflow.sh full
2. 결과 분석 후 커스텀 설정 조정
3. ./quick_test.sh custom (파인 튜닝)
4. 최종 설정으로 대량 생성
```

---

## 📖 관련 문서

- [SOFT_SPATIAL_CG_GUIDE.md](SOFT_SPATIAL_CG_GUIDE.md): 상세 사용 가이드
- [README_SOFT_CG.md](README_SOFT_CG.md): 빠른 시작 가이드
- [configs/soft_spatial_cg_tuning.yaml](configs/soft_spatial_cg_tuning.yaml): 설정 파일

---

## 🚀 다음 단계

1. **빠른 테스트로 시작**: `./quick_test.sh default`
2. **추천 preset 실행**: `./run_soft_cg_experiment.sh strong`
3. **결과 확인 후 조정**: 파라미터 튜닝
4. **전체 워크플로우**: `./run_complete_workflow.sh full`
5. **프로덕션 배포**: 최적 설정 적용

Happy experimenting! 🎉
