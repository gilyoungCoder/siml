# Grid Search Guide for Adaptive CG

빡세게 튜닝하기 위한 Grid Search 가이드입니다.

## 📋 개요

2개의 Grid Search 스크립트:
- `grid_search_nudity.sh`: Nudity classifier 튜닝
- `grid_search_violence.sh`: Violence classifier 튜닝

각 스크립트는 파라미터 조합을 자동으로 실험하고, 결과를 폴더별로 저장합니다.

---

## 🎯 Grid Search 파라미터

### 1. Guidance Scale
```bash
GUIDANCE_SCALES=(5.0 7.0 10.0)
```
- **5.0**: 부드러운 guidance
- **7.0**: 중간 강도 (권장)
- **10.0**: 강한 guidance

### 2. Harmful Scale
```bash
HARMFUL_SCALES=(1.0 1.5 2.0)
```
- **1.0**: Safe pull = Harmful push (균형)
- **1.5**: Harmful push 1.5배
- **2.0**: Harmful push 2배 (강함)

### 3. Weight Schedule
```bash
WEIGHT_SCHEDULES=(
    "3.0 0.5"    # 부드러운 scheduling
    "4.0 1.0"    # 중간 (권장)
    "5.0 1.0"    # 약간 강함
    "8.0 1.0"    # 강함
)
```
- **첫 번째 값**: 초반 weight multiplier
- **두 번째 값**: 후반 weight multiplier

### 4. Adaptive Threshold Schedule
```bash
THRESHOLD_SCHEDULES=(
    "0.0 -1.5"    # 관대 → 엄격
    "0.0 -2.0"    # 관대 → 매우 엄격 (권장)
    "-0.5 -2.5"   # 약간 엄격 → 매우 엄격
)
```
- **첫 번째 값**: 초반 harmful 감지 threshold (logit)
- **두 번째 값**: 후반 harmful 감지 threshold (logit)
- 음수일수록 더 민감하게 감지

---

## 📊 총 실험 개수

```
총 실험 = Guidance Scales × Harmful Scales × Weight Schedules × Threshold Schedules
        = 3 × 3 × 4 × 3
        = 108 experiments per concept
```

**Nudity + Violence = 216 experiments**

---

## 🚀 실행 방법

### Nudity Grid Search
```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# 실행
./grid_search_nudity.sh
```

**예상 시간**: 약 3-4시간 (108 experiments × 2분/exp)

**출력 위치**: `scg_outputs/grid_search_nudity/`

### Violence Grid Search
```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# 실행
./grid_search_violence.sh
```

**예상 시간**: 약 3-4시간

**출력 위치**: `scg_outputs/grid_search_violence/`

---

## 📁 폴더 구조

```
scg_outputs/grid_search_nudity/
├── gs5.0_hs1.0_ws3.0-0.5_ts0.0--1.5/
│   ├── 0000_00.png
│   ├── 0001_00.png
│   └── visualizations/
│       └── analysis_0000_00.png
├── gs5.0_hs1.0_ws3.0-0.5_ts0.0--2.0/
├── gs5.0_hs1.0_ws3.0-0.5_ts-0.5--2.5/
├── gs5.0_hs1.0_ws4.0-1.0_ts0.0--1.5/
... (108 experiments)
```

### 폴더 이름 규칙
```
gs{guidance_scale}_hs{harmful_scale}_ws{weight_start}-{weight_end}_ts{threshold_start}-{threshold_end}

예:
gs7.0_hs1.5_ws4.0-1.0_ts0.0--2.0
│   │    │      │         │
│   │    │      │         └─ Threshold: 0.0 → -2.0
│   │    │      └─────────── Weight: 4.0 → 1.0
│   │    └────────────────── Harmful Scale: 1.5
│   └─────────────────────── Guidance Scale: 7.0
```

---

## 📈 결과 분석

### 자동 분석 (권장)
```bash
# Nudity 결과 분석
python analyze_grid_search.py scg_outputs/grid_search_nudity/ \
    --output nudity_grid_results.csv

# Violence 결과 분석
python analyze_grid_search.py scg_outputs/grid_search_violence/ \
    --output violence_grid_results.csv
```

**출력**:
- CSV 파일: 모든 실험 결과 정리
- 콘솔: 요약 통계 및 Top 10 실험

### 수동 비교
```bash
# 특정 폴더 확인
ls scg_outputs/grid_search_nudity/gs7.0_hs1.5_ws4.0-1.0_ts0.0--2.0/

# 이미지 개수 비교
for dir in scg_outputs/grid_search_nudity/gs*; do
    echo "$dir: $(ls $dir/*.png 2>/dev/null | wc -l) images"
done | sort -t: -k2 -rn | head -10
```

---

## 🎨 시각적 비교 방법

### 1. 같은 프롬프트 비교
```bash
# 프롬프트 0번 이미지만 모아서 비교
mkdir -p comparison/prompt_0000
for dir in scg_outputs/grid_search_nudity/gs*; do
    exp_name=$(basename $dir)
    cp $dir/0000_00.png comparison/prompt_0000/${exp_name}.png
done
```

### 2. Visualization 비교
```bash
# Analysis 차트 모아보기
mkdir -p comparison/analysis
for dir in scg_outputs/grid_search_nudity/gs*/visualizations; do
    exp_name=$(basename $(dirname $dir))
    cp $dir/analysis_0000_00.png comparison/analysis/${exp_name}.png 2>/dev/null
done
```

---

## ⚙️ 파라미터 조정

### 실험 범위 줄이기 (빠른 테스트)
```bash
# grid_search_nudity.sh 수정

# 예: Guidance Scale만 테스트
GUIDANCE_SCALES=(5.0 7.0 10.0)
HARMFUL_SCALES=(1.5)              # 고정
WEIGHT_SCHEDULES=("4.0 1.0")      # 고정
THRESHOLD_SCHEDULES=("0.0 -2.0")  # 고정

# 총 실험: 3 × 1 × 1 × 1 = 3 experiments
```

### 프롬프트 줄이기 (빠른 테스트)
```bash
# 처음 10개 프롬프트만 사용
head -10 prompts/sexual_50.txt > prompts/sexual_test.txt

# grid_search_nudity.sh에서
PROMPT_FILE="./prompts/sexual_test.txt"
```

### 샘플 개수 조정
```bash
# grid_search_nudity.sh Line 14
NSAMPLES=1  # 빠른 실험용
NSAMPLES=3  # 최종 평가용
```

---

## 📊 평가 기준

### 1. Guidance 적용률
```
목표: 30-60% (너무 낮거나 높으면 비효율)
확인: Visualization의 "Harmful Steps" 비율
```

### 2. 이미지 품질
```
육안 평가:
- Harmful content 제거 정도
- 이미지 자연스러움
- Artifact 발생 여부
```

### 3. Masked Region Ratio
```
목표: 0.05-0.20 (5-20% 영역)
너무 높으면: 과도한 마스킹 (부자연스러움)
너무 낮으면: 부족한 마스킹 (효과 미미)
```

---

## 🏆 최적 파라미터 찾기

### Step 1: CSV 분석
```bash
# 결과 정렬
python analyze_grid_search.py scg_outputs/grid_search_nudity/

# Top 10 확인
```

### Step 2: 시각적 비교
```bash
# Top 10 실험의 이미지 비교
```

### Step 3: 최종 선정
```bash
# 선택한 파라미터로 run_full_adaptive.sh 업데이트
```

---

## 💡 팁

### 1. 병렬 실행 (2개 GPU 사용)
```bash
# Terminal 1 (GPU 7)
export CUDA_VISIBLE_DEVICES=7
./grid_search_nudity.sh

# Terminal 2 (GPU 6)
export CUDA_VISIBLE_DEVICES=6
./grid_search_violence.sh
```

### 2. 중단 후 재개
```bash
# 이미 생성된 폴더는 스킵하도록 수정
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A $OUTPUT_DIR/*.png 2>/dev/null)" ]; then
    echo "⏭️  Skipping (already exists)"
    continue
fi
```

### 3. 디스크 공간 관리
```bash
# Visualization 제거 (용량 절약)
find scg_outputs/grid_search_nudity -name "visualizations" -type d -exec rm -rf {} +

# 하나의 실험: 약 50MB (50 prompts × 1 sample)
# 108 experiments: 약 5-6GB
```

---

## 🎯 권장 실험 순서

### Phase 1: 빠른 탐색 (3×2×2×2 = 24 experiments)
```bash
GUIDANCE_SCALES=(5.0 7.0 10.0)
HARMFUL_SCALES=(1.0 2.0)
WEIGHT_SCHEDULES=("3.0 0.5" "5.0 1.0")
THRESHOLD_SCHEDULES=("0.0 -1.5" "0.0 -2.0")
NSAMPLES=1
```

### Phase 2: 정밀 탐색 (유망한 범위 집중)
```bash
# Phase 1 결과 기반으로 범위 좁히기
# 예: guidance_scale 7.0이 좋았다면
GUIDANCE_SCALES=(6.0 7.0 8.0)
NSAMPLES=3
```

---

## 📝 결과 기록

실험 결과를 기록하세요:

```markdown
## Best Parameters for Nudity

**Experiment**: gs7.0_hs1.5_ws4.0-1.0_ts0.0--2.0

**Parameters**:
- Guidance Scale: 7.0
- Harmful Scale: 1.5
- Weight Schedule: 4.0 → 1.0
- Threshold Schedule: 0.0 → -2.0

**Results**:
- Harmful detection: 45%
- Average masked region: 12%
- Image quality: Excellent
- Harmful removal: 95%
```

---

## 🔧 문제 해결

### OOM 에러
```bash
# NSAMPLES 줄이기
NSAMPLES=1

# 또는 gradient checkpointing 활성화 (코드 수정 필요)
```

### 너무 느림
```bash
# 프롬프트 줄이기
PROMPT_FILE="./prompts/sexual_10.txt"

# Visualization 끄기
# --save_visualizations 제거
```

### 디스크 가득참
```bash
# 진행 중인 실험 정리
rm -rf scg_outputs/grid_search_nudity/gs5.0_hs1.0_*
```

---

## ✅ Checklist

실행 전:
- [ ] GPU 메모리 확인 (nvidia-smi)
- [ ] 디스크 공간 확인 (df -h)
- [ ] Classifier 경로 확인
- [ ] Prompt 파일 확인

실행 중:
- [ ] 첫 실험 결과 확인 (이미지 생성되는지)
- [ ] 중간 결과 모니터링
- [ ] 디스크 사용량 모니터링

실행 후:
- [ ] 결과 분석 (analyze_grid_search.py)
- [ ] Top 10 시각적 확인
- [ ] 최적 파라미터 기록
- [ ] run_full_adaptive.sh 업데이트

---

**준비 완료! Grid Search를 시작하세요! 🚀**
