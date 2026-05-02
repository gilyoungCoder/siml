# Selective Classifier Guidance - Implementation Summary

## 📦 구현 완료된 파일들

### 1. 핵심 유틸리티
- **`geo_utils/selective_guidance_utils.py`** (약 400줄)
  - `SelectiveGuidanceMonitor`: Latent 모니터링 및 harmful 감지
  - `SpatiallyMaskedGuidance`: 공간적 masking된 guidance 적용
  - `visualize_selective_guidance()`: 통계 시각화

### 2. 메인 스크립트
- **`generate_selective_cg.py`** (약 350줄)
  - Selective guidance 메인 로직
  - Callback 기반 step-wise guidance
  - 통계 수집 및 로깅

### 3. 실행 스크립트
- **`run_selective_cg.sh`**: 표준 실험 설정
- **`run_selective_cg_benign.sh`**: Benign prompts 테스트용 (conservative 설정)
- **`test_selective_cg.sh`**: 빠른 검증 테스트 (5 steps)

### 4. 문서
- **`README_selective_cg.md`**: 상세 사용 가이드
- **`SELECTIVE_CG_SUMMARY.md`**: 이 파일 (구현 요약)

---

## 🎯 핵심 개념

### 문제점 (기존 방식)
```
generate_classifier_masked.py:
- 모든 timestep에서 Grad-CAM masking 적용
- Benign prompt도 불필요하게 훼손
- GENEVAL score 저하
```

### 해결책 (Selective CG)
```
각 timestep마다:
  1. Classifier로 latent 평가 → harmful_score 계산
  2. IF harmful_score > threshold:
       → Grad-CAM으로 harmful 영역 탐지
       → Safe 방향 gradient 계산
       → Harmful 영역에만 gradient 적용
  3. ELSE:
       → Guidance 건너뛰기 (vanilla diffusion)
```

### 장점
✅ Benign prompts: 최소 개입 → GENEVAL 유지
✅ Harmful prompts: 타겟 억제
✅ 공간적 정밀도: Grad-CAM
✅ 계산 효율: Selective application

---

## 🚀 사용 방법

### 1. 빠른 테스트 (추천: 먼저 실행!)

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
./test_selective_cg.sh
```

**기대 결과**:
- 2개 prompt × 1 sample × 5 steps
- 약 1-2분 소요
- `test_output_selective_cg/` 에 이미지 생성
- ✅ 에러 없이 완료되면 구현 성공!

### 2. 본 실험

```bash
# Harmful prompts 테스트
./run_selective_cg.sh

# Benign prompts 테스트
./run_selective_cg_benign.sh
```

### 3. 모니터링

```bash
# 실시간 로그 확인
tail -f ./logs/run_selective_cg_*.log

# 출력 이미지 확인
ls -lh ./outputs/selective_cg_v1/

# 시각화 확인
eog ./outputs/selective_cg_v1/visualizations/*.png
```

---

## ⚙️ 주요 파라미터

| 파라미터 | 기본값 | 설명 | 조정 방법 |
|---------|-------|------|----------|
| `HARMFUL_THRESHOLD` | 0.5 | Harmful 감지 임계값 | ↑ 보수적 (덜 개입), ↓ 공격적 (더 개입) |
| `GUIDANCE_SCALE` | 5.0 | Guidance 강도 | ↑ 강한 억제, ↓ 약한 억제 |
| `SPATIAL_PERCENTILE` | 0.3 | Masking 비율 (상위 30%) | ↑ 넓은 영역, ↓ 좁은 영역 |
| `GUIDANCE_START_STEP` | 0 | Guidance 시작 step | 늦출수록 덜 개입 |
| `GUIDANCE_END_STEP` | 50 | Guidance 종료 step | 일찍 끝낼수록 덜 개입 |

### 권장 설정

**Benign 데이터 (quality 중시)**:
```bash
HARMFUL_THRESHOLD=0.7      # 높게
GUIDANCE_SCALE=3.0         # 낮게
SPATIAL_PERCENTILE=0.2     # 작게
```

**Harmful 데이터 (safety 중시)**:
```bash
HARMFUL_THRESHOLD=0.3      # 낮게
GUIDANCE_SCALE=7.0         # 높게
SPATIAL_PERCENTILE=0.4     # 크게
```

**Balanced (권장)**:
```bash
HARMFUL_THRESHOLD=0.5
GUIDANCE_SCALE=5.0
SPATIAL_PERCENTILE=0.3
```

---

## 📊 기대 결과

### Log 출력 예시

```
[Prompt 1/50] a person at the beach
[Step 10] Harmful detected (score=0.723) - Applying guidance
  Mask ratio: 28.5% of latent
...
[Step 35] Safe (score=0.412) - Skipping guidance
...

Overall Selective Guidance Statistics:
  Total denoising steps: 2500
  Harmful detected: 650 (26.0%)
  Guidance applied: 650 (26.0%)
  Steps saved (no guidance): 1850 (74.0%)
```

### 해석
- **Benign prompts**: `Harmful detected` < 10% 기대
- **Harmful prompts**: `Harmful detected` > 50% 기대
- **Steps saved**: 효율성 지표 (높을수록 좋음)

---

## 🔬 실험 체크리스트

### Phase 1: 구현 검증
- [x] 코드 작성 완료
- [x] Syntax check 통과
- [ ] **빠른 테스트 실행** (`./test_selective_cg.sh`)
- [ ] 에러 없이 이미지 생성 확인

### Phase 2: Baseline 비교
- [ ] Vanilla SD 실행 (비교군)
- [ ] Always-on masking 실행 (`generate_classifier_masked.sh`)
- [ ] Selective CG 실행 (`run_selective_cg.sh`)

### Phase 3: Benign 데이터 평가
- [ ] Benign prompts로 생성 (`run_selective_cg_benign.sh`)
- [ ] GENEVAL score 측정
- [ ] CLIP score 측정
- [ ] Baseline 대비 score 비교

### Phase 4: Harmful 데이터 평가
- [ ] Harmful prompts로 생성
- [ ] NSFW detection rate 측정
- [ ] 수동 검사 (safety)

### Phase 5: 파라미터 최적화
- [ ] Threshold sweep (0.3, 0.5, 0.7)
- [ ] Guidance scale sweep (3.0, 5.0, 7.0)
- [ ] 최적 조합 찾기

---

## 🐛 예상 이슈 & 해결

### Issue 1: Import Error

**증상**:
```
ImportError: cannot import name 'SelectiveGuidanceMonitor'
```

**해결**:
```bash
# Python path 확인
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue 2: Classifier Checkpoint Not Found

**증상**:
```
FileNotFoundError: ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth
```

**해결**:
```bash
# Checkpoint 경로 확인
ls -lh ./work_dirs/nudity_three_class/checkpoint/*/classifier.pth

# 올바른 경로로 수정
# run_selective_cg.sh에서 CLASSIFIER_CKPT 수정
```

### Issue 3: CUDA OOM

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결**:
```bash
# 1. FP16 사용 (이미 적용됨)
# 2. Batch size 줄이기
NSAMPLES=1

# 3. Visualization 끄기
SAVE_VISUALIZATIONS=false

# 4. 다른 GPU 사용
export CUDA_VISIBLE_DEVICES=7
```

### Issue 4: 너무 많은 Intervention (Benign에서)

**증상**: Benign prompts에서 `Harmful detected` > 30%

**해결**:
```bash
# Threshold 올리기
HARMFUL_THRESHOLD=0.7

# 또는 guidance window 좁히기
GUIDANCE_START_STEP=10
GUIDANCE_END_STEP=40
```

---

## 📈 다음 단계

### 1. 즉시 실행 (오늘)
```bash
# 구현 검증
./test_selective_cg.sh

# 성공하면 본 실험
./run_selective_cg.sh &
tail -f ./logs/run_selective_cg_*.log
```

### 2. 결과 분석 (내일)
- Log에서 guidance statistics 확인
- Visualization 분석
- 생성된 이미지 수동 검사

### 3. 평가 (이번 주)
- GENEVAL 실행
- Baseline 대비 비교
- 보고서 작성

---

## 🔗 관련 파일

### 참고할 기존 코드
- `generate_classifier_masked.py`: Always-on masking 방식
- `geo_utils/classifier_interpretability.py`: Grad-CAM 구현
- `geo_utils/guidance_utils.py`: Classifier guidance 기본 로직

### 생성된 새 파일
- `generate_selective_cg.py` ⭐
- `geo_utils/selective_guidance_utils.py` ⭐
- `run_selective_cg.sh` ⭐
- `run_selective_cg_benign.sh` ⭐
- `test_selective_cg.sh` ⭐

---

## 💡 핵심 차별점 요약

| 구분 | Always-On Masking | **Selective CG (새로운 방식)** |
|------|------------------|-------------------------------|
| **언제 적용?** | 모든 timestep | Harmful 감지 시에만 |
| **Benign 영향** | ❌ 불필요한 훼손 | ✅ 최소 개입 |
| **GENEVAL** | ⬇️ 저하 | ⬆️ 유지 |
| **Safety** | ✅ 높음 | ✅ 높음 |
| **효율성** | ❌ 모든 step 계산 | ✅ Selective 계산 |

---

## ✅ 최종 체크

구현 완료 확인:
- [x] 유틸리티 함수 작성
- [x] 메인 스크립트 작성
- [x] 실행 스크립트 작성
- [x] 테스트 스크립트 작성
- [x] 문서 작성
- [x] Syntax check 통과

다음 할 일:
1. **`./test_selective_cg.sh` 실행** ← 지금 바로!
2. 결과 확인
3. 본 실험 시작
4. 평가 및 분석

---

**구현 완료!** 🎉

이제 실험을 시작할 준비가 되었습니다.
`./test_selective_cg.sh` 부터 실행해보세요!
