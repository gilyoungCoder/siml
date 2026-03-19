# 🚀 빠른 시작 가이드

## 현재 상황

기존 `generate_selective_cg.py`는 Soft Spatial CG의 기본 구조를 가지고 있지만,
새로 추가한 **soft masking**과 **weight scheduling** 기능을 사용하려면 코드를 수정해야 합니다.

---

## ✅ 지금 바로 사용 가능한 스크립트

### 1. `run_soft_cg_fixed.sh` - 기존 코드 호환 버전

```bash
# 추천 preset 실행
./run_soft_cg_fixed.sh strong

# 모든 preset 실행
./run_soft_cg_fixed.sh all-presets

# Guidance scale 비교
./run_soft_cg_fixed.sh scale
```

**특징:**
- ✅ 기존 `generate_selective_cg.py`와 바로 호환
- ✅ Bidirectional guidance 사용
- ✅ Spatial masking 사용
- ⚠️  Soft masking과 weight scheduling은 아직 미구현 (코드 수정 필요)

---

### 2. `batch_generate_from_prompts.sh` - 배치 생성 (수정 필요)

현재는 새로운 인자 구조를 사용하도록 작성되어 있어, 기존 코드와 바로 호환되지 않습니다.

**임시 해결책**: 수동으로 프롬프트 파일 생성

```bash
# 1. 프롬프트 파일 만들기
cat > my_prompts.txt << EOF
a nude woman on the beach
a woman in a red dress on the beach
naked people at a party
EOF

# 2. 기존 generate_selective_cg.py로 실행
python generate_selective_cg.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file my_prompts.txt \
    --output_dir outputs/my_batch \
    --nsamples 1 \
    --selective_guidance \
    --classifier_ckpt checkpoints/nude_classifier_best.pth \
    --guidance_scale 5.0 \
    --use_bidirectional
```

---

## 🔧 Soft Spatial CG 완전 구현을 위해 필요한 작업

### Option 1: 기존 코드 확장 (추천)

`generate_selective_cg.py`에 새로운 argument 추가:

```python
# parse_args()에 추가
parser.add_argument("--use_soft_mask", action="store_true",
                    help="Use soft spatial masking")
parser.add_argument("--soft_mask_temperature", type=float, default=1.0,
                    help="Temperature for soft masking")
parser.add_argument("--soft_mask_gaussian_sigma", type=float, default=0.0,
                    help="Gaussian smoothing sigma")

# Weight scheduling 추가
parser.add_argument("--use_weight_scheduling", action="store_true",
                    help="Use weight scheduling")
parser.add_argument("--weight_strategy", type=str, default="constant",
                    choices=["constant", "linear_increase", "linear_decrease",
                             "cosine_anneal", "exponential_decay"])
parser.add_argument("--weight_start_value", type=float, default=1.0)
parser.add_argument("--weight_end_value", type=float, default=1.0)
```

그리고 main 함수에서 `SelectiveGuidanceMonitor`와 `SpatiallyMaskedGuidance` 초기화시 사용:

```python
monitor = SelectiveGuidanceMonitor(
    classifier_model=classifier,
    use_soft_mask=args.use_soft_mask,
    soft_mask_temperature=args.soft_mask_temperature,
    soft_mask_gaussian_sigma=args.soft_mask_gaussian_sigma,
    ...
)

scheduler = WeightScheduler(
    strategy=args.weight_strategy,
    start_weight=args.weight_start_value,
    end_weight=args.weight_end_value,
    ...
) if args.use_weight_scheduling else None

guidance = SpatiallyMaskedGuidance(
    classifier_model=classifier,
    weight_scheduler=scheduler,
    ...
)
```

### Option 2: 새로운 스크립트 작성

`generate_soft_spatial_cg.py`를 새로 만들어서 모든 기능 포함.

---

## 📝 현재 사용 가능한 워크플로우

### 시나리오 1: 기본 Selective CG 테스트

```bash
# 1. 프롬프트 파일 만들기
cat > test.txt << EOF
a nude woman on the beach
a woman in a swimsuit
EOF

# 2. 실행
python generate_selective_cg.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file test.txt \
    --output_dir outputs/test \
    --selective_guidance \
    --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth \
    --guidance_scale 5.0 \
    --harmful_scale 1.5 \
    --use_bidirectional \
    --debug

# 3. 결과 확인
ls outputs/test/
```

### 시나리오 2: Bash 스크립트로 빠른 실험

```bash
# Strong Decay preset 실행
./run_soft_cg_fixed.sh strong

# 결과 확인
ls outputs/soft_cg_experiments/strong_decay/
```

### 시나리오 3: 시각화

```bash
# Weight scheduling 시각화 (Python 직접 실행)
python visualize_weight_schedules.py --output_dir visualizations/
```

---

## 💡 추천 다음 단계

### 즉시 실행 가능:
1. `./run_soft_cg_fixed.sh strong` - 기본 Selective CG 테스트
2. 결과 확인 및 파라미터 조정
3. `./run_soft_cg_fixed.sh scale` - Scale 비교 실험

### 완전한 Soft Spatial CG 사용하려면:
1. `generate_selective_cg.py`에 argument 추가 (위 Option 1 참조)
2. 또는 새로운 스크립트 작성
3. 그 후 모든 bash 스크립트 사용 가능

---

## 📚 관련 문서

현재 구현된 기능:
- ✅ `geo_utils/selective_guidance_utils.py` - Soft masking + Weight scheduling 구현됨
- ✅ `configs/soft_spatial_cg_tuning.yaml` - 설정 파일
- ✅ `configs/multi_concept_test_prompts.json` - 테스트 프롬프트
- ✅ `visualize_weight_schedules.py` - 시각화 도구

사용 방법:
- 📖 `README_SOFT_CG.md` - 빠른 시작
- 📚 `SOFT_SPATIAL_CG_GUIDE.md` - 상세 가이드
- 💻 `SCRIPTS_GUIDE.md` - Bash 스크립트 가이드
- 📝 `BATCH_GENERATION_GUIDE.md` - 배치 생성 가이드

---

## ⚠️  주의사항

**현재 상황:**
- Soft masking과 Weight scheduling은 `selective_guidance_utils.py`에 구현되어 있음
- 하지만 `generate_selective_cg.py`에서 사용하려면 argument 추가 필요
- `run_soft_cg_fixed.sh`는 기존 기능만 사용 (Bidirectional + Spatial masking)

**해결 방법:**
코드를 수정하거나, 현재 기능으로 먼저 실험 후 점진적으로 확장

---

## 🎯 간단 요약

**지금 바로 실행:**
```bash
./run_soft_cg_fixed.sh strong
```

**배치 생성 (수동):**
```bash
# 1. 프롬프트 파일 만들기
echo "a nude woman" > prompts.txt
echo "a woman in dress" >> prompts.txt

# 2. 실행
python generate_selective_cg.py CompVis/stable-diffusion-v1-4 \
    --prompt_file prompts.txt \
    --output_dir outputs/batch \
    --selective_guidance \
    --classifier_ckpt checkpoints/classifier.pth \
    --guidance_scale 5.0 \
    --use_bidirectional
```

**완전한 Soft CG 사용하려면:**
`generate_selective_cg.py`에 argument 추가 후 재실행
