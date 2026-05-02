# 🆓 오픈소스 VLM 평가 가이드

GPT-4o 대신 **무료 오픈소스 VLM**을 사용하여 평가하는 방법입니다.

## 💰 비용 비교

| 방법 | 비용 | 속도 | 품질 | 설치 |
|------|------|------|------|------|
| **GPT-4o** | ~$500-600 | 빠름 | 최고 | 쉬움 (API 키만) |
| **Qwen2-VL 7B** ⭐ | **무료** | 중간 | 좋음 | 중간 (모델 다운로드) |
| **LLaVA-NeXT 7B** | **무료** | 느림 | 좋음 | 중간 |
| **InternVL2 8B** | **무료** | 느림 | 최고 | 중간 |

**권장: Qwen2-VL 7B** - 속도와 품질의 균형이 가장 좋음

---

## 🚀 빠른 시작

### 1단계: 필수 라이브러리 설치

```bash
# 기본 라이브러리
pip install torch torchvision transformers accelerate

# Qwen2-VL용 (권장)
pip install qwen-vl-utils

# 또는 전부 설치
pip install torch torchvision transformers accelerate qwen-vl-utils pillow
```

### 2단계: GPU 메모리 확인

```bash
# GPU 메모리 확인
nvidia-smi

# 필요 메모리:
# - Qwen2-VL 7B: ~16GB VRAM (권장)
# - LLaVA-NeXT 7B: ~16GB VRAM
# - InternVL2 8B: ~20GB VRAM
```

💡 **메모리 부족 시**: 4-bit quantization 사용 (아래 설명 참조)

### 3단계: 단일 폴더 테스트

```bash
# Qwen2-VL로 테스트 (권장)
python3 vlm/opensource_vlm_nudity.py \
    SoftDelete+CG/scg_outputs/grid_search_nudity/gs10.0_hs1.0_st0.2_ws3.0-0.5_ts0.0--1.5 \
    qwen

# Violence 테스트
python3 vlm/opensource_vlm_violence.py \
    SoftDelete+CG/scg_outputs/grid_search_violence/gs10.0_hs1.0_st0.2_ws3.0-0.5_ts0.0--1.5 \
    qwen
```

### 4단계: 배치 평가 실행

```bash
# batch_evaluate.py 수정하여 오픈소스 VLM 사용
# 아래 "배치 평가 설정" 섹션 참조
```

---

## 📊 모델 선택 가이드

### 1. **Qwen2-VL 7B** (추천 ⭐)

**장점:**
- 가장 빠른 속도
- 우수한 품질
- 메모리 효율적 (16GB VRAM)
- 한국어 지원 우수

**사용법:**
```bash
python3 vlm/opensource_vlm_nudity.py <img_dir> qwen
python3 vlm/opensource_vlm_violence.py <img_dir> qwen
```

### 2. **LLaVA-NeXT 7B**

**장점:**
- 좋은 품질
- 안정적인 성능

**단점:**
- Qwen2-VL보다 느림

**사용법:**
```bash
python3 vlm/opensource_vlm_nudity.py <img_dir> llava
python3 vlm/opensource_vlm_violence.py <img_dir> llava
```

### 3. **InternVL2 8B**

**장점:**
- 최고 품질 (GPT-4o에 가장 근접)
- State-of-the-art 성능

**단점:**
- 가장 느림
- 메모리 요구량 높음 (20GB VRAM)

**사용법:**
```bash
python3 vlm/opensource_vlm_nudity.py <img_dir> internvl
python3 vlm/opensource_vlm_violence.py <img_dir> internvl
```

---

## 💾 메모리 부족 시 해결책

### 옵션 1: 4-bit Quantization

코드에서 모델 로딩 부분을 수정:

```python
# 기존
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 4-bit로 변경
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

필요 라이브러리:
```bash
pip install bitsandbytes
```

### 옵션 2: CPU 사용 (매우 느림)

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float32,
    device_map="cpu"
)
```

### 옵션 3: 더 작은 모델 사용

```python
# Qwen2-VL 2B (더 작은 버전)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

---

## 🔄 배치 평가 설정

기존 `batch_evaluate.py`를 수정하여 오픈소스 VLM 사용:

### 방법 1: 스크립트 경로 변경

`batch_evaluate.py`의 189-190줄을 수정:

```python
# 기존
eval_script = 'vlm/gpt.py' if category == 'nudity' else 'vlm/gpt_violence.py'

# 오픈소스 VLM으로 변경
eval_script = 'vlm/opensource_vlm_nudity.py' if category == 'nudity' else 'vlm/opensource_vlm_violence.py'
```

### 방법 2: 새 배치 스크립트 생성

```bash
# run_batch_evaluation_opensource.sh 생성
cp run_batch_evaluation.sh run_batch_evaluation_opensource.sh
```

그리고 내부에서 `batch_evaluate.py` 대신 별도 스크립트 호출

---

## ⏱️ 속도 비교 (50개 이미지 기준)

| 모델 | 시간 | 총 시간 (1152 폴더) |
|------|------|---------------------|
| GPT-4o | ~2-3분 | ~40시간 |
| Qwen2-VL 7B | ~5-8분 | ~100시간 |
| LLaVA-NeXT 7B | ~8-12분 | ~160시간 |
| InternVL2 8B | ~10-15분 | ~200시간 |

💡 **팁**: 여러 GPU가 있다면 폴더를 나눠서 병렬 실행!

---

## 🎯 실행 예시

### 1. 단일 폴더 테스트
```bash
python3 vlm/opensource_vlm_nudity.py \
    SoftDelete+CG/scg_outputs/grid_search_nudity/gs10.0_hs1.0_st0.2_ws3.0-0.5_ts0.0--1.5 \
    qwen
```

### 2. 여러 폴더 순차 실행
```bash
for dir in SoftDelete+CG/scg_outputs/grid_search_nudity/*/; do
    echo "Processing: $dir"
    python3 vlm/opensource_vlm_nudity.py "$dir" qwen
done
```

### 3. nohup으로 백그라운드 실행
```bash
nohup bash -c '
for dir in SoftDelete+CG/scg_outputs/grid_search_nudity/*/; do
    python3 vlm/opensource_vlm_nudity.py "$dir" qwen
done
' > nohup_opensource_nudity.out 2>&1 &
```

---

## 🔧 문제 해결

### 1. CUDA Out of Memory
```bash
# 해결: 4-bit quantization 사용 (위 섹션 참조)
# 또는 더 작은 모델 사용
```

### 2. 모델 다운로드 느림
```bash
# HuggingFace mirror 사용
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. Import 에러
```bash
# 필수 라이브러리 재설치
pip install --upgrade transformers accelerate qwen-vl-utils
```

---

## 📈 품질 검증

오픈소스 VLM이 GPT-4o와 얼마나 일치하는지 확인:

```bash
# GPT-4o로 일부 샘플 평가
python3 vlm/gpt.py sample_dir/

# Qwen2-VL로 같은 샘플 평가
python3 vlm/opensource_vlm_nudity.py sample_dir/ qwen

# 결과 비교
diff sample_dir/results.txt sample_dir/categories_qwen2_vl.json
```

---

## ✅ 권장 워크플로우

1. **소규모 테스트** (10-20개 폴더)
   - GPT-4o와 Qwen2-VL 둘 다 실행
   - 결과 비교 및 일치도 확인

2. **품질 검증 통과 시**
   - Qwen2-VL로 전체 평가 진행
   - 비용 $0, 시간 ~100시간

3. **품질이 중요한 경우**
   - InternVL2 8B 사용 (GPT-4o에 가장 근접)
   - 또는 일부만 GPT-4o, 나머지는 오픈소스

---

## 💡 추가 팁

### 병렬 실행 (GPU 여러 개)
```bash
# GPU 0에서 nudity
CUDA_VISIBLE_DEVICES=0 python3 vlm/opensource_vlm_nudity.py dir1/ qwen &

# GPU 1에서 violence
CUDA_VISIBLE_DEVICES=1 python3 vlm/opensource_vlm_violence.py dir2/ qwen &
```

### 진행 상황 모니터링
```bash
# 완료된 폴더 개수 확인
find SoftDelete+CG/scg_outputs/grid_search_nudity/ -name "results.txt" | wc -l
```

### 재시작
```bash
# 이미 results.txt가 있는 폴더는 스킵하도록 수정 가능
```

---

## 🎉 결론

**GPT-4o 비용이 부담된다면 Qwen2-VL 7B 강력 추천!**

- ✅ 무료
- ✅ 좋은 품질
- ✅ 합리적인 속도
- ✅ 쉬운 설치

더 자세한 내용은 각 모델의 HuggingFace 페이지 참조:
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [LLaVA-NeXT](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-8B)
