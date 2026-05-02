# GenEval Score Evaluation (Improved Version)

공식 GenEval 구현을 기반으로 한 이미지 생성 품질 평가 스크립트입니다.

## GenEval vs CLIP Score

### CLIP Score
- **측정 방식**: 이미지와 텍스트의 전체적인 유사도 (단일 점수)
- **장점**: 빠르고 간단함
- **단점**: 세부적인 구성 요소(객체 개수, 색상, 위치 등)를 평가하지 못함
- **인간 평가 일치도**: 80%

### GenEval
- **측정 방식**: 객체 탐지 기반의 구성적(compositional) 평가
- **평가 항목**:
  - ✓ 객체 존재 여부
  - ✓ 객체 개수 정확도
  - ✓ 색상 속성 정확도
  - ✓ 위치 관계 (공간적 배치)
  - ✓ 속성 결합 (attribute binding)
  - ✓ CLIP Score (전체 유사도)
- **장점**: 세밀한 평가, 실패 원인 진단 가능, 높은 인간 일치도
- **인간 평가 일치도**: 83% (복잡한 태스크에서 더 높음)

### 이 구현의 특징

본 스크립트는 공식 GenEval의 핵심 기능을 구현합니다:

1. **OWL-ViT 객체 탐지**: 오픈 vocabulary 객체 탐지로 프롬프트에 명시된 객체 찾기
2. **CLIP 기반 색상 분류**: 탐지된 객체의 색상을 Berlin-Kay 11가지 기본 색상으로 분류
3. **개수 정확도**: 프롬프트에 명시된 객체 개수와 실제 탐지된 개수 비교
4. **CLIP Score**: 전체적인 이미지-텍스트 정렬도 측정
5. **종합 GenEval Score**: 위 메트릭들을 결합한 최종 점수

## 설치

```bash
pip install -r requirements_geneval.txt
```

필요한 패키지:
- `torch>=2.0.0`
- `transformers>=4.30.0` (CLIP, OWL-ViT 모델)
- `Pillow>=9.0.0`
- `numpy>=1.21.0`
- `tqdm>=4.60.0`

## 사용법

### 1. 쉘 스크립트 사용 (권장)

```bash
# GPU 7번 사용 예시
export CUDA_VISIBLE_DEVICES=7

./run_geneval.sh \
    --img_dir /path/to/images \
    --prompt_file /path/to/prompts.txt \
    --output results.json
```

### 2. Python 직접 실행

```bash
python geneval_score.py \
    --img_dir ./generated_images \
    --prompt_file ./prompts.txt \
    --output ./geneval_results.json \
    --device cuda
```

## 프롬프트 파일 형식

### TXT 파일
```
A red apple on a table
Two cats sitting on a chair
Three blue birds in the sky
```

### JSON 파일
```json
[
    "A red apple on a table",
    "Two cats sitting on a chair",
    "Three blue birds in the sky"
]
```

또는

```json
{
    "0": "A red apple on a table",
    "1": "Two cats sitting on a chair",
    "2": "Three blue birds in the sky"
}
```

## 출력 결과 해석

```
======================================================================
GenEval Evaluation Results
======================================================================
Total Samples: 50

GenEval Score: 0.3245

CLIP Score: 26.8234 ± 5.2341
Color Accuracy: 0.7500 ± 0.1234 (20 samples)
Count Accuracy: 0.6000 ± 0.2000 (15 samples)
======================================================================
```

### 점수 의미

1. **GenEval Score** (0~1): 전체 종합 점수
   - CLIP Score (정규화), Color Accuracy, Count Accuracy의 평균
   - 높을수록 프롬프트를 잘 따름

2. **CLIP Score** (0~100): 이미지-텍스트 전체 유사도
   - 일반적으로 20~35 범위
   - 높을수록 프롬프트와 이미지가 유사

3. **Color Accuracy** (0~1): 색상 속성 정확도
   - 프롬프트에 색상이 명시된 경우만 평가
   - 예: "red apple" → 사과가 실제로 빨간색인지

4. **Count Accuracy** (0~1): 개수 정확도
   - 프롬프트에 개수가 명시된 경우만 평가
   - 예: "three cats" → 정확히 3마리의 고양이가 탐지되는지

## 결과 JSON 파일

```json
{
  "total_samples": 50,
  "geneval_score": 0.3245,
  "clip_score": {
    "mean": 26.8234,
    "std": 5.2341
  },
  "color_accuracy": {
    "mean": 0.75,
    "std": 0.1234,
    "num_evaluated": 20
  },
  "count_accuracy": {
    "mean": 0.6,
    "std": 0.2,
    "num_evaluated": 15
  },
  "results": [
    {
      "index": 0,
      "image_path": "/path/to/image.png",
      "prompt": "a red apple",
      "clip_score": 28.5,
      "object_detection": {
        "expected": ["apple"],
        "detected": ["apple"],
        "details": [...]
      },
      "color_accuracy": {
        "apple": {
          "expected": "red",
          "predicted": "red",
          "confidence": 0.89,
          "correct": true
        }
      }
    }
  ]
}
```

## 프롬프트 작성 팁

GenEval이 제대로 평가하려면 프롬프트에 다음 정보를 명시하세요:

### ✓ 좋은 예시
- "**three red** apples on a table" → 개수(3) + 색상(red) + 객체(apple)
- "**two blue** cars and **one yellow** bus" → 여러 객체 + 각각의 색상/개수
- "**a brown** dog sitting on **a green** chair" → 객체 + 색상 + 위치

### ✗ 나쁜 예시
- "beautiful scenery" → 추상적, 객체/색상 없음
- "something interesting" → 모호함
- 너무 긴 프롬프트 (CLIP 77 토큰 제한)

## 지원하는 색상

Berlin-Kay 11가지 기본 색상:
- red, orange, yellow, green, blue
- purple, pink, gray, brown, black, white

## 제한사항

1. **객체 파싱**: 휴리스틱 기반이므로 복잡한 문장은 제대로 파싱 못할 수 있음
2. **위치 평가**: 현재 구현에서는 미지원 (향후 추가 예정)
3. **OWL-ViT 성능**: Mask2Former보다 정확도가 낮을 수 있음
4. **메모리**: 큰 모델들을 GPU에 로드하므로 충분한 VRAM 필요 (~8GB)

## 참고자료

- 공식 GenEval: https://github.com/djghosh13/geneval
- 논문: GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment (NeurIPS 2023)
- ArXiv: https://arxiv.org/abs/2310.11513
