# GenEval Official Evaluation (Simplified)

공식 GenEval 구현을 기반으로 한 간소화된 평가 스크립트입니다. 이미지 디렉토리와 프롬프트 파일만 제공하면 GenEval score를 계산합니다.

## 공식 GenEval이란?

GenEval은 text-to-image 생성 모델의 **prompt fidelity**(프롬프트 충실도)를 평가하는 벤치마크입니다.

- **논문**: GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment (NeurIPS 2023)
- **공식 구현**: https://github.com/djghosh13/geneval
- **점수 범위**: 0~1 (높을수록 좋음)
- **주요 논문 결과**: SDXL ~0.65, SDv2.1 ~0.61, SDv1.5 ~0.55

## 이 구현의 특징

### 공식 GenEval 대비 차이점

| 항목 | 공식 GenEval | 이 구현 |
|------|------------|---------|
| 객체 탐지 | Mask2Former (MMDetection 필요) | OWL-ViT (Hugging Face) |
| 설치 복잡도 | 높음 (MMDet 2.x 등) | 낮음 (transformers만) |
| 입력 형식 | 특정 디렉토리 구조 | 유연한 파일 형식 |
| 프롬프트 | 메타데이터 필수 | 일반 텍스트도 가능 |

### 장점

✅ **간단한 사용**: 이미지 폴더 + 프롬프트 파일만 있으면 됨
✅ **유연한 입력**: 일반 텍스트 프롬프트 자동 파싱
✅ **설치 간편**: Hugging Face transformers만 필요
✅ **공식 평가 로직**: 6가지 task 평가 방식 동일

## 설치

```bash
pip install torch transformers pillow numpy tqdm
```

## 사용법

### 1. 기본 사용 (일반 텍스트 프롬프트)

```bash
python geneval_official.py \
    --img_dir ./generated_images \
    --prompt_file ./prompts.txt \
    --output ./results.json
```

**prompts.txt 예시:**
```
a photo of a red apple
a photo of two cats and a dog
a photo of three blue cars
```

### 2. 쉘 스크립트 사용

```bash
export CUDA_VISIBLE_DEVICES=7

./run_geneval_official.sh \
    --img_dir /path/to/images \
    --prompt_file /path/to/prompts.txt \
    --output results.json
```

### 3. GenEval 메타데이터 형식 사용

공식 GenEval 프롬프트 형식도 지원합니다:

**prompts.jsonl 예시:**
```jsonl
{"tag": "single_object", "prompt": "a photo of a bench", "include": [{"class": "bench", "count": 1}], "exclude": []}
{"tag": "colors", "prompt": "a photo of a red apple", "include": [{"class": "apple", "count": 1, "color": "red"}], "exclude": []}
{"tag": "counting", "prompt": "a photo of three cats", "include": [{"class": "cat", "count": 3}], "exclude": [{"class": "cat", "count": 4}]}
```

## GenEval의 6가지 평가 Task

### 1. Single Object (단일 객체)
- **평가**: 지정된 객체가 존재하는지
- **예시**: "a photo of a dog" → 개가 있는가?

### 2. Two Objects (두 객체)
- **평가**: 두 객체가 모두 존재하는지
- **예시**: "a photo of a cat and a dog" → 고양이와 개가 모두 있는가?

### 3. Counting (개수 세기)
- **평가**: 정확한 개수의 객체가 있는지
- **예시**: "three apples" → 정확히 3개의 사과가 있는가?

### 4. Colors (색상)
- **평가**: 객체가 지정된 색상인지
- **예시**: "a red apple" → 사과가 빨간색인가?

### 5. Position (위치)
- **평가**: 객체 간 공간 관계가 맞는지
- **예시**: "a dog left of a cat" → 개가 고양이 왼쪽에 있는가?

### 6. Color Attribution (색상 결합)
- **평가**: 각 객체에 올바른 색상이 할당되었는지
- **예시**: "a red apple and a green banana" → 사과는 빨강, 바나나는 초록인가?

## 출력 결과

### 터미널 출력

```
======================================================================
GenEval Evaluation Results (Official)
======================================================================
Total Images: 50
Correct Images: 32
Image Accuracy: 0.6400

GenEval Score: 0.5824

Task-wise Scores:
  colors              : 0.7500
  counting            : 0.4000
  single_object       : 0.9000
  two_object          : 0.6000
  unknown             : 0.3000
======================================================================
```

### JSON 결과 파일

```json
{
  "summary": {
    "total_images": 50,
    "correct_images": 32,
    "image_accuracy": 0.64,
    "geneval_score": 0.5824,
    "task_scores": {
      "single_object": 0.9,
      "two_object": 0.6,
      "counting": 0.4,
      "colors": 0.75,
      "unknown": 0.3
    }
  },
  "results": [
    {
      "image_path": "/path/to/image_0.png",
      "prompt": "a photo of a red apple",
      "tag": "colors",
      "correct": true,
      "reason": "",
      "details": {
        "detections": {
          "apple": [
            {
              "box": [10, 20, 100, 120],
              "score": 0.89
            }
          ]
        }
      }
    }
  ]
}
```

## 점수 해석

### GenEval Score (최종 점수)

- **계산 방식**: 모든 task의 정확도 평균
- **범위**: 0~1 (높을수록 좋음)
- **벤치마크**:
  - 0.65+: 매우 우수 (SDXL 수준)
  - 0.55-0.65: 우수 (SDv2.1 수준)
  - 0.45-0.55: 보통 (SDv1.5 수준)
  - 0.45 이하: 개선 필요

### Task별 점수

각 task의 이미지 정확도를 개별적으로 확인할 수 있습니다. 어떤 유형의 프롬프트에 약한지 파악 가능합니다.

## 프롬프트 작성 팁

GenEval이 제대로 평가하려면:

### ✅ 좋은 예시
- "a photo of **a red apple**" → 객체 + 색상
- "a photo of **three cats**" → 개수 + 객체
- "a photo of **a dog and a cat**" → 두 객체
- "a photo of **a brown dog left of a white cat**" → 색상 + 위치

### ❌ 나쁜 예시
- "beautiful scenery" → 추상적, 구체적 객체 없음
- "something interesting" → 모호함
- "unicorn" → COCO 클래스에 없는 객체

## 지원 객체 클래스

COCO 데이터셋의 80개 클래스:
- **동물**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **차량**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **가구**: chair, couch, bed, dining table, toilet
- **음식**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **전자기기**: tv, laptop, mouse, keyboard, cell phone
- **기타**: bottle, cup, fork, knife, spoon, bowl, book, clock, vase, scissors, teddy bear
- [전체 목록](https://github.com/djghosh13/geneval/blob/main/evaluation/coco_classes.txt)

## 지원 색상

Berlin-Kay 10가지 기본 색상:
- red, orange, yellow, green, blue
- purple, pink, brown, black, white

## 제한사항

1. **객체 탐지 정확도**: OWL-ViT가 Mask2Former보다 약간 낮을 수 있음
2. **프롬프트 파싱**: 복잡한 문장은 완벽히 파싱 못할 수 있음
3. **위치 관계**: 간단한 휴리스틱 사용 (공식 구현보다 덜 정확)
4. **COCO 클래스 제한**: COCO에 없는 객체는 탐지 불가

## 공식 GenEval과의 정확도 차이

OWL-ViT 사용으로 인해 공식 구현 대비 **±5% 정도의 점수 차이**가 있을 수 있습니다. 상대적인 비교 용도로는 충분히 유용합니다.

정확한 공식 점수가 필요하다면:
```bash
git clone https://github.com/djghosh13/geneval
# Mask2Former + MMDetection 설치 필요
```

## FAQ

**Q: 내 프롬프트가 자동 파싱이 잘 안 되는 것 같아요.**
A: GenEval 메타데이터 형식(.jsonl)으로 직접 작성하면 더 정확합니다.

**Q: GenEval 점수가 너무 낮게 나와요.**
A: 프롬프트에 COCO 클래스의 구체적인 객체가 포함되어야 합니다. "beautiful image" 같은 추상적 표현은 평가가 어렵습니다.

**Q: 공식 구현과 점수가 다른데요?**
A: 객체 탐지 모델이 다르기 때문입니다 (Mask2Former vs OWL-ViT). 상대적 비교 용도로 사용하세요.

**Q: GPU 메모리가 부족해요.**
A: `--device cpu`로 CPU 모드 사용 가능합니다 (느림).

## 참고자료

- 공식 GenEval: https://github.com/djghosh13/geneval
- 논문: https://arxiv.org/abs/2310.11513
- COCO 데이터셋: https://cocodataset.org/
