# GenEval 공식 데이터셋 정보

## 다운로드 완료!

공식 GenEval 저장소에서 프롬프트와 메타데이터를 다운로드했습니다.

### 파일 위치

```
/mnt/home/yhgil99/unlearning/vlm/
├── geneval_prompts.jsonl          # 공식 메타데이터 (553개)
├── geneval_prompts.txt             # 텍스트 프롬프트만 (553개)
└── geneval_official_repo/          # 전체 저장소
    ├── prompts/
    │   ├── evaluation_metadata.jsonl
    │   ├── generation_prompts.txt
    │   └── create_prompts.py
    ├── evaluation/
    └── ...
```

## GenEval 프롬프트 구성

### 총 553개 프롬프트, 6가지 태스크

| Task | 개수 | 설명 | 예시 |
|------|------|------|------|
| **single_object** | 80 | 단일 객체 생성 | "a photo of a bench" |
| **two_object** | 99 | 두 객체 생성 | "a photo of a cat and a dog" |
| **counting** | 80 | 개수 세기 | "a photo of three apples" |
| **colors** | 94 | 색상 속성 | "a photo of a red apple" |
| **position** | 100 | 위치 관계 | "a photo of a dog to the left of a cat" |
| **color_attr** | 100 | 색상 결합 | "a photo of a red apple and a green banana" |

## 메타데이터 형식 (JSONL)

```json
{
  "tag": "single_object",
  "include": [{"class": "bench", "count": 1}],
  "prompt": "a photo of a bench"
}
```

```json
{
  "tag": "counting",
  "include": [{"class": "apple", "count": 3}],
  "exclude": [{"class": "apple", "count": 4}],
  "prompt": "a photo of three apples"
}
```

```json
{
  "tag": "color_attr",
  "include": [
    {"class": "apple", "count": 1, "color": "red"},
    {"class": "banana", "count": 1, "color": "green"}
  ],
  "prompt": "a photo of a red apple and a green banana"
}
```

## 사용 방법

### 1. 메타데이터 형식으로 평가 (권장)

```bash
python geneval_official.py \
    --img_dir ./your_generated_images \
    --prompt_file geneval_prompts.jsonl \
    --output results.json
```

**주의**: 이미지 파일명이 프롬프트 순서와 일치해야 합니다.
- `image_0000.png` → 첫 번째 프롬프트
- `image_0001.png` → 두 번째 프롬프트
- ...

### 2. 텍스트 프롬프트로 평가

```bash
python geneval_official.py \
    --img_dir ./your_generated_images \
    --prompt_file geneval_prompts.txt \
    --output results.json
```

텍스트 파일 사용 시 자동으로 메타데이터로 파싱됩니다 (덜 정확).

## GenEval 점수 계산 방식

1. **각 task별로 정확도 계산**
   - single_object: 80개 중 몇 개 정답?
   - counting: 80개 중 몇 개 정답?
   - ...

2. **GenEval Score = 6개 task 정확도의 평균**

```
GenEval Score = (single_object + two_object + counting +
                 colors + position + color_attr) / 6
```

## 논문 벤치마크 점수

| 모델 | GenEval Score |
|------|---------------|
| SDXL | ~0.65 |
| SDv2.1 | ~0.61 |
| SDv1.5 | ~0.55 |
| SDv1.4 | ~0.54 |

## 이미지 생성 가이드

공식 GenEval 평가를 위해 이미지를 생성하려면:

```python
# generation_prompts.txt 사용
with open('geneval_prompts.txt') as f:
    prompts = [line.strip() for line in f]

for i, prompt in enumerate(prompts):
    # 이미지 생성
    image = your_model.generate(prompt)
    # 순서대로 저장 (중요!)
    image.save(f'generated_images/image_{i:04d}.png')
```

## 참고

- 공식 저장소: https://github.com/djghosh13/geneval
- 논문: https://arxiv.org/abs/2310.11513
- 원본 구현은 Mask2Former 사용 (더 정확하지만 설치 복잡)
- 우리 구현은 OWL-ViT 사용 (설치 간편하지만 약간 덜 정확)
