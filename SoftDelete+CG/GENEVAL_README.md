# GenEval Score Evaluation

이미지 생성 모델의 품질을 평가하는 GenEval 스크립트입니다.

## 설치

```bash
pip install -r requirements_geneval.txt
```

## 사용법

### 기본 사용법

```bash
python geneval_score.py \
    --img_dir /path/to/generated/images \
    --prompt_file /path/to/prompts.txt \
    --output results.json
```

### 프롬프트 파일 형식

#### 1. 텍스트 파일 (.txt)
각 줄에 하나의 프롬프트를 작성:

```
A cat sitting on a chair
A beautiful sunset over the ocean
A futuristic city with flying cars
```

#### 2. JSON 파일 (.json)

**리스트 형식:**
```json
[
    "A cat sitting on a chair",
    "A beautiful sunset over the ocean",
    "A futuristic city with flying cars"
]
```

**딕셔너리 형식:**
```json
{
    "0": "A cat sitting on a chair",
    "1": "A beautiful sunset over the ocean",
    "2": "A futuristic city with flying cars"
}
```

### 이미지 디렉토리

이미지들은 알파벳/숫자 순서로 정렬되어 프롬프트와 매칭됩니다:

```
images/
├── 0000.png  -> 첫 번째 프롬프트
├── 0001.png  -> 두 번째 프롬프트
├── 0002.png  -> 세 번째 프롬프트
└── ...
```

지원 이미지 형식: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## 예시

```bash
# CUDA 사용
python geneval_score.py \
    --img_dir ./generated_images \
    --prompt_file ./prompts.json \
    --output ./eval_results.json \
    --device cuda

# CPU 사용
python geneval_score.py \
    --img_dir ./generated_images \
    --prompt_file ./prompts.txt \
    --output ./eval_results.json \
    --device cpu
```

## 출력 결과

결과는 JSON 파일로 저장되며 다음 정보를 포함합니다:

```json
{
  "total_samples": 100,
  "avg_clip_score": 28.5432,
  "std_clip_score": 2.1234,
  "avg_text_similarity": 0.7856,
  "std_text_similarity": 0.0432,
  "results": [
    {
      "index": 0,
      "image_path": "/path/to/image.png",
      "prompt": "A cat sitting on a chair",
      "generated_caption": "a cat is sitting on a wooden chair",
      "clip_score": 29.123,
      "text_similarity": 0.812
    },
    ...
  ]
}
```

### 평가 지표

- **CLIP Score**: 이미지와 텍스트 간의 정렬 정도 (높을수록 좋음)
- **Text Similarity**: 원본 프롬프트와 생성된 캡션 간의 유사도 (0~1, 높을수록 좋음)

## 참고사항

- 첫 실행 시 모델을 다운로드하므로 시간이 걸릴 수 있습니다
- CUDA 사용 시 GPU 메모리가 충분한지 확인하세요
- 프롬프트 개수와 이미지 개수가 다르면 적은 쪽에 맞춰 평가합니다
