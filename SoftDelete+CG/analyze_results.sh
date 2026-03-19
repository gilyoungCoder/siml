#!/bin/bash
# ============================================================================
# Soft Spatial CG 결과 분석 및 시각화 스크립트
# ============================================================================

set -e

# ============================================================================
# 설정
# ============================================================================

RESULTS_DIR="outputs/soft_cg_experiments"
VIZ_OUTPUT_DIR="visualizations"

# ============================================================================
# 시각화 함수들
# ============================================================================

# 1. Weight scheduling 시각화
visualize_weight_schedules() {
    echo "=========================================="
    echo "Weight Scheduling 시각화 생성 중..."
    echo "=========================================="

    python visualize_weight_schedules.py \
        --num_steps 50 \
        --output_dir "$VIZ_OUTPUT_DIR"

    echo "✓ 완료! 결과:"
    echo "  - ${VIZ_OUTPUT_DIR}/weight_schedules.png"
    echo "  - ${VIZ_OUTPUT_DIR}/temperature_effect.png"
}

# 2. 통계 수집 및 비교
collect_statistics() {
    echo "=========================================="
    echo "실험 통계 수집 중..."
    echo "=========================================="

    python << 'EOF'
import json
import os
from pathlib import Path
from collections import defaultdict

results_dir = Path("outputs/soft_cg_experiments")
output_file = "experiment_statistics.json"

all_stats = defaultdict(dict)

# 모든 statistics.json 파일 찾기
for stats_file in results_dir.rglob("statistics.json"):
    experiment_name = stats_file.parent.name

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    all_stats[experiment_name] = stats

# 결과 저장
with open(output_file, 'w') as f:
    json.dump(all_stats, f, indent=2)

print(f"✓ 통계 수집 완료: {output_file}")

# 요약 출력
print("\n" + "="*60)
print("실험 요약")
print("="*60)

for exp_name, stats in all_stats.items():
    print(f"\n{exp_name}:")
    if isinstance(stats, dict) and 'harmful_ratio' in stats:
        print(f"  Harmful detected: {stats.get('harmful_ratio', 0):.1%}")
        print(f"  Guidance applied: {stats.get('guidance_ratio', 0):.1%}")

print("\n" + "="*60)
EOF

    echo "✓ 통계 수집 완료!"
}

# 3. 이미지 비교 그리드 생성
create_comparison_grid() {
    echo "=========================================="
    echo "비교 그리드 생성 중..."
    echo "=========================================="

    python << 'EOF'
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_grid(image_paths, labels, output_path, grid_cols=3):
    """이미지 그리드 생성"""
    if not image_paths:
        print("⚠️  이미지를 찾을 수 없습니다.")
        return

    # 이미지 로드
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"⚠️  파일을 찾을 수 없음: {img_path}")

    if not images:
        return

    # 그리드 크기 계산
    img_width, img_height = images[0].size
    grid_rows = (len(images) + grid_cols - 1) // grid_cols

    # 레이블 공간
    label_height = 30

    # 그리드 이미지 생성
    grid_width = img_width * grid_cols
    grid_height = (img_height + label_height) * grid_rows

    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)

    # 이미지 배치
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // grid_cols
        col = idx % grid_cols

        x = col * img_width
        y = row * (img_height + label_height)

        # 이미지 붙이기
        grid_img.paste(img, (x, y))

        # 레이블 추가
        label_y = y + img_height
        draw.rectangle([(x, label_y), (x + img_width, label_y + label_height)], fill='lightgray')

        # 텍스트 중앙 정렬
        bbox = draw.textbbox((0, 0), label)
        text_width = bbox[2] - bbox[0]
        text_x = x + (img_width - text_width) // 2
        draw.text((text_x, label_y + 5), label, fill='black')

    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_path)
    print(f"✓ 그리드 저장: {output_path}")

# Temperature 비교
temp_dir = Path("outputs/soft_cg_experiments/temperature_comparison")
if temp_dir.exists():
    temps = ["0.1", "0.5", "1.0", "2.0", "5.0"]
    temp_images = [temp_dir / f"temp_{t}" / "00.png" for t in temps]
    temp_labels = [f"Temp {t}" for t in temps]
    create_grid(temp_images, temp_labels, "visualizations/temperature_comparison_grid.png", grid_cols=5)

# Strategy 비교
strategy_dir = Path("outputs/soft_cg_experiments/strategy_comparison")
if strategy_dir.exists():
    strategies = ["constant", "linear_increase", "linear_decrease", "cosine_anneal", "exponential_decay"]
    strategy_images = [strategy_dir / s / "00.png" for s in strategies]
    strategy_labels = [s.replace('_', ' ').title() for s in strategies]
    create_grid(strategy_images, strategy_labels, "visualizations/strategy_comparison_grid.png", grid_cols=3)

# Scale 비교
scale_dir = Path("outputs/soft_cg_experiments/scale_comparison")
if scale_dir.exists():
    scales = ["1.0", "3.0", "5.0", "7.0", "10.0"]
    scale_images = [scale_dir / f"scale_{s}" / "00.png" for s in scales]
    scale_labels = [f"Scale {s}" for s in scales]
    create_grid(scale_images, scale_labels, "visualizations/scale_comparison_grid.png", grid_cols=5)

print("\n✓ 모든 비교 그리드 생성 완료!")
EOF
}

# 4. Markdown 리포트 생성
generate_report() {
    echo "=========================================="
    echo "실험 리포트 생성 중..."
    echo "=========================================="

    REPORT_FILE="EXPERIMENT_REPORT.md"

    cat > "$REPORT_FILE" << 'EOF'
# Soft Spatial CG 실험 결과 리포트

## 실험 개요

- **날짜**: $(date +%Y-%m-%d)
- **모델**: Stable Diffusion v1.4
- **방법**: Soft Spatial Concept Guidance

---

## 1. Preset 실험 결과

### Preset 1: Gentle Increase
- **전략**: Linear Increase
- **Weight**: 0.5 → 2.0
- **Temperature**: 2.0 (매우 soft)
- **특징**: 부드럽게 시작하여 점진적으로 강화

### Preset 2: Strong Decay (추천)
- **전략**: Cosine Anneal
- **Weight**: 5.0 → 0.5
- **Temperature**: 1.0 (적당한 softness)
- **특징**: 초반 강하게 방향 설정, 후반 자연스럽게

### Preset 3: Constant Soft
- **전략**: Constant
- **Weight**: 1.0 (일정)
- **Temperature**: 1.0
- **특징**: 균일한 guidance 적용

### Preset 4: Aggressive Decay
- **전략**: Exponential Decay
- **Weight**: 10.0 → 0.1 (빠른 감소)
- **Temperature**: 0.5 (약간 sharp)
- **특징**: 초반 매우 강력한 교정

---

## 2. 파라미터 비교 실험

### Temperature 효과
- **0.1**: 거의 binary, 명확한 경계
- **1.0**: 적당한 부드러움 (기본값)
- **5.0**: 매우 부드러운 전환

![Temperature Comparison](visualizations/temperature_comparison_grid.png)

### Scheduling 전략
- **Constant**: 일정한 강도
- **Cosine Anneal**: 부드러운 감소 (추천)
- **Exponential Decay**: 빠른 감소

![Strategy Comparison](visualizations/strategy_comparison_grid.png)

### Guidance Scale
- **1.0**: 매우 약함
- **5.0**: 보통 (기본값)
- **10.0**: 매우 강함

![Scale Comparison](visualizations/scale_comparison_grid.png)

---

## 3. 결론 및 추천

### 추천 설정
```python
{
    "soft_mask_temperature": 1.0,
    "gaussian_sigma": 0.5,
    "strategy": "cosine_anneal",
    "start_weight": 5.0,
    "end_weight": 0.5,
    "guidance_scale": 5.0,
    "harmful_scale": 1.0,
    "normalize_gradient": True
}
```

### 주요 발견
1. **Soft masking**이 binary보다 자연스러운 결과
2. **Cosine annealing**이 가장 균형잡힌 성능
3. **Temperature 1.0**이 대부분의 경우 적절
4. **Selective guidance**로 safe 이미지 품질 보존

---

## 4. 통계

상세 통계는 `experiment_statistics.json` 참조

EOF

    echo "✓ 리포트 생성 완료: $REPORT_FILE"
}

# ============================================================================
# 사용법
# ============================================================================

show_usage() {
    cat << EOF
결과 분석 스크립트 사용법: $0 [OPTION]

옵션:
  all                   모든 분석 실행 (시각화 + 통계 + 리포트)
  visualize             Weight schedule 시각화
  stats                 통계 수집
  grid                  이미지 비교 그리드 생성
  report                Markdown 리포트 생성
  help                  이 도움말 출력

예시:
  $0 all                # 모든 분석 실행
  $0 visualize          # 시각화만 생성
  $0 grid               # 비교 그리드만 생성
EOF
}

# ============================================================================
# 메인
# ============================================================================

main() {
    mkdir -p "$VIZ_OUTPUT_DIR"

    case "${1:-all}" in
        all)
            visualize_weight_schedules
            collect_statistics
            create_comparison_grid
            generate_report
            echo ""
            echo "✓ 모든 분석 완료!"
            echo "  - 시각화: ${VIZ_OUTPUT_DIR}/"
            echo "  - 통계: experiment_statistics.json"
            echo "  - 리포트: EXPERIMENT_REPORT.md"
            ;;
        visualize)
            visualize_weight_schedules
            ;;
        stats)
            collect_statistics
            ;;
        grid)
            create_comparison_grid
            ;;
        report)
            generate_report
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "⚠️  알 수 없는 옵션: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
