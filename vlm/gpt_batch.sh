#!/usr/bin/env bash
set -euo pipefail          # 오류 즉시 중단 + 미정의 변수 방지

# --------- 공통 환경 변수 ---------
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# 평가할 루트 디렉터리 목록
ROOTS=(
  "/mnt/home/yhgil99/unlearning/sdxl-lightening-4classification/sdxl_guidance_out/CountryNudeBody/Cheating/Round2"
)

# 서브폴더(스케일) 목록
SCALES=(5 10 15 20 25)

# 메인 루프
for ROOT in "${ROOTS[@]}"; do
  for SCALE in "${SCALES[@]}"; do
    IMG_DIR="${ROOT}/${SCALE}"
    if [ ! -d "$IMG_DIR" ]; then
      echo "⚠️  폴더가 없습니다: $IMG_DIR"
      continue
    fi

    echo "▶️  VLM 평가 시작: ${IMG_DIR}"
    # 평가 스크립트 호출 (gpt.py 대신 실제 스크립트 이름으로)
    python gpt.py "${IMG_DIR}"
    echo "✅ 완료: ${IMG_DIR}"
    echo "-----------------------------------------------"
  done
done

echo "🎉 모든 폴더 VLM 평가 완료"