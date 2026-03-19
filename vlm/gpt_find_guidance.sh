#!/usr/bin/env bash
set -euo pipefail          # 오류 즉시 중단 + 미정의 변수 방지

# --------- 공통 환경 변수 ---------
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# --------- 순차 처리 대상 폴더 목록 ---------
DIR_LIST=(
    "/mnt/home/yhgil99/safe-diffusion/images/esd_monet_countryNudebody"
    "/mnt/home/yhgil99/safe-diffusion/images/sdd_monet_countryNudebody"
)

# --------- 메인 루프 ---------
for IMG_DIR in "${DIR_LIST[@]}"; do
    echo "▶️  시작: ${IMG_DIR}"
    python gpt.py "${IMG_DIR}"
    echo "✅ 완료: ${IMG_DIR}"
    echo "-----------------------------------------------"
    sleep 10
done

echo "🎉 모든 폴더 처리 완료"
