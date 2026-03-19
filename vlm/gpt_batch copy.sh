#!/usr/bin/env bash
set -euo pipefail

# --------- 공통 환경 변수 ---------
##yhgil99@naver.com
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
BASE_IMG_DIR="/mnt/home/yhgil99/unlearning/10_classificaiton/Continual/CountryBodyPure"
LOG_FILE="${BASE_IMG_DIR}/combined_log.txt"
# --------- 백업 및 로그 초기화 ---------
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.bak_$(date +%Y%m%d_%H%M%S)"
fi

# --------- nohup으로 전체 실행 ---------
nohup bash -c '

DIR_LIST=(
  "'"$BASE_IMG_DIR"'/5"
  "'"$BASE_IMG_DIR"'/10"
  "'"$BASE_IMG_DIR"'/15"
  "'"$BASE_IMG_DIR"'/20"
  "'"$BASE_IMG_DIR"'/25"
)

for IMG_DIR in "${DIR_LIST[@]}"; do
    echo "▶️ 시작: ${IMG_DIR}" | tee -a "'"$LOG_FILE"'"
    python gpt.py "$IMG_DIR" >> "'"$LOG_FILE"'" 2>&1
    echo "✅ 완료: ${IMG_DIR}" | tee -a "'"$LOG_FILE"'"
    echo "-----------------------------------------------" | tee -a "'"$LOG_FILE"'"
done

echo "🎉 모든 폴더 처리 완료" | tee -a "'"$LOG_FILE"'"
' > /dev/null 2>&1 &
