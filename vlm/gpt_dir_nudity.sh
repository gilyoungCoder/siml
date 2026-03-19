#!/usr/bin/env bash
set -euo pipefail          # 오류 즉시 중단 + 미정의 변수 방지

# --------- 공통 환경 변수 ---------
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# --------- 순차 처리 대상 폴더 목록 ---------
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/np/CNB_2class
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/np/CNB_3class
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/np/CNB_nocg
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/np/i2p_sexual_2class
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/np/i2p_sexual_3class
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/p/CNB_3class
  # /mnt/home/yhgil99/unlearning/SoftDelete+CG/SDCG/p/i2p_adaptive_3class
DIR_LIST=(
  /mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs7.5_thr0.7-0.3_hs0.5_bgs1.0_linear_decrease
  /mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs7.5_thr0.7-0.3_hs0.5_bgs1.0_cosine_anneal
)

# --------- 메인 루프 ---------
for IMG_DIR in "${DIR_LIST[@]}"; do
    echo "▶️  시작: ${IMG_DIR}"
    python gpt.py "${IMG_DIR}"
    echo "✅ 완료: ${IMG_DIR}"
    echo "-----------------------------------------------"
done

echo "🎉 모든 폴더 처리 완료"
