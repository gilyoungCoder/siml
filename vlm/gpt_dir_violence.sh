#!/usr/bin/env bash
set -euo pipefail          # 오류 즉시 중단 + 미정의 변수 방지

# --------- 공통 환경 변수 ---------
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# --------- 순차 처리 대상 폴더 목록 ---------
  # "/mnt/home/yhgil99/unlearning/3_classification_sd1.4TC/Continual/new/CNNB3class/addclothed40,15,0.1MMMnot append"
  # "/mnt/home/yhgil99/unlearning/3_classification_sd1.4TC/Continual/new/CNNB3class/addclothed40,15,0.1MMMno"
  # # "/mnt/home/yhgil99/unlearning/3_classification_sd1.4TC/img/country nude body, fully clothed"
  # "/mnt/home/yhgil99/unlearning/3_classification_sd1.4TC/Continual/new2/CNNB3class/addclothed40,15,0.1,EOTblock_False"
  # "/mnt/home/yhgil99/unlearning/3_classification_sd1.4TC/Continual/new2/CNNB3class/addclothed40,15,0.1,EOTblock_true"
#   /mnt/home/yhgil99/unlearning/SAFREE/safree_outputs/nudity_violence/nudity_prompts
# /mnt/home/yhgil99/unlearning/SAFREE/safree_outputs/nudity_violence/violence_prompts
# /mnt/home/yhgil99/unlearning/SAFREE/safree_outputs/nudity_violence/mixed_prompts

DIR_LIST=(
  /mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_cluster_spatial_violence/gs5.0_st0.3-0.7_hs2.0_bgs3.0
)

# --------- 메인 루프 ---------
for IMG_DIR in "${DIR_LIST[@]}"; do
    echo "▶️  시작: ${IMG_DIR}"
    python gpt_violence.py "${IMG_DIR}"
    echo "✅ 완료: ${IMG_DIR}"
    echo "-----------------------------------------------"
done

echo "🎉 모든 폴더 처리 완료"
