#!/bin/bash
# GPT-4o로 safree_regenerated nudity datasets 평가
# Usage: bash eval_gpt4o_safree_regenerated.sh

BASE="/mnt/home/yhgil99/unlearning"
cd "${BASE}"

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

echo "=============================================="
echo "GPT-4o Evaluation (safree_regenerated)"
echo "=============================================="

for dataset in i2p mma ringabell; do
    DIR="${BASE}/outputs/nudity_datasets/${dataset}/safree_regenerated"
    if [ -d "${DIR}/generated" ]; then
        TARGET="${DIR}/generated"
    else
        TARGET="${DIR}"
    fi

    if [ -d "${TARGET}" ]; then
        echo ""
        echo "[${dataset}] ${TARGET}"
        python vlm/gpt.py "${TARGET}"
    else
        echo "[${dataset}] NOT FOUND: ${TARGET}"
    fi
done

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
