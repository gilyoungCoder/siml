#!/bin/bash
# Violence 13-class Qwen2-VL Evaluation

if [ $# -lt 1 ]; then
    echo "Usage: $0 <GPU>"
    exit 1
fi

GPU=$1
export CUDA_VISIBLE_DEVICES=${GPU}

cd /mnt/home/yhgil99/unlearning

VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"

echo "=============================================="
echo "Violence 13-class Qwen2-VL Evaluation"
echo "GPU: ${GPU}"
echo "=============================================="

# SCG violence 13-class step28400 skip (먼저 평가)
echo ""
echo ">>> SCG violence 13-class step28400 skip (360 folders)"
for folder in /mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/*/; do
    [ -d "$folder" ] || continue
    if [ ! -f "$folder/results_qwen2_vl_violence.txt" ]; then
        echo "[EVAL] $(basename $folder)"
        python "$VLM_SCRIPT" "$folder" violence qwen
    else
        echo "[SKIP] $(basename $folder)"
    fi
done

# SCG violence 13-class step28400 (non-skip)
echo ""
echo ">>> SCG violence 13-class step28400 (199 folders)"
for folder in /mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/*/; do
    [ -d "$folder" ] || continue
    if [ ! -f "$folder/results_qwen2_vl_violence.txt" ]; then
        echo "[EVAL] $(basename $folder)"
        python "$VLM_SCRIPT" "$folder" violence qwen
    else
        echo "[SKIP] $(basename $folder)"
    fi
done

echo ""
echo "=============================================="
echo "Violence 13-class Evaluation Complete!"
echo "=============================================="
