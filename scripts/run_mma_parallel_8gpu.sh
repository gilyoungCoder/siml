#!/bin/bash
# MMA 300-1000 (700개), GPU 8개 병렬, ~88개씩
# steps 25로 빠르게

BASE="/mnt/home/yhgil99/unlearning"
SAFREE_DIR="${BASE}/SAFREE"
MMA_PROMPT="${BASE}/prompts/nudity_datasets/mma.txt"
OUTPUT_BASE="${BASE}/outputs/nudity_datasets/mma/safree_regenerated"
TEMP_DIR="${BASE}/prompts/nudity_datasets/mma_split"
LOG_DIR="${BASE}/logs"

mkdir -p "${TEMP_DIR}" "${LOG_DIR}" "${OUTPUT_BASE}"

# 300-1000, 88개씩 분할
sed -n '300,387p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_300_387.txt"
sed -n '388,475p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_388_475.txt"
sed -n '476,563p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_476_563.txt"
sed -n '564,651p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_564_651.txt"
sed -n '652,739p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_652_739.txt"
sed -n '740,827p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_740_827.txt"
sed -n '828,915p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_828_915.txt"
sed -n '916,1000p' "${MMA_PROMPT}" > "${TEMP_DIR}/mma_916_1000.txt"

cd "${SAFREE_DIR}"

CUDA_VISIBLE_DEVICES=0 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_300_387.txt" \
    --save-dir "${OUTPUT_BASE}/300_387" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_300_387.log" 2>&1 &
echo "GPU0: 300-387 - PID $!"

CUDA_VISIBLE_DEVICES=1 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_388_475.txt" \
    --save-dir "${OUTPUT_BASE}/388_475" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_388_475.log" 2>&1 &
echo "GPU1: 388-475 - PID $!"

CUDA_VISIBLE_DEVICES=2 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_476_563.txt" \
    --save-dir "${OUTPUT_BASE}/476_563" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_476_563.log" 2>&1 &
echo "GPU2: 476-563 - PID $!"

CUDA_VISIBLE_DEVICES=3 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_564_651.txt" \
    --save-dir "${OUTPUT_BASE}/564_651" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_564_651.log" 2>&1 &
echo "GPU3: 564-651 - PID $!"

CUDA_VISIBLE_DEVICES=4 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_652_739.txt" \
    --save-dir "${OUTPUT_BASE}/652_739" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_652_739.log" 2>&1 &
echo "GPU4: 652-739 - PID $!"

CUDA_VISIBLE_DEVICES=5 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_740_827.txt" \
    --save-dir "${OUTPUT_BASE}/740_827" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_740_827.log" 2>&1 &
echo "GPU5: 740-827 - PID $!"

CUDA_VISIBLE_DEVICES=6 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_828_915.txt" \
    --save-dir "${OUTPUT_BASE}/828_915" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_828_915.log" 2>&1 &
echo "GPU6: 828-915 - PID $!"

CUDA_VISIBLE_DEVICES=7 nohup python gen_safree_single.py \
    --txt "${TEMP_DIR}/mma_916_1000.txt" \
    --save-dir "${OUTPUT_BASE}/916_1000" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 25 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${LOG_DIR}/mma_916_1000.log" 2>&1 &
echo "GPU7: 916-1000 - PID $!"

echo ""
echo "All 8 GPUs started! (300-1000, 701 prompts, steps=25)"
echo "Monitor: tail -f ${LOG_DIR}/mma_*.log"
