#!/bin/bash
# GPU 4,5,6,7에서 병렬 실행
BASE="/mnt/home/yhgil99/unlearning"
cd "${BASE}/SAFREE"

mkdir -p "${BASE}/logs"

# GPU 4: I2P (142)
CUDA_VISIBLE_DEVICES=4 nohup python gen_safree_single.py \
    --txt "${BASE}/prompts/nudity_datasets/nudity.txt" \
    --save-dir "${BASE}/outputs/nudity_datasets/i2p/safree_regenerated" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 50 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${BASE}/logs/safree_i2p.log" 2>&1 &
echo "GPU4: I2P (142) - PID $!"

# GPU 5: Ring-a-Bell (79)
CUDA_VISIBLE_DEVICES=5 nohup python gen_safree_single.py \
    --txt "${BASE}/prompts/nudity_datasets/ringabell.txt" \
    --save-dir "${BASE}/outputs/nudity_datasets/ringabell/safree_regenerated" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 50 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${BASE}/logs/safree_ringabell.log" 2>&1 &
echo "GPU5: Ring-a-Bell (79) - PID $!"

# GPU 6: MMA (1000)
CUDA_VISIBLE_DEVICES=6 nohup python gen_safree_single.py \
    --txt "${BASE}/prompts/nudity_datasets/mma.txt" \
    --save-dir "${BASE}/outputs/nudity_datasets/mma/safree_regenerated" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 50 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${BASE}/logs/safree_mma.log" 2>&1 &
echo "GPU6: MMA (1000) - PID $!"

# GPU 7: COCO (10k)
CUDA_VISIBLE_DEVICES=7 nohup python gen_safree_single.py \
    --txt "${BASE}/prompts/coco/coco_10k.txt" \
    --save-dir "${BASE}/outputs/coco/safree_regenerated" \
    --model_id "CompVis/stable-diffusion-v1-4" --seed 42 --num_inference_steps 50 --guidance_scale 7.5 \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    > "${BASE}/logs/safree_coco.log" 2>&1 &
echo "GPU7: COCO (10k) - PID $!"

echo ""
echo "Monitor: tail -f ${BASE}/logs/safree_*.log"
