#!/bin/bash
# Watch for negative images completion, then launch Safe_Denoiser + SGF
BASE=/mnt/home3/yhgil99/unlearning
SD_DIR=$BASE/unlearning-baselines/Safe_Denoiser_official
SGF_DIR=$BASE/unlearning-baselines/SGF_official/nudity_sdv1
VENV_PY=$SD_DIR/.venv/bin/python3.10
NUDENET=$SD_DIR/pretrained/classifier_model.onnx
OUT=$BASE/unlearning-baselines/outputs
SD_DS=$SD_DIR/datasets
NEG_DIR=$SD_DIR/datasets/nudity/i2p_sexual

echo "[$(date)] Watcher started. Waiting for 500+ negative images..."

while true; do
    COUNT=$(ls $NEG_DIR/*.png 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 500 ]; then
        echo "[$(date)] $COUNT negative images ready. Launching experiments."
        break
    fi
    echo "[$(date)] $COUNT/500 images..."
    sleep 60
done

# Copy neg images to SGF
echo "[$(date)] Copying negative images to SGF..."
mkdir -p $SGF_DIR/datasets/nudity/i2p_sexual
cp $NEG_DIR/*.png $SGF_DIR/datasets/nudity/i2p_sexual/

echo "[$(date)] Launching Safe_Denoiser on GPUs 2,3..."

# GPU 2: Safe_Denoiser Ring-A-Bell + P4DN
CUDA_VISIBLE_DEVICES=2 $VENV_PY $SD_DIR/run_nudity.py \
    --config $SD_DIR/configs/base/vanilla/safree_neg_prompt_config.json \
    --data $SD_DS/nudity-ring-a-bell.csv \
    --save-dir $OUT/safe_denoiser/ringabell \
    --erase_id safree_neg_prompt_rep_threshold_time \
    --category nudity \
    --task_config $SD_DIR/configs/nudity/safe_denoiser.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sd_gpu2_ringabell.log

CUDA_VISIBLE_DEVICES=2 $VENV_PY $SD_DIR/run_nudity.py \
    --config $SD_DIR/configs/base/vanilla/safree_neg_prompt_config.json \
    --data $SD_DS/p4dn_16_prompt.csv \
    --save-dir $OUT/safe_denoiser/p4dn \
    --erase_id safree_neg_prompt_rep_threshold_time \
    --category nudity \
    --task_config $SD_DIR/configs/nudity/safe_denoiser.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sd_gpu2_p4dn.log &
PID_SD1=$!

# GPU 3: Safe_Denoiser MMA + UnlearnDiff
(CUDA_VISIBLE_DEVICES=3 $VENV_PY $SD_DIR/run_nudity.py \
    --config $SD_DIR/configs/base/vanilla/safree_neg_prompt_config.json \
    --data $SD_DS/mma-diffusion-nsfw-adv-prompts.csv \
    --save-dir $OUT/safe_denoiser/mma \
    --erase_id safree_neg_prompt_rep_threshold_time \
    --category nudity \
    --task_config $SD_DIR/configs/nudity/safe_denoiser.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sd_gpu3_mma.log && \
CUDA_VISIBLE_DEVICES=3 $VENV_PY $SD_DIR/run_nudity.py \
    --config $SD_DIR/configs/base/vanilla/safree_neg_prompt_config.json \
    --data $SD_DS/nudity.csv \
    --save-dir $OUT/safe_denoiser/unlearndiff \
    --erase_id safree_neg_prompt_rep_threshold_time \
    --category nudity \
    --task_config $SD_DIR/configs/nudity/safe_denoiser.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sd_gpu3_unlearndiff.log) &
PID_SD2=$!

echo "[$(date)] Launching SGF on GPUs 5,6..."

# GPU 5: SGF Ring-A-Bell + P4DN
(cd $SGF_DIR && \
CUDA_VISIBLE_DEVICES=5 $VENV_PY generate_unsafe_sgf.py \
    --config configs/base/safree_neg_prompt_config.json \
    --data $SD_DS/nudity-ring-a-bell.csv \
    --save-dir $OUT/sgf/ringabell \
    --erase_id safree_neg_prompt_rep_time \
    --category nudity \
    --task_config configs/sgf/sgf.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sgf_gpu5_ringabell.log && \
CUDA_VISIBLE_DEVICES=5 $VENV_PY generate_unsafe_sgf.py \
    --config configs/base/safree_neg_prompt_config.json \
    --data $SD_DS/p4dn_16_prompt.csv \
    --save-dir $OUT/sgf/p4dn \
    --erase_id safree_neg_prompt_rep_time \
    --category nudity \
    --task_config configs/sgf/sgf.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sgf_gpu5_p4dn.log) &
PID_SGF1=$!

# Wait for SDErasure GPU6 to finish, then use GPU 6 for SGF MMA + UnlearnDiff
while nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null | grep -q "00:A1"; do
    sleep 30
done

(cd $SGF_DIR && \
CUDA_VISIBLE_DEVICES=6 $VENV_PY generate_unsafe_sgf.py \
    --config configs/base/safree_neg_prompt_config.json \
    --data $SD_DS/mma-diffusion-nsfw-adv-prompts.csv \
    --save-dir $OUT/sgf/mma \
    --erase_id safree_neg_prompt_rep_time \
    --category nudity \
    --task_config configs/sgf/sgf.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sgf_gpu6_mma.log && \
CUDA_VISIBLE_DEVICES=6 $VENV_PY generate_unsafe_sgf.py \
    --config configs/base/safree_neg_prompt_config.json \
    --data $SD_DS/nudity.csv \
    --save-dir $OUT/sgf/unlearndiff \
    --erase_id safree_neg_prompt_rep_time \
    --category nudity \
    --task_config configs/sgf/sgf.yaml \
    --safe_level MEDIUM \
    --device cuda:0 \
    --nudenet-path $NUDENET \
    --nudity_thr 0.6 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    2>&1 | tee $BASE/logs/sgf_gpu6_unlearndiff.log) &
PID_SGF2=$!

echo "[$(date)] All launched. Waiting..."
wait $PID_SD1 && echo "[$(date)] Safe_Denoiser GPU2 done" || echo "[$(date)] Safe_Denoiser GPU2 FAILED"
wait $PID_SD2 && echo "[$(date)] Safe_Denoiser GPU3 done" || echo "[$(date)] Safe_Denoiser GPU3 FAILED"
wait $PID_SGF1 && echo "[$(date)] SGF GPU5 done" || echo "[$(date)] SGF GPU5 FAILED"
wait $PID_SGF2 && echo "[$(date)] SGF GPU6 done" || echo "[$(date)] SGF GPU6 FAILED"
echo "[$(date)] ALL EXPERIMENTS COMPLETE"
