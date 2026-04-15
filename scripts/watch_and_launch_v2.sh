#!/bin/bash
BASE=/mnt/home3/yhgil99/unlearning
SD_DIR=$BASE/unlearning-baselines/Safe_Denoiser_official
SGF_DIR=$BASE/unlearning-baselines/SGF_official/nudity_sdv1
VENV_PY=$SD_DIR/.venv/bin/python3.10
SAFREE_PY=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
NUDENET=$SD_DIR/pretrained/classifier_model.onnx
OUT=$BASE/unlearning-baselines/outputs
SD_DS=$SD_DIR/datasets
NEG_DIR=$SD_DIR/datasets/nudity/i2p_sexual

echo "[$(date)] Watcher v2 started. Waiting for 500+ negative images..."
while true; do
    COUNT=$(ls $NEG_DIR/*.png 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 500 ]; then
        echo "[$(date)] $COUNT negative images ready!"
        break
    fi
    echo "[$(date)] $COUNT/500 images..."
    sleep 60
done

# Copy to SGF
mkdir -p $SGF_DIR/datasets/nudity/i2p_sexual
cp $NEG_DIR/*.png $SGF_DIR/datasets/nudity/i2p_sexual/
echo "[$(date)] Neg images copied to SGF."

# GPU 4: Safe_Denoiser all 4 datasets
echo "[$(date)] Starting Safe_Denoiser on GPU 4..."
for DNAME in ringabell mma unlearndiff p4dn; do
    case $DNAME in
        ringabell)  CSV=$SD_DS/nudity-ring-a-bell.csv ;;
        mma)        CSV=$SD_DS/mma-diffusion-nsfw-adv-prompts.csv ;;
        unlearndiff) CSV=$SD_DS/nudity.csv ;;
        p4dn)       CSV=$SD_DS/p4dn_16_prompt.csv ;;
    esac
    echo "[$(date)] Safe_Denoiser: $DNAME"
    cd $SD_DIR
    CUDA_VISIBLE_DEVICES=4 $VENV_PY run_nudity.py \
        --config configs/base/vanilla/safree_neg_prompt_config.json \
        --data $CSV \
        --save-dir $OUT/safe_denoiser/$DNAME \
        --erase_id safree_neg_prompt_rep_threshold_time \
        --category nudity \
        --task_config configs/nudity/safe_denoiser.yaml \
        --safe_level MEDIUM \
        --device cuda:0 \
        --nudenet-path $NUDENET \
        --nudity_thr 0.6 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        2>&1 | tee $BASE/logs/sd_gpu4_${DNAME}.log
    echo "[$(date)] Safe_Denoiser $DNAME done."
done &
PID_SD=$!

# GPU 5: SGF all 4 datasets
echo "[$(date)] Starting SGF on GPU 5..."
for DNAME in ringabell mma unlearndiff p4dn; do
    case $DNAME in
        ringabell)  CSV=$SD_DS/nudity-ring-a-bell.csv ;;
        mma)        CSV=$SD_DS/mma-diffusion-nsfw-adv-prompts.csv ;;
        unlearndiff) CSV=$SD_DS/nudity.csv ;;
        p4dn)       CSV=$SD_DS/p4dn_16_prompt.csv ;;
    esac
    echo "[$(date)] SGF: $DNAME"
    cd $SGF_DIR
    CUDA_VISIBLE_DEVICES=5 $VENV_PY generate_unsafe_sgf.py \
        --config configs/base/safree_neg_prompt_config.json \
        --data $CSV \
        --save-dir $OUT/sgf/$DNAME \
        --erase_id safree_neg_prompt_rep_time \
        --category nudity \
        --task_config configs/sgf/sgf.yaml \
        --safe_level MEDIUM \
        --device cuda:0 \
        --nudenet-path $NUDENET \
        --nudity_thr 0.6 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        2>&1 | tee $BASE/logs/sgf_gpu5_${DNAME}.log
    echo "[$(date)] SGF $DNAME done."
done &
PID_SGF=$!

wait $PID_SD && echo "[$(date)] Safe_Denoiser ALL DONE" || echo "[$(date)] Safe_Denoiser FAILED"
wait $PID_SGF && echo "[$(date)] SGF ALL DONE" || echo "[$(date)] SGF FAILED"
echo "[$(date)] ALL EXPERIMENTS COMPLETE"
