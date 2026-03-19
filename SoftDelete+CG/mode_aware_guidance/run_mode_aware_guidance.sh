#!/bin/bash
# Mode-Aware Classifier Guidance 실행 스크립트
#
# 사용법:
#   chmod +x run_mode_aware_guidance.sh
#   ./run_mode_aware_guidance.sh

set -e  # 에러 발생 시 중단

# ===========================================
# 설정 (필요에 따라 수정)
# ===========================================
SD_CKPT="runwayml/stable-diffusion-v1-5"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
CLUSTER_OUTPUT="cluster_centroids/harmful_clusters.pt"
N_CLUSTERS=10
NUM_SAMPLES_PER_PROMPT=20

# ===========================================
# Step 1: Clustering (처음 한 번만 실행)
# ===========================================
echo "=============================================="
echo "Step 1: Harmful Latent Clustering"
echo "=============================================="

if [ -f "$CLUSTER_OUTPUT" ]; then
    echo "[SKIP] Cluster centroids already exist: $CLUSTER_OUTPUT"
    echo "       Delete this file to re-run clustering"
else
    echo "[RUN] Collecting harmful latents and clustering..."

    python cluster_harmful_latents.py \
        --mode generate \
        --ckpt_path "$SD_CKPT" \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --prompts \
            "nude person" \
            "naked body" \
            "unclothed figure" \
            "explicit scene" \
            "person without clothes" \
            "bare skin" \
            "revealing pose" \
            "intimate scene" \
        --num_samples_per_prompt $NUM_SAMPLES_PER_PROMPT \
        --n_clusters $N_CLUSTERS \
        --capture_timestep 500 \
        --confidence_threshold 0.3 \
        --output_path "$CLUSTER_OUTPUT" \
        --save_latents

    echo "[DONE] Clustering complete!"
fi

# ===========================================
# Step 2: Config 파일 업데이트
# ===========================================
echo ""
echo "=============================================="
echo "Step 2: Updating Config"
echo "=============================================="

CONFIG_FILE="configs/models/mode_aware_discriminator.yaml"

# centroids_path를 실제 경로로 업데이트
if grep -q "centroids_path: null" "$CONFIG_FILE"; then
    echo "[UPDATE] Setting centroids_path in config..."
    sed -i "s|centroids_path: null|centroids_path: \"$CLUSTER_OUTPUT\"|g" "$CONFIG_FILE"
    echo "[DONE] Config updated!"
else
    echo "[SKIP] Config already has centroids_path set"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Now you can run inference with mode-aware guidance:"
echo ""
echo "  python generate_classifier_masked.py your_ckpt_path \\"
echo "      --classifier_guidance \\"
echo "      --classifier_config configs/models/mode_aware_discriminator.yaml \\"
echo "      --classifier_ckpt $CLASSIFIER_CKPT \\"
echo "      --guidance_scale 5.0 \\"
echo "      --prompt_file your_prompts.txt \\"
echo "      --output_dir output_mode_aware"
echo ""
