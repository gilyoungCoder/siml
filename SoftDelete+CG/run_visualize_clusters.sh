#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# Cluster Visualization Script
# 각 cluster의 latent를 완전히 denoise (t=0)한 후 시각화
#───────────────────────────────────────────────────────────────────────────────#

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/mnt/home/yhgil99/unlearning/SoftDelete+CG"

# Input paths
LATENTS_PATH="cluster_centroids/violence_clusters_latents.pt"
CENTROIDS_PATH="cluster_centroids/violence_clusters.pt"

# Output
OUTPUT_DIR="cluster_visualizations/violence"

# Model path (for denoising + VAE)
MODEL_PATH="CompVis/stable-diffusion-v1-4"

# Visualization options
SAMPLES_PER_CLUSTER=5
LATENT_TIMESTEP=500  # timestep at which latents were captured (ignored if multi-timestep dict)

# 실행
python visualize_clusters.py \
    --latents_path "$LATENTS_PATH" \
    --centroids_path "$CENTROIDS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --samples_per_cluster $SAMPLES_PER_CLUSTER \
    --latent_timestep $LATENT_TIMESTEP

echo "========================================"
echo "Visualization complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
