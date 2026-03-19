#!/bin/bash
# Wait for nudity training to complete, then run full eval pipeline
set -e

TRAIN_PID=$1
OUTPUT_DIR="/mnt/home/yhgil99/unlearning/SDErasure/outputs/sderasure_nudity"
UNET_DIR="$OUTPUT_DIR/unet"

echo "Waiting for training PID $TRAIN_PID to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
    # Check latest progress
    tail -1 "$OUTPUT_DIR/../sderasure_nudity_train.log" 2>/dev/null | tr -d '\r' | grep -o 'Training:[^|]*' | tail -1
done

echo ""
echo "Training complete! Checking outputs..."
ls -la "$UNET_DIR/" 2>/dev/null | head -5

if [ -f "$UNET_DIR/config.json" ]; then
    echo "UNet checkpoint found. Starting evaluation..."
    cd /mnt/home/yhgil99/unlearning/SDErasure
    bash eval_pipeline.sh "$UNET_DIR" sderasure_nudity 2 3
else
    echo "ERROR: UNet not found at $UNET_DIR"
    exit 1
fi
