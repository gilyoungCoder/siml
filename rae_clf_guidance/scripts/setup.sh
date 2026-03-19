#!/usr/bin/env bash
# Download pretrained models from HuggingFace for RAE classifier guidance
# Run this once before training/generation.

set -e

cd /mnt/home/yhgil99/unlearning/rae_clf_guidance

echo "=== Downloading pretrained models from HuggingFace ==="
echo "Repository: nyu-visionx/RAE-collections"

# Install huggingface_hub if needed
pip install -q huggingface_hub

# Download all three required files (correct paths without models/ prefix)
huggingface-cli download nyu-visionx/RAE-collections \
  DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt \
  decoders/dinov2/wReg_base/ViTXL_n08/model.pt \
  stats/dinov2/wReg_base/imagenet1k/stat.pt \
  --local-dir pretrained_models

echo ""
echo "=== Creating symlinks for easier access ==="

# Symlink targets are relative to the symlink's directory (pretrained_models/)
ln -sf DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt \
  pretrained_models/stage2_model.pt

ln -sf decoders/dinov2/wReg_base/ViTXL_n08/model.pt \
  pretrained_models/decoder_model.pt

ln -sf stats/dinov2/wReg_base/imagenet1k/stat.pt \
  pretrained_models/stat.pt

echo ""
echo "=== Setup complete ==="
echo "Files:"
echo "  DiTDH-XL:  pretrained_models/stage2_model.pt"
echo "  Decoder:   pretrained_models/decoder_model.pt"
echo "  Stats:     pretrained_models/stat.pt"
