#!/bin/bash

# GenEval Installation Script
# ============================
#
# This script installs GenEval and its dependencies.
# Run this from the SoftDelete+CG directory.
#
# Usage:
#   bash install_geneval.sh

set -e  # Exit on error

echo "======================================"
echo "GenEval Installation"
echo "======================================"
echo ""

# Check if in correct directory
if [ ! -d "geneval" ]; then
    echo "ERROR: geneval directory not found!"
    echo "Please run this script from SoftDelete+CG directory."
    exit 1
fi

cd geneval

echo "Step 1: Installing PyTorch dependencies..."
echo "=========================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 2: Installing core dependencies..."
echo "=========================================="
pip install networkx==2.8.8 open-clip-torch clip-benchmark
pip install openmim einops diffusers transformers tomli platformdirs
pip install --upgrade setuptools

echo ""
echo "Step 3: Installing mmengine and mmcv..."
echo "=========================================="
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"

echo ""
echo "Step 4: Installing mmdetection..."
echo "=========================================="
pip install "mmdet>=3.0.0,<4.0.0"

echo ""
echo "Step 5: Downloading object detector models..."
echo "=========================================="
mkdir -p models
chmod +x evaluation/download_models.sh
./evaluation/download_models.sh models/

echo ""
echo "======================================"
echo "GenEval installation complete!"
echo "======================================"
echo ""
echo "You can now run: bash run_geneval.sh"
echo ""

cd ..
