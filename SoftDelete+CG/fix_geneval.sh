#!/bin/bash

# Fix GenEval Installation
# =========================
# This script fixes the missing MMDetection config files issue.

set -e  # Exit on error

echo "======================================"
echo "Fixing GenEval Installation"
echo "======================================"
echo ""

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Step 1: Clone MMDetection (for config files)
echo "Step 1: Cloning MMDetection repository..."
echo "=========================================="
if [ -d "mmdetection" ]; then
    echo "MMDetection directory already exists, skipping clone..."
else
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    git checkout v3.3.0  # Stable version
    cd ..
fi

# Step 2: Install MMDetection in editable mode
echo ""
echo "Step 2: Installing MMDetection..."
echo "=========================================="
cd mmdetection
pip install -v -e .
cd ..

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying installation..."
echo "=========================================="
python -c "import mmdet; print('mmdet version:', mmdet.__version__)"
python -c "from mmdet.apis import init_detector; print('mmdet APIs loaded successfully')"

# Step 4: Check config files
echo ""
echo "Step 4: Checking config files..."
echo "=========================================="
MMDET_PATH=$(python -c "import mmdet; import os; print(os.path.dirname(mmdet.__file__))")
CONFIG_PATH="${MMDET_PATH}/../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"

if [ -f "$CONFIG_PATH" ]; then
    echo "✓ Config file found: $CONFIG_PATH"
else
    echo "✗ Config file not found: $CONFIG_PATH"
    echo ""
    echo "Trying alternative location..."

    # Link mmdetection configs to site-packages
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    ln -sf "$(pwd)/mmdetection/configs" "${SITE_PACKAGES}/configs"

    if [ -f "$CONFIG_PATH" ]; then
        echo "✓ Config file now accessible"
    else
        echo "✗ Still cannot find config file"
        echo "You may need to manually specify config path"
    fi
fi

# Step 5: Download models
echo ""
echo "Step 5: Downloading object detector models..."
echo "=========================================="
cd geneval
mkdir -p models
./evaluation/download_models.sh models/
cd ..

# Check if model downloaded
if [ -f "geneval/models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    echo "✓ Model downloaded successfully"
    MODEL_SIZE=$(du -h geneval/models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth | cut -f1)
    echo "  Size: $MODEL_SIZE"
else
    echo "✗ Model download failed"
fi

echo ""
echo "======================================"
echo "GenEval fix complete!"
echo "======================================"
echo ""
echo "Now try running: bash run_geneval.sh"
echo ""
