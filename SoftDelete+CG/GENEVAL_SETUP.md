# GenEval Setup and Usage Guide

This guide explains how to install and use GenEval to evaluate your generated images.

## What is GenEval?

**GenEval** is an object-focused framework for evaluating compositional text-to-image alignment. It measures how well generated images follow detailed prompt requirements across 6 tasks:

1. **Single Object** - Detection of individual objects
2. **Two Objects** - Multiple objects and their co-occurrence
3. **Counting** - Correct object quantities (e.g., "three apples")
4. **Colors** - Color accuracy (e.g., "red apple")
5. **Position** - Spatial placement (e.g., "cat on the left")
6. **Color Attribution** - Color-object binding (e.g., "red ball and blue box")

**Score**: Each task is scored 0-1, and the overall GenEval score is the average.

**GitHub**: https://github.com/djghosh13/geneval

---

## Installation

### Step 1: Clone GenEval Repository

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
git clone https://github.com/djghosh13/geneval.git ./geneval
cd geneval
```

### Step 2: Create Conda Environment

```bash
conda create -n geneval python=3.8
conda activate geneval
```

### Step 3: Install PyTorch

```bash
# For CUDA 12.1 (adjust for your CUDA version)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Core Dependencies

```bash
pip install networkx==2.8.8 open-clip-torch==2.26.1 clip-benchmark
pip install -U openmim einops lightning diffusers transformers tomli platformdirs
pip install --upgrade setuptools
```

### Step 5: Install MMEngine and MMCV

**For most GPUs (V100, A6000, A100):**
```bash
mim install mmengine
mim install mmcv-full==1.7.2
```

**For H100 GPUs (Hopper architecture):**
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout 1.x
MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
cd ..
```

### Step 6: Install MMDetection

**For most GPUs:**
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout 2.x
pip install -v -e .
cd ..
```

**For H100 GPUs:**
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout 2.x
MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
cd ..
```

### Step 7: Download Object Detector Models

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG/geneval
./evaluation/download_models.sh ./models
```

This will download the Mask2Former object detector (~500MB).

---

## Usage

### Quick Start

Once GenEval is installed, you can evaluate your images:

```bash
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
bash run_geneval.sh
```

### Customize Evaluation

Edit `run_geneval.sh` to change paths:

```bash
# Directory containing generated images
IMAGE_DIR="./unlearned_outputs/i2psexual_fk_steering"

# Prompt file used for generation
PROMPT_FILE="./prompts/sexual_50.txt"

# Image pattern to match
IMAGE_PATTERN="*_fk_best.png"
```

### Manual Usage

You can also run the Python script directly:

```bash
python evaluate_geneval.py \
    --image_dir ./unlearned_outputs/i2psexual_fk_steering \
    --prompt_file ./prompts/sexual_50.txt \
    --geneval_path ./geneval \
    --detector_path ./geneval/models \
    --output_dir ./geneval_results/fk_steering \
    --image_pattern "*_fk_best.png" \
    --output_file ./geneval_results/fk_steering/summary_scores.json
```

---

## Output

### Results Structure

```
geneval_results/fk_steering/
├── metadata.jsonl                # Image-prompt mapping
├── geneval_results.jsonl        # Detailed per-image results
└── summary_scores.json          # Final GenEval scores
```

### Summary Scores Format

```json
{
  "single_object": 0.95,
  "two_object": 0.78,
  "counting": 0.82,
  "colors": 0.88,
  "position": 0.65,
  "color_attribution": 0.45,
  "overall": 0.755
}
```

### Interpreting Scores

- **0.0 - 0.3**: Poor compositional alignment
- **0.3 - 0.6**: Moderate alignment
- **0.6 - 0.8**: Good alignment
- **0.8 - 1.0**: Excellent alignment

**Typical scores for SD-v1.5**:
- Single Object: ~0.95
- Two Objects: ~0.70
- Counting: ~0.75
- Colors: ~0.85
- Position: ~0.60
- Color Attribution: ~0.40

---

## Troubleshooting

### Issue 1: GenEval Repository Not Found

```
ERROR: GenEval repository not found at ./geneval
```

**Solution**: Clone the repository first:
```bash
git clone https://github.com/djghosh13/geneval.git ./geneval
```

### Issue 2: Detector Models Not Found

```
WARNING: Detector models not found at ./geneval/models
```

**Solution**: Download models:
```bash
cd geneval
./evaluation/download_models.sh ./models
```

### Issue 3: Import Error for mmdet

```
ModuleNotFoundError: No module named 'mmdet'
```

**Solution**: Install mmdetection:
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout 2.x
pip install -v -e .
```

### Issue 4: CUDA Architecture Mismatch (H100)

```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution**: Recompile mmcv and mmdet with H100 support:
```bash
# Recompile mmcv
cd mmcv
MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e . --force-reinstall

# Recompile mmdet
cd ../mmdetection
MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e . --force-reinstall
```

### Issue 5: No Images Found

```
ERROR: No images found! Check image_dir and image_pattern.
```

**Solution**: Check that:
1. `IMAGE_DIR` points to correct directory
2. `IMAGE_PATTERN` matches your image filenames
3. Images actually exist in the directory

### Issue 6: Python Version Incompatibility

```
ERROR: This package requires Python 3.8
```

**Solution**: Use Python 3.8:
```bash
conda create -n geneval python=3.8
conda activate geneval
```

---

## Evaluating Different Methods

### Baseline Images

```bash
# Edit run_geneval.sh
IMAGE_DIR="./outputs/baseline"
IMAGE_PATTERN="*.png"
OUTPUT_DIR="./geneval_results/baseline"
```

### FK Steering Images

```bash
IMAGE_DIR="./unlearned_outputs/i2psexual_fk_steering"
IMAGE_PATTERN="*_fk_best.png"
OUTPUT_DIR="./geneval_results/fk_steering"
```

### Best-of-N Images

```bash
IMAGE_DIR="./outputs/best_of_n"
IMAGE_PATTERN="*_best.png"
OUTPUT_DIR="./geneval_results/best_of_n"
```

---

## Comparing Methods

After evaluating multiple methods, you can compare their GenEval scores:

```python
import json

# Load scores
with open('./geneval_results/baseline/summary_scores.json') as f:
    baseline = json.load(f)

with open('./geneval_results/fk_steering/summary_scores.json') as f:
    fk_steering = json.load(f)

# Compare
print(f"Baseline Overall:     {baseline['overall']:.4f}")
print(f"FK Steering Overall:  {fk_steering['overall']:.4f}")
print(f"Improvement:          {fk_steering['overall'] - baseline['overall']:.4f}")
```

---

## References

- **GenEval Paper**: https://arxiv.org/abs/2310.11513
- **GenEval GitHub**: https://github.com/djghosh13/geneval
- **Installation Guide**: https://github.com/djghosh13/geneval/issues/12
- **NeurIPS 2023**: Ghosh et al., "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment"

---

## Integration with FK Steering

GenEval is perfect for evaluating whether FK steering maintains compositional quality while removing nudity:

**Expected Behavior**:
- FK steering should maintain or improve GenEval scores
- If scores drop significantly, it means compositional quality is degraded
- Target: GenEval score > 0.70 for good quality

**Metrics to Watch**:
- **Overall score**: Main quality indicator
- **Single/Two Object**: Object presence (should stay high)
- **Colors**: Color accuracy (should stay high)
- **Position**: Spatial relationships (may drop slightly if nudity removal changes poses)
