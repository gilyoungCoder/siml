#!/bin/bash
# ============================================================================
# Batch Opensource Evaluation (NudeNet + Q16)
# Evaluates all image folders across multiple GPUs
#
# Usage:
#   ./batch_eval_opensource.sh <dir1> [dir2] [dir3] ...
#   CUDA_VISIBLE_DEVICES=0,3,4,5 ./batch_eval_opensource.sh /path/to/folder1 /path/to/folder2
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/opensource_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Parse GPU list
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(0 1 2 3)
fi
NUM_GPUS=${#GPU_LIST[@]}

echo "============================================================"
echo "Batch Opensource Evaluation (NudeNet + Q16)"
echo "============================================================"
echo "GPUs: ${GPU_LIST[*]}"
echo "Log dir: ${LOG_DIR}"
echo ""

# Collect all directories to evaluate
declare -a ALL_DIRS=()

for base_dir in "$@"; do
    if [ ! -d "$base_dir" ]; then
        echo "Warning: Directory not found: $base_dir"
        continue
    fi

    # Check if base_dir itself contains images
    img_count=$(find "$base_dir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
    if [ "$img_count" -gt 0 ]; then
        ALL_DIRS+=("$base_dir")
    fi

    # Check subdirectories
    for subdir in "$base_dir"/*/; do
        if [ -d "$subdir" ]; then
            # Skip logs folder
            if [[ "$(basename "$subdir")" == "logs" ]]; then
                continue
            fi
            img_count=$(find "$subdir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
            if [ "$img_count" -gt 0 ]; then
                ALL_DIRS+=("$subdir")
            fi
        fi
    done
done

TOTAL_DIRS=${#ALL_DIRS[@]}
echo "Total directories to evaluate: ${TOTAL_DIRS}"
echo ""

if [ "$TOTAL_DIRS" -eq 0 ]; then
    echo "No directories with images found!"
    exit 1
fi

# Function to evaluate a directory
eval_dir() {
    local gpu_idx=$1
    local dir_path=$2
    local log_prefix=$3

    local actual_gpu=${GPU_LIST[$gpu_idx]}
    local dir_name=$(basename "$dir_path")
    local log_file="${LOG_DIR}/${log_prefix}_${dir_name}.log"

    echo "[GPU ${actual_gpu}] Evaluating: ${dir_path}"

    {
        echo "============================================================"
        echo "Directory: ${dir_path}"
        echo "GPU: ${actual_gpu}"
        echo "Time: $(date)"
        echo "============================================================"
        echo ""

        # Run NudeNet
        echo ">>> Running NudeNet..."
        CUDA_VISIBLE_DEVICES=${actual_gpu} python "${SCRIPT_DIR}/eval_nudenet.py" "$dir_path"

        echo ""

        # Run Q16
        echo ">>> Running Q16..."
        CUDA_VISIBLE_DEVICES=${actual_gpu} python "${SCRIPT_DIR}/eval_q16.py" "$dir_path"

        echo ""
        echo "Completed: $(date)"

    } > "$log_file" 2>&1

    echo "[GPU ${actual_gpu}] Done: ${dir_name}"
}

# Track running jobs per GPU
declare -A GPU_PIDS

# Process all directories
idx=0
for dir_path in "${ALL_DIRS[@]}"; do
    # Find available GPU
    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            pid="${GPU_PIDS[$gpu]:-}"

            # Check if slot is free
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                # Run evaluation
                eval_dir $gpu "$dir_path" "eval_${idx}" &
                GPU_PIDS[$gpu]=$!

                idx=$((idx + 1))
                echo "Progress: ${idx}/${TOTAL_DIRS}"

                sleep 1
                break 2
            fi
        done

        sleep 3
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs..."
wait

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo "Logs saved to: ${LOG_DIR}"
echo ""

# Aggregate results
echo "Aggregating results..."

python3 << 'EOF'
import os
import sys
from pathlib import Path

log_dir = os.environ.get('LOG_DIR', '.')
base_dirs = sys.argv[1:] if len(sys.argv) > 1 else []

# Find all result files
results = []

for arg in '''$@'''.split():
    base_path = Path(arg.strip())
    if not base_path.exists():
        continue

    # Check subdirectories
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir() or subdir.name == 'logs':
            continue

        nudenet_file = subdir / 'results_nudenet.txt'
        q16_file = subdir / 'results_q16.txt'

        nudenet_unsafe = None
        q16_unsafe = None

        if nudenet_file.exists():
            with open(nudenet_file) as f:
                for line in f:
                    if 'Unsafe rate' in line or 'unsafe' in line.lower():
                        try:
                            # Try to extract percentage
                            import re
                            match = re.search(r'(\d+\.?\d*)%', line)
                            if match:
                                nudenet_unsafe = float(match.group(1))
                        except:
                            pass

        if q16_file.exists():
            with open(q16_file) as f:
                for line in f:
                    if 'Inappropriate rate' in line or 'inappropriate' in line.lower():
                        try:
                            import re
                            match = re.search(r'(\d+\.?\d*)%', line)
                            if match:
                                q16_unsafe = float(match.group(1))
                        except:
                            pass

        if nudenet_unsafe is not None or q16_unsafe is not None:
            results.append({
                'dir': str(subdir),
                'name': subdir.name,
                'nudenet_unsafe': nudenet_unsafe,
                'q16_unsafe': q16_unsafe
            })

# Sort by NudeNet unsafe rate
results.sort(key=lambda x: x.get('nudenet_unsafe', 100) or 100)

print(f"\nFound {len(results)} evaluated directories\n")

if results:
    print("Top 10 safest (by NudeNet):")
    for i, r in enumerate(results[:10]):
        nn = f"{r['nudenet_unsafe']:.1f}%" if r['nudenet_unsafe'] is not None else "N/A"
        q16 = f"{r['q16_unsafe']:.1f}%" if r['q16_unsafe'] is not None else "N/A"
        print(f"  {i+1}. {r['name']}: NudeNet={nn}, Q16={q16}")
EOF

echo ""
echo "Done!"
