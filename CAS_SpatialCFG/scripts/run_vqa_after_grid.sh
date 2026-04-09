#!/bin/bash
# VQA Alignment evaluation on v13 full grid search results
# Waits for grid search (PID 2916319) to finish, then runs VQA alignment on all ringabell79_* dirs

PYTHON_SDD="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VQA_SCRIPT="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore_alignment.py"
PROMPTS_RB="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/anchor_strict_ringabell.csv"
OUTBASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v13"
LOG="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/logs/vqa_after_grid.log"

echo "=========================================" | tee $LOG
echo "VQA Alignment after v13 Grid Search" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG
echo "=========================================" | tee -a $LOG

# Wait for grid search to finish
GRID_PID=2916319
echo "Waiting for grid search PID $GRID_PID to finish..." | tee -a $LOG
while kill -0 $GRID_PID 2>/dev/null; do
    sleep 300  # check every 5 min
done
echo "Grid search finished: $(date)" | tee -a $LOG

# Find all ringabell79_* output dirs
DIRS=$(find $OUTBASE -maxdepth 1 -type d -name "ringabell79_*" | sort)
TOTAL=$(echo "$DIRS" | wc -l)
echo "Found $TOTAL output dirs to evaluate" | tee -a $LOG

# Run VQA alignment across 8 GPUs
GPU_IDX=0
RUNNING=0
BATCH_SIZE=8
COUNT=0

for dir in $DIRS; do
    dirname=$(basename $dir)

    # Skip if already evaluated
    if [ -f "$dir/results_vqascore_alignment.json" ]; then
        echo "  SKIP $dirname (already done)" | tee -a $LOG
        COUNT=$((COUNT + 1))
        continue
    fi

    # Skip if no images
    PNG_COUNT=$(ls $dir/*.png 2>/dev/null | wc -l)
    if [ "$PNG_COUNT" -eq 0 ]; then
        echo "  SKIP $dirname (no images)" | tee -a $LOG
        COUNT=$((COUNT + 1))
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_IDX PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
        $dir --prompts $PROMPTS_RB --prompt_type all >> $LOG 2>&1 &

    COUNT=$((COUNT + 1))
    RUNNING=$((RUNNING + 1))
    GPU_IDX=$(( (GPU_IDX + 1) % 8 ))

    # Wait when batch is full
    if [ $RUNNING -ge $BATCH_SIZE ]; then
        wait
        RUNNING=0
        echo "  [$COUNT/$TOTAL] Batch done: $(date)" | tee -a $LOG
    fi
done

# Wait for remaining
wait
echo "" | tee -a $LOG
echo "All VQA evaluations done: $(date)" | tee -a $LOG

# Summary table
echo "" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "VQA Alignment Summary (anchor_strict)" | tee -a $LOG
echo "==========================================" | tee -a $LOG
printf "%-55s %10s %12s %10s\n" "Config" "VQA(orig)" "VQA(anchor)" "Gap(a-o)" | tee -a $LOG
echo "--------------------------------------------------------------------------------------------" | tee -a $LOG

for dir in $DIRS; do
    dirname=$(basename $dir)
    json="$dir/results_vqascore_alignment.json"
    if [ -f "$json" ]; then
        python3 -c "
import json
with open('$json') as f:
    d = json.load(f)
s = d.get('summary', {})
o = s.get('original', {}).get('mean', 0)
a = s.get('anchor', {}).get('mean', 0)
g = a - o
m = ' ***' if g > 0 else ''
print(f'$dirname'[:55].ljust(55) + f'{o:10.4f} {a:12.4f} {g:+10.4f}{m}')
" | tee -a $LOG
    fi
done

echo "" | tee -a $LOG
echo "Done: $(date)" | tee -a $LOG
