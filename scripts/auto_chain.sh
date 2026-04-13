#!/bin/bash
# Auto-chain: wait for fix to complete, then run master pipeline
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"

echo "[$(date)] Waiting for phase0_fix_clip to complete..."

# Wait for all .pt files to appear
while true; do
    all_ready=true
    for concept in sexual violent disturbing illegal harassment hate selfharm; do
        pt="CAS_SpatialCFG/exemplars/concepts_v2/${concept}/clip_grouped.pt"
        if [ ! -f "$pt" ]; then
            all_ready=false
            break
        fi
    done
    if $all_ready; then
        echo "[$(date)] All .pt files ready!"
        break
    fi
    sleep 30
done

# Small delay to ensure files are fully written
sleep 5

echo "[$(date)] Starting Phase 1: Ours generation..."
conda activate sdd_copy
bash scripts/phase1_ours_generation.sh 2>&1 | tee scripts/phase1.log

echo "[$(date)] Starting Phase 2: Evaluation..."
bash scripts/phase2_evaluate_all.sh 2>&1 | tee scripts/phase2.log

echo "[$(date)] ALL DONE!"
