#!/bin/bash
# Image-count saturation REDO with --probe_mode image_only.
# K ∈ {1, 2, 4, 8, 16, 32} per family for sexual concept.
set -uo pipefail
GPU=${1:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase_img_saturation_imageonly
LOG=$BASE/logs/img_saturation_imageonly_g${GPU}.log
mkdir -p $OUTBASE
> $LOG

SEXUAL_ARGS=$BASE/paper_results/single/sexual/args.json

# K=1,2,4 use exemplars_K_subsampled, K=8,16,32 use exemplars_K_extended
declare -A PACK_FOR_K=(
  [1]=$BASE/exemplars_K_subsampled/clip_grouped_K1.pt
  [2]=$BASE/exemplars_K_subsampled/clip_grouped_K2.pt
  [4]=$BASE/exemplars_K_subsampled/clip_grouped_K4.pt
  [8]=$BASE/exemplars_K_extended/clip_grouped_K8.pt
  [16]=$BASE/exemplars_K_extended/clip_grouped_K16.pt
  [32]=$BASE/exemplars_K_extended/clip_grouped_K32.pt
)

for K in 1 2 4 8 16 32; do
    PACK=${PACK_FOR_K[$K]}
    if [ ! -f "$PACK" ]; then echo "[err] missing pack K=$K: $PACK" | tee -a $LOG; continue; fi
    OUTDIR=$OUTBASE/sexual_K${K}
    mkdir -p $OUTDIR
    EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 60 ]; then
        echo "[skip] K=$K ($EXISTING/60)" | tee -a $LOG
        continue
    fi
    echo "[$(date +%H:%M:%S)] [K=$K] gen start (image_only) pack=$PACK" | tee -a $LOG
    cd $REPO/SafeGen
    $PY -c "
import json, subprocess
a = json.load(open('$SEXUAL_ARGS'))
cmd = ['env', 'CUDA_VISIBLE_DEVICES=$GPU', '$PY', '-m', 'safegen.generate_family',
       '--prompts', a['prompts'], '--outdir', '$OUTDIR',
       '--family_guidance', '--family_config', '$PACK',
       '--probe_mode', 'image',  # ← KEY CHANGE
       '--how_mode', a['how_mode'],
       '--cas_threshold', str(a['cas_threshold']),
       '--safety_scale', str(a['safety_scale']),
       '--attn_threshold', str(a['attn_threshold']),
       '--img_attn_threshold', str(a['img_attn_threshold']),
       '--n_img_tokens', '4',
       '--steps', '50', '--seed', '42', '--cfg_scale', '7.5',
       '--target_concepts', *a['target_concepts'],
       '--anchor_concepts', *a['anchor_concepts']]
subprocess.run(cmd, check=False)
" >> $LOG 2>&1
    final=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] [K=$K] DONE imgs=$final" | tee -a $LOG
done
echo "[$(date)] image_only saturation gen done" | tee -a $LOG
