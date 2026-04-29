#!/bin/bash
# Extended image-count saturation: K ∈ {8, 16, 32} using extended exemplar packs.
set -uo pipefail
GPU=${1:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
PACK_DIR=$BASE/exemplars_K_extended
OUTBASE=$BASE/outputs/phase_img_saturation
LOG=$BASE/logs/img_saturation_ext_g${GPU}.log
> $LOG

SEXUAL_ARGS=$BASE/paper_results/single/sexual/args.json

for K in 8 16 32; do
    OUTDIR=$OUTBASE/sexual_K${K}
    mkdir -p $OUTDIR
    EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 60 ]; then
        echo "[skip] K=$K ($EXISTING/60)" | tee -a $LOG
        continue
    fi
    echo "[$(date +%H:%M:%S)] [K=$K] gen start" | tee -a $LOG
    cd $REPO/SafeGen
    $PY -c "
import json, subprocess
a = json.load(open('$SEXUAL_ARGS'))
cmd = ['env', 'CUDA_VISIBLE_DEVICES=$GPU', '$PY', '-m', 'safegen.generate_family',
       '--prompts', a['prompts'], '--outdir', '$OUTDIR',
       '--family_guidance', '--family_config', '$PACK_DIR/clip_grouped_K$K.pt',
       '--probe_mode', a['probe_mode'], '--how_mode', a['how_mode'],
       '--cas_threshold', str(a['cas_threshold']),
       '--safety_scale', str(a['safety_scale']),
       '--attn_threshold', str(a['attn_threshold']),
       '--img_attn_threshold', str(a['img_attn_threshold']),
       '--n_img_tokens', '$K',
       '--steps', '50', '--seed', '42', '--cfg_scale', '7.5',
       '--target_concepts', *a['target_concepts'],
       '--anchor_concepts', *a['anchor_concepts']]
subprocess.run(cmd, check=False)
" >> $LOG 2>&1
    final=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] [K=$K] DONE imgs=$final" | tee -a $LOG
done
echo "[$(date)] ext saturation gen done" | tee -a $LOG
