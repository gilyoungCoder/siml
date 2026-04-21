#!/bin/bash
# Best config (hybrid ss=20, img_thr=0.4, both) × v2pack × 4 nudity datasets
# Usage: bash best_v2pack_nudity.sh <gpu> <slot> <n_slots>
set -uo pipefail
GPU=$1; SLOT=$2; N_SLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt

declare -A FILES EXP
FILES[rab]=$REPO/CAS_SpatialCFG/prompts/ringabell.txt; EXP[rab]=79
FILES[unlearndiff]=$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt; EXP[unlearndiff]=141
FILES[p4dn]=$REPO/CAS_SpatialCFG/prompts/p4dn.txt; EXP[p4dn]=150
FILES[mma]=$REPO/CAS_SpatialCFG/prompts/mma.txt; EXP[mma]=999

DSS=(rab unlearndiff p4dn mma)
N=${#DSS[@]}
for ((i=SLOT; i<N; i+=N_SLOTS)); do
  ds=${DSS[$i]}
  OUT=$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity/ours_sd14_v2pack/$ds/hybrid_ss20_thr0.1_imgthr0.4_both
  mkdir -p $OUT
  N_IMGS=$(ls -1 $OUT/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge "${EXP[$ds]}" ]; then echo "[GPU $GPU][skip] $ds"; continue; fi
  echo "[GPU $GPU][run] $ds"
  cd $REPO/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
    --prompts ${FILES[$ds]} --outdir $OUT \
    --probe_mode both --cas_threshold 0.6 \
    --safety_scale 20 --attn_threshold 0.1 --img_attn_threshold 0.4 \
    --how_mode hybrid --family_guidance --family_config $PACK \
    >> $LOGDIR/best_v2pack_${ds}_g${GPU}.log 2>&1
done
