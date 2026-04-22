#!/bin/bash
# v2pack anchor_inpaint best cfg (ss1.2_imgthr0.3_both, from RAB v1pack best) on 4 nudity benchmarks.
# Generate + auto-eval (v5 nudity = v3 nudity, deterministic).
# Usage: bash v2pack_anchor_best_nudity.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYEVAL=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PYEVAL" ] || PYEVAL=$PYTHON
PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt
LOGDIR=$REPO/logs/launch_0420_nudity
mkdir -p $LOGDIR

declare -A PROMPTS EXPECTED
PROMPTS[rab]="$REPO/CAS_SpatialCFG/prompts/ringabell.txt";       EXPECTED[rab]=79
PROMPTS[unlearndiff]="$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt"; EXPECTED[unlearndiff]=141
PROMPTS[p4dn]="$REPO/CAS_SpatialCFG/prompts/p4dn.txt";          EXPECTED[p4dn]=150
PROMPTS[mma]="$REPO/CAS_SpatialCFG/prompts/mma.txt";            EXPECTED[mma]=999

DSETS=(rab unlearndiff p4dn mma)
N=${#DSETS[@]}
LOG="$LOGDIR/v2pack_anchor_best_g${GPU}_s${SLOT}.log"

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  DS=${DSETS[$i]}
  prompts="${PROMPTS[$DS]}"
  expected="${EXPECTED[$DS]}"
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity/ours_sd14_v2pack/${DS}/anchor_ss1.2_thr0.1_imgthr0.3_both"
  n=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n" -lt "$expected" ]; then
    mkdir -p "$outdir"
    echo "[$(date)] [g$GPU] [gen] $DS expected=$expected current=$n" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
      --prompts "$prompts" --outdir "$outdir" \
      --family_guidance --family_config "$PACK" \
      --probe_mode both --how_mode anchor_inpaint \
      --cas_threshold 0.6 --safety_scale 1.2 \
      --attn_threshold 0.1 --img_attn_threshold 0.3 \
      --target_concepts nudity nude_person naked_body \
      --anchor_concepts clothed_person fully_dressed_person \
      --start_idx "$n" \
      >> "$LOG" 2>&1 || { echo "[g$GPU] GEN FAILED $DS" | tee -a "$LOG"; continue; }
  else
    echo "[$(date)] [g$GPU] [gen-skip] $DS ($n/$expected)" | tee -a "$LOG"
  fi
  # Auto v5 nudity eval
  jf="$outdir/categories_qwen3_vl_nudity_v5.json"
  if [ ! -f "$jf" ]; then
    echo "[$(date)] [g$GPU] [eval-v5] $DS" | tee -a "$LOG"
    cd $REPO/vlm
    CUDA_VISIBLE_DEVICES=$GPU $PYEVAL opensource_vlm_i2p_all_v5.py "$outdir" nudity qwen >> "$LOG" 2>&1 || echo "[g$GPU] EVAL FAILED $DS" | tee -a "$LOG"
    cd $REPO/SafeGen
  fi
done
echo "[$(date)] [g$GPU s$SLOT] done"
