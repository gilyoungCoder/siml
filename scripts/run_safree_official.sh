#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
TASK=${2:?task}

export PYTHONNOUSERSITE=1
P=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
REPO=/mnt/home3/yhgil99/unlearning
SDIR=$REPO/unlearning-baselines/SAFREE_official
VLD=$REPO/vlm
OUTBASE=$REPO/unlearning-baselines/official_rerun/safree_official

case "$TASK" in
  nudity_rab)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.csv; CAT=nudity; EC=nudity ;;
  nudity_p4dn) PF=$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv; CAT=nudity; EC=nudity ;;
  nudity_ud)   PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv; CAT=nudity; EC=nudity ;;
  nudity_mma)  PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv; CAT=nudity; EC=nudity ;;
  nudity_i2p)  PF=$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv; CAT=nudity; EC=nudity ;;
  artist_vangogh) PF=$SDIR/datasets/big_artist_prompts.csv; CAT=artists-VanGogh; EC=artist_vangogh ;;
  artist_kelly)   PF=$SDIR/datasets/short_niche_art_prompts.csv; CAT=artists-KellyMcKernan; EC=artist_kelly ;;
  *) echo "unknown task: $TASK"; exit 2 ;;
esac

ODIR=$OUTBASE/$TASK
[ -f "$ODIR/results_qwen3_vl_${EC}.txt" ] && echo "[SKIP] SAFREE $TASK" && exit 0
mkdir -p "$ODIR"

if [ ! -f "$PF" ] && [[ "$TASK" == nudity_rab ]]; then
  python3 - <<'PY'
from pathlib import Path
src=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.txt')
dst=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.csv')
if not dst.exists():
    lines=[x.strip() for x in src.read_text().splitlines() if x.strip()]
    with dst.open('w') as f:
        f.write('prompt\n')
        for line in lines:
            f.write('"'+line.replace('"','""')+'"\n')
PY
fi

cd "$SDIR"
echo "[$(date +%H:%M)] GPU $GPU: SAFREE $TASK"
CUDA_VISIBLE_DEVICES=$GPU "$P" generate_safree.py \
  --config ./configs/sd_config.json \
  --data "$PF" \
  --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
  --num-samples 1 \
  --erase-id std \
  --model_id CompVis/stable-diffusion-v1-4 \
  --category "$CAT" \
  --save-dir "$ODIR" \
  --safree -svf -lra

if [[ "$CAT" == nudity ]]; then
  IMGDIR="$ODIR/all"; [ -d "$IMGDIR" ] || IMGDIR="$ODIR"
  N=$(find "$IMGDIR" -maxdepth 1 -name '*.png' | wc -l)
  if [ "$N" -ge 10 ]; then
    echo "[$(date +%H:%M)] Eval SAFREE $TASK ($N imgs)"
    CUDA_VISIBLE_DEVICES=$GPU "$VLP" "$VLD/opensource_vlm_i2p_all.py" "$IMGDIR" "$EC" qwen
  fi
fi

echo "[$(date +%H:%M)] DONE: SAFREE $TASK"
