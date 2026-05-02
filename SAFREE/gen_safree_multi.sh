#!/usr/bin/env bash
set -Eeuo pipefail

# ===== GPU 선택(필요 시) =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PY=gen_safree_multi.py         # <- 네가 쓰는 멀티 스크립트 이름
PROMPTS=../3_classification_sd1.4TC/prompts/VincentVanGogh.txt
OUTBASE=../3_classification_sd1.4TC/Continual/SAFREE

# 공통 하이퍼파라미터
STEPS=50
GUIDE=7.5
HEIGHT=512
WIDTH=512
SEED=42       # 재현성 필요 없으면 -1

# 로그 디렉토리 (타임스탬프별로 분리)
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTBASE/logs/$TS"
mkdir -p "$LOGDIR"

run_nohup () {
  local name="$1"
  shift
  local logfile="$LOGDIR/${name}.log"
  local pidfile="$LOGDIR/${name}.pid"

  echo "[RUN] $name"
  echo "[LOG] $logfile"
  nohup stdbuf -oL -eL "$@" >"$logfile" 2>&1 &

  local pid=$!
  echo "$pid" > "$pidfile"
  echo "[PID] $pid (saved to $pidfile)"
  echo
}

# # ---------- A) 단일 개념(누드만) ----------
# run_nohup "single_vangogh_Vangogh" \
# python "$PY" \
#   --txt "$PROMPTS" \
#   --model_id CompVis/stable-diffusion-v1-4 \
#   --outdir "$OUTBASE/single_nudity_Va_gogh" \
#   --num_images 1 --steps "$STEPS" --guidance "$GUIDE" \
#   --height "$HEIGHT" --width "$WIDTH" \
#   --seed "$SEED" \
#   --safree --lra --svf --sf_alpha 0.01 --re_attn_t=-1,4 --up_t 10 \
#   --categories van_gogh \
#   --use_default_negative

# ---------- B) 다중 개념(누드+폭력+Van Gogh 스타일) ----------
# 필요에 따라 categories 뒤에 원하는 개념을 쉼표로 추가하면 됨
run_nohup "multi_nudity_Vangogh" \
python "$PY" \
  --txt "$PROMPTS" \
  --model_id CompVis/stable-diffusion-v1-4 \
  --outdir "$OUTBASE/multi_nudity_Van_gogh" \
  --num_images 1 --steps "$STEPS" --guidance "$GUIDE" \
  --height "$HEIGHT" --width "$WIDTH" \
  --seed "$SEED" \
  --safree --lra --svf --sf_alpha 0.01 --re_attn_t=-1,4 --up_t 10 \
  --categories nudity,van_gogh \
  --use_default_negative

echo "==== JOBS STARTED ===="
echo "logs: $LOGDIR"
# echo "tail -f \"$LOGDIR/single_nudity_Vangogh.log\""
echo "tail -f \"$LOGDIR/multi_nudity_violence_Van_gogh.log\""
