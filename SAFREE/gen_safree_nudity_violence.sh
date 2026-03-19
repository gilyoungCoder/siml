#!/usr/bin/env bash
# ============================================================================
# SAFREE Multi-Concept Erasure - Nudity + Violence
# ============================================================================
set -Eeuo pipefail

# GPU 설정
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

PY=gen_safree_multi.py

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTBASE="${SCRIPT_DIR}/safree_outputs/nudity_violence"

# Prompt files (absolute paths)
NUDITY_PROMPTS="${SCRIPT_DIR}/../SoftDelete+CG/prompts/sexual_50.txt"
VIOLENCE_PROMPTS="${SCRIPT_DIR}/../SoftDelete+CG/prompts/violence_50.txt"
MIXED_PROMPTS="${SCRIPT_DIR}/../SoftDelete+CG/prompts/mixed_concepts.txt"

# 공통 하이퍼파라미터
STEPS=50
GUIDE=7.5
HEIGHT=512
WIDTH=512
SEED=123

# SAFREE 파라미터
SF_ALPHA=0.01
RE_ATTN_T="-1,4"
UP_T=10

# 로그 디렉토리
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

echo "============================================"
echo "🔍 SAFREE Multi-Concept Erasure"
echo "============================================"
echo "Nudity + Violence simultaneous removal"
echo ""

# ============================================================================
# Experiment 1: Nudity prompts with multi-concept erasure
# ============================================================================
if [ -f "$NUDITY_PROMPTS" ]; then
    echo "============================================"
    echo "Experiment 1: Nudity prompts"
    echo "============================================"
    echo "Prompts: $NUDITY_PROMPTS"
    echo "Output: $OUTBASE/nudity_prompts"
    echo ""

    run_nohup "nudity_prompts" \
    python "$PY" \
      --txt "$NUDITY_PROMPTS" \
      --model_id CompVis/stable-diffusion-v1-4 \
      --outdir "$OUTBASE/nudity_prompts" \
      --num_images 1 --steps "$STEPS" --guidance "$GUIDE" \
      --height "$HEIGHT" --width "$WIDTH" \
      --seed "$SEED" \
      --safree --lra --svf --sf_alpha "$SF_ALPHA" --re_attn_t="$RE_ATTN_T" --up_t "$UP_T" \
      --categories nudity,violence \
      --use_default_negative

    echo "✅ Experiment 1 started!"
    echo ""
fi

# ============================================================================
# Experiment 2: Violence prompts with multi-concept erasure
# ============================================================================
if [ -f "$VIOLENCE_PROMPTS" ]; then
    echo "============================================"
    echo "Experiment 2: Violence prompts"
    echo "============================================"
    echo "Prompts: $VIOLENCE_PROMPTS"
    echo "Output: $OUTBASE/violence_prompts"
    echo ""

    run_nohup "violence_prompts" \
    python "$PY" \
      --txt "$VIOLENCE_PROMPTS" \
      --model_id CompVis/stable-diffusion-v1-4 \
      --outdir "$OUTBASE/violence_prompts" \
      --num_images 1 --steps "$STEPS" --guidance "$GUIDE" \
      --height "$HEIGHT" --width "$WIDTH" \
      --seed "$SEED" \
      --safree --lra --svf --sf_alpha "$SF_ALPHA" --re_attn_t="$RE_ATTN_T" --up_t "$UP_T" \
      --categories nudity,violence \
      --use_default_negative

    echo "✅ Experiment 2 started!"
    echo ""
fi

# ============================================================================
# Experiment 3: Mixed prompts (nudity + violence in same prompt)
# ============================================================================
if [ -f "$MIXED_PROMPTS" ]; then
    echo "============================================"
    echo "Experiment 3: Mixed concept prompts"
    echo "============================================"
    echo "Prompts: $MIXED_PROMPTS"
    echo "Output: $OUTBASE/mixed_prompts"
    echo ""

    run_nohup "mixed_prompts" \
    python "$PY" \
      --txt "$MIXED_PROMPTS" \
      --model_id CompVis/stable-diffusion-v1-4 \
      --outdir "$OUTBASE/mixed_prompts" \
      --num_images 1 --steps "$STEPS" --guidance "$GUIDE" \
      --height "$HEIGHT" --width "$WIDTH" \
      --seed "$SEED" \
      --safree --lra --svf --sf_alpha "$SF_ALPHA" --re_attn_t="$RE_ATTN_T" --up_t "$UP_T" \
      --categories nudity,violence \
      --use_default_negative

    echo "✅ Experiment 3 started!"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "🎉 SAFREE Multi-Concept Jobs Started!"
echo "============================================"
echo ""
echo "📁 Output directories (copy-paste ready):"
echo ""
[ -f "$NUDITY_PROMPTS" ] && echo "Nudity prompts:"
[ -f "$NUDITY_PROMPTS" ] && echo "  ${OUTBASE}/nudity_prompts"
[ -f "$NUDITY_PROMPTS" ] && echo ""
[ -f "$VIOLENCE_PROMPTS" ] && echo "Violence prompts:"
[ -f "$VIOLENCE_PROMPTS" ] && echo "  ${OUTBASE}/violence_prompts"
[ -f "$VIOLENCE_PROMPTS" ] && echo ""
[ -f "$MIXED_PROMPTS" ] && echo "Mixed prompts:"
[ -f "$MIXED_PROMPTS" ] && echo "  ${OUTBASE}/mixed_prompts"
echo ""
echo "============================================"
echo "📊 Log files:"
echo ""
[ -f "$NUDITY_PROMPTS" ] && echo "  ${LOGDIR}/nudity_prompts.log"
[ -f "$VIOLENCE_PROMPTS" ] && echo "  ${LOGDIR}/violence_prompts.log"
[ -f "$MIXED_PROMPTS" ] && echo "  ${LOGDIR}/mixed_prompts.log"
echo ""
echo "To monitor progress:"
[ -f "$NUDITY_PROMPTS" ] && echo "  tail -f ${LOGDIR}/nudity_prompts.log"
[ -f "$VIOLENCE_PROMPTS" ] && echo "  tail -f ${LOGDIR}/violence_prompts.log"
[ -f "$MIXED_PROMPTS" ] && echo "  tail -f ${LOGDIR}/mixed_prompts.log"
echo ""
echo "To check running jobs:"
echo "  ps aux | grep gen_safree_multi.py"
echo ""
echo "Configuration:"
echo "  Categories: nudity, violence"
echo "  SAFREE enabled: Yes (LRA + SVF)"
echo "  sf_alpha: $SF_ALPHA"
echo "  re_attn_t: $RE_ATTN_T"
echo "  up_t: $UP_T"
echo "  Steps: $STEPS"
echo "  Guidance: $GUIDE"
echo "============================================"
