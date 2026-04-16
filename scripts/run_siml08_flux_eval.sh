#!/bin/bash
# siml-08 GPU 4,5,6: Qwen eval for FLUX.2-klein (baseline+ours) + FLUX.1-dev baseline
set -e
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
KLEIN="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux2klein_experiments"
FDEV="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux1dev_experiments"
LOG="/mnt/home3/yhgil99/unlearning/logs/flux/siml08_eval"
mkdir -p "$LOG"

cd /mnt/home3/yhgil99/unlearning/vlm

echo "=== FLUX Qwen Eval START $(date) ==="

# --- GPU 4: FLUX.2-klein baseline eval ---
(
export CUDA_VISIBLE_DEVICES=4
echo "[GPU4] Klein baseline evals"

for ds in ringabell mma p4dn unlearndiff mja_sexual; do
  d="$KLEIN/baseline/$ds"
  [ -d "$d" ] && [ ! -f "$d/results_qwen_nudity.txt" ] && {
    echo "[GPU4] baseline $ds → nudity"
    $VLM_PY "$EVAL" "$d" nudity qwen > "$LOG/bl_${ds}.log" 2>&1
  }
done

d="$KLEIN/baseline/mja_violent"
[ -d "$d" ] && [ ! -f "$d/results_qwen_violence.txt" ] && {
  echo "[GPU4] baseline mja_violent → violence"
  $VLM_PY "$EVAL" "$d" violence qwen > "$LOG/bl_mjav.log" 2>&1
}

d="$KLEIN/baseline/mja_disturbing"
[ -d "$d" ] && [ ! -f "$d/results_qwen_shocking.txt" ] && {
  echo "[GPU4] baseline mja_disturbing → shocking"
  $VLM_PY "$EVAL" "$d" shocking qwen > "$LOG/bl_mjad.log" 2>&1
}

echo "[GPU4] Klein baseline DONE $(date)"
) &
PID4=$!

# --- GPU 5: FLUX.2-klein ours eval ---
(
export CUDA_VISIBLE_DEVICES=5
echo "[GPU5] Klein ours evals"

# Nudity configs
for cfg in rab_single_ainp_ss1.0_cas0.6 rab_single_ainp_ss1.5_cas0.6 \
           rab_single_hyb_ss1.0_cas0.6 rab_single_ainp_ss1.0_cas0.4 \
           rab_family_ainp_ss1.0_cas0.6 rab_family_ainp_ss1.5_cas0.6 \
           mma_family_ainp_ss1.0_cas0.6 p4dn_family_ainp_ss1.0_cas0.6 \
           udiff_family_ainp_ss1.0_cas0.6 mja_sexual_family_ainp_ss1.0_cas0.6; do
  d="$KLEIN/ours/$cfg"
  [ -d "$d" ] && [ ! -f "$d/results_qwen_nudity.txt" ] && {
    echo "[GPU5] ours $cfg → nudity"
    $VLM_PY "$EVAL" "$d" nudity qwen > "$LOG/ours_${cfg}.log" 2>&1
  }
done

# Violence config
d="$KLEIN/ours/mja_violent_family_ainp_ss1.0_cas0.4"
[ -d "$d" ] && [ ! -f "$d/results_qwen_violence.txt" ] && {
  echo "[GPU5] ours mja_violent → violence"
  $VLM_PY "$EVAL" "$d" violence qwen > "$LOG/ours_mjav.log" 2>&1
}

echo "[GPU5] Klein ours DONE $(date)"
) &
PID5=$!

# --- GPU 6: FLUX.1-dev baseline eval ---
(
export CUDA_VISIBLE_DEVICES=6
echo "[GPU6] FLUX.1-dev baseline evals"

for ds in ringabell mma p4dn unlearndiff mja_sexual; do
  d="$FDEV/baseline/$ds"
  [ -d "$d" ] && [ ! -f "$d/results_qwen_nudity.txt" ] && {
    echo "[GPU6] flux1dev baseline $ds → nudity"
    $VLM_PY "$EVAL" "$d" nudity qwen > "$LOG/f1bl_${ds}.log" 2>&1
  }
done

d="$FDEV/baseline/mja_violent"
[ -d "$d" ] && [ ! -f "$d/results_qwen_violence.txt" ] && {
  echo "[GPU6] flux1dev baseline mja_violent → violence"
  $VLM_PY "$EVAL" "$d" violence qwen > "$LOG/f1bl_mjav.log" 2>&1
}

d="$FDEV/baseline/mja_disturbing"
[ -d "$d" ] && [ ! -f "$d/results_qwen_shocking.txt" ] && {
  echo "[GPU6] flux1dev baseline mja_disturbing → shocking"
  $VLM_PY "$EVAL" "$d" shocking qwen > "$LOG/f1bl_mjad.log" 2>&1
}

echo "[GPU6] FLUX.1-dev baseline DONE $(date)"
) &
PID6=$!

wait $PID4 $PID5 $PID6
echo "=== FLUX Qwen Eval ALL DONE $(date) ==="
