#!/bin/bash
# siml-08 GPU 4,5,6: SD3 Safe_Denoiser eval + SD3 SGF parallel
set -e
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
SGF_GEN="/mnt/home3/yhgil99/unlearning/scripts/sd3/generate_sd3_sgf.py"
SD3_OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
PROMPT_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
PR="/mnt/home3/yhgil99/unlearning/SafeGen/prompts"
LOG="/mnt/home3/yhgil99/unlearning/logs/siml08_wave2"
mkdir -p "$LOG"

echo "=== siml-08 wave2 START $(date) ==="

# --- GPU 4: SD3 Safe_Denoiser Qwen eval ---
(
export CUDA_VISIBLE_DEVICES=4
cd /mnt/home3/yhgil99/unlearning/vlm
SD_OUT="$SD3_OUT/safe_denoiser"

for ds in rab mma p4dn unlearndiff mja_sexual; do
  [ -d "$SD_OUT/$ds" ] && [ ! -f "$SD_OUT/$ds/results_qwen_nudity.txt" ] && {
    echo "[GPU4] safe_denoiser $ds → nudity"
    $VLM_PY "$EVAL" "$SD_OUT/$ds" nudity qwen > "$LOG/sd_eval_${ds}.log" 2>&1
  }
done

[ -d "$SD_OUT/mja_violent" ] && [ ! -f "$SD_OUT/mja_violent/results_qwen_violence.txt" ] && {
  echo "[GPU4] safe_denoiser mja_violent → violence"
  $VLM_PY "$EVAL" "$SD_OUT/mja_violent" violence qwen > "$LOG/sd_eval_mjav.log" 2>&1
}

[ -d "$SD_OUT/mja_disturbing" ] && [ ! -f "$SD_OUT/mja_disturbing/results_qwen_shocking.txt" ] && {
  echo "[GPU4] safe_denoiser mja_disturbing → shocking"
  $VLM_PY "$EVAL" "$SD_OUT/mja_disturbing" shocking qwen > "$LOG/sd_eval_mjad.log" 2>&1
}

[ -d "$SD_OUT/mja_illegal" ] && [ ! -f "$SD_OUT/mja_illegal/results_qwen_illegal.txt" ] && {
  echo "[GPU4] safe_denoiser mja_illegal → illegal"
  $VLM_PY "$EVAL" "$SD_OUT/mja_illegal" illegal qwen > "$LOG/sd_eval_mjail.log" 2>&1
}

echo "[GPU4] DONE $(date)"
) &
PID4=$!

# --- GPU 5: SD3 SGF generation (mma, p4dn, unlearndiff) ---
(
export CUDA_VISIBLE_DEVICES=5

echo "[GPU5] SGF mma"
$PY "$SGF_GEN" --prompts "$PROMPT_DIR/mma-diffusion-nsfw-adv-prompts.csv" \
  --outdir "$SD3_OUT/sgf/mma" --device cuda:0 > "$LOG/sgf_mma.log" 2>&1

echo "[GPU5] SGF p4dn"
$PY "$SGF_GEN" --prompts "$PROMPT_DIR/p4dn_16_prompt.csv" \
  --outdir "$SD3_OUT/sgf/p4dn" --device cuda:0 > "$LOG/sgf_p4dn.log" 2>&1

echo "[GPU5] SGF unlearndiff"
$PY "$SGF_GEN" --prompts "$PROMPT_DIR/unlearn_diff_nudity.csv" \
  --outdir "$SD3_OUT/sgf/unlearndiff" --device cuda:0 > "$LOG/sgf_udiff.log" 2>&1

# After gen, run Qwen eval for SGF
cd /mnt/home3/yhgil99/unlearning/vlm
for ds in mma p4dn unlearndiff; do
  echo "[GPU5] SGF Qwen $ds → nudity"
  $VLM_PY "$EVAL" "$SD3_OUT/sgf/$ds" nudity qwen > "$LOG/sgf_eval_${ds}.log" 2>&1
done

echo "[GPU5] DONE $(date)"
) &
PID5=$!

# --- GPU 6: SD3 SGF generation (mja 3종 + coco) ---
(
export CUDA_VISIBLE_DEVICES=6

echo "[GPU6] SGF mja_sexual"
$PY "$SGF_GEN" --prompts "$PR/mja_sexual.txt" \
  --outdir "$SD3_OUT/sgf/mja_sexual" --device cuda:0 > "$LOG/sgf_mjas.log" 2>&1

echo "[GPU6] SGF mja_violent"
$PY "$SGF_GEN" --prompts "$PR/mja_violent.txt" \
  --outdir "$SD3_OUT/sgf/mja_violent" --device cuda:0 > "$LOG/sgf_mjav.log" 2>&1

echo "[GPU6] SGF mja_disturbing"
$PY "$SGF_GEN" --prompts "$PR/mja_disturbing.txt" \
  --outdir "$SD3_OUT/sgf/mja_disturbing" --device cuda:0 > "$LOG/sgf_mjad.log" 2>&1

echo "[GPU6] SGF coco250"
$PY "$SGF_GEN" --prompts "$PROMPT_DIR/coco_30k_10k.csv" \
  --outdir "$SD3_OUT/sgf/coco250" --device cuda:0 --end 250 > "$LOG/sgf_coco.log" 2>&1

# After gen, run Qwen eval for SGF MJA
cd /mnt/home3/yhgil99/unlearning/vlm
echo "[GPU6] SGF Qwen mja_sexual → nudity"
$VLM_PY "$EVAL" "$SD3_OUT/sgf/mja_sexual" nudity qwen > "$LOG/sgf_eval_mjas.log" 2>&1
echo "[GPU6] SGF Qwen mja_violent → violence"
$VLM_PY "$EVAL" "$SD3_OUT/sgf/mja_violent" violence qwen > "$LOG/sgf_eval_mjav.log" 2>&1
echo "[GPU6] SGF Qwen mja_disturbing → shocking"
$VLM_PY "$EVAL" "$SD3_OUT/sgf/mja_disturbing" shocking qwen > "$LOG/sgf_eval_mjad.log" 2>&1

echo "[GPU6] DONE $(date)"
) &
PID6=$!

wait $PID4 $PID5 $PID6
echo "=== siml-08 wave2 ALL DONE $(date) ==="
