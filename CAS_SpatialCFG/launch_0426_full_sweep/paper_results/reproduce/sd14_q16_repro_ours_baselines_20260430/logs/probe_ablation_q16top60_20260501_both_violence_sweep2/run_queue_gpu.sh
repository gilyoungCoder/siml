#!/usr/bin/env bash
set -euo pipefail
GPU=$1
QUEUE=$2
HOSTTAG=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
CFGDIR=$ROOT/configs/probe_ablation_q16top60_20260501_both_violence_sweep2
OUTROOT=$ROOT/outputs/probe_ablation_q16top60_20260501_both_violence_sweep2
LOGDIR=$ROOT/logs/probe_ablation_q16top60_20260501_both_violence_sweep2
PYGEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
while IFS=, read -r name sh tau tht thi; do
  [ -n "$name" ] || continue
  CFG=$CFGDIR/${name}.json
  OUT=$OUTROOT/${name}
  mkdir -p "$OUT"
  cat > "$CFG" <<JSON
{
  "ckpt": "CompVis/stable-diffusion-v1-4",
  "prompts": "$CAND/prompts/i2p_q16_top60/violence_q16_top60.txt",
  "outdir": "$OUT",
  "nsamples": 1,
  "steps": 50,
  "seed": 42,
  "cfg_scale": 7.5,
  "start_idx": 0,
  "end_idx": -1,
  "cas_threshold": $tau,
  "probe_mode": "both",
  "family_config": "$CAND/exemplars/i2p_v1/violence/clip_grouped.pt",
  "attn_threshold": $tht,
  "img_attn_threshold": $thi,
  "attn_sigmoid_alpha": 10.0,
  "blur_sigma": 1.0,
  "probe_fusion": "union",
  "target_words": ["violence", "bloody", "scene", "weapon", "fight"],
  "n_img_tokens": 4,
  "how_mode": "hybrid",
  "safety_scale": $sh,
  "family_guidance": true,
  "target_concepts": ["violence", "bloody_scene", "weapon", "fight"],
  "anchor_concepts": ["peaceful_scene", "calm_interaction"],
  "attn_resolutions": [16, 32],
  "save_maps": false,
  "note": "both violence q16 top60 sweep2; host=$HOSTTAG gpu=$GPU"
}
JSON
  RES=$OUT/results_qwen3_vl_violence_v5.txt
  n=$(find "$OUT" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  echo "[$(date)] $HOSTTAG GPU=$GPU START $name sh=$sh tau=$tau txt=$tht img=$thi existing_n=$n" | tee -a "$LOGDIR/worker_${HOSTTAG}_gpu${GPU}.log"
  if [ "$n" -lt 60 ]; then
    rm -f "$RES"
    CUDA_VISIBLE_DEVICES=$GPU REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$GPU" --config "$CFG" > "$LOGDIR/${name}_${HOSTTAG}_gpu${GPU}_gen.log" 2>&1
  fi
  n=$(find "$OUT" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  echo "[$(date)] $HOSTTAG GPU=$GPU GEN_DONE $name n=$n" | tee -a "$LOGDIR/worker_${HOSTTAG}_gpu${GPU}.log"
  if [ ! -s "$RES" ]; then
    CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$OUT" violence qwen > "$LOGDIR/${name}_${HOSTTAG}_gpu${GPU}_eval.log" 2>&1
  fi
  cat "$RES" | tee -a "$LOGDIR/worker_${HOSTTAG}_gpu${GPU}.log"
  echo "[$(date)] $HOSTTAG GPU=$GPU DONE $name" | tee -a "$LOGDIR/worker_${HOSTTAG}_gpu${GPU}.log"
done < "$QUEUE"
