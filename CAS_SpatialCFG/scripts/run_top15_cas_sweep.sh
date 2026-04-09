#!/bin/bash
# Top-15 SR configs × CAS threshold 0.5/0.6/0.7 sweep
# + NudeNet + VLM eval pipeline
# 8 GPUs, round-robin assignment

BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
OUTBASE="$BASE/outputs/top15_sweep"
PY4="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v4.py"
PY5="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v5.py"
PY3="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate.py"
NUDENET_PY="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python /mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_PY="/mnt/home/yhgil99/.conda/envs/vlm/bin/python /mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COMMON="--nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5 --cas_sticky"
LOG="$OUTBASE/run.log"
mkdir -p "$OUTBASE"

run_eval() {
    local gpu=$1; local outdir=$2
    CUDA_VISIBLE_DEVICES=$gpu $NUDENET_PY "$outdir" --threshold 0.6 --save_path results_nudenet_06.txt >> "$LOG" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu $VLM_PY "$outdir" nudity qwen >> "$LOG" 2>&1
}

# ─── GPU 0 ──────────────────────────────────────────────────────────────────
gpu0_worker() {
CUDA_VISIBLE_DEVICES=0 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s09_cas05 \
  --cas_threshold 0.5 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 0 $OUTBASE/v4_ainp_s09_cas05

CUDA_VISIBLE_DEVICES=0 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s10_pm01_cas05 \
  --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1 $COMMON >> "$LOG" 2>&1
run_eval 0 $OUTBASE/v5_ainp_s10_pm01_cas05

CUDA_VISIBLE_DEVICES=0 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s10_cas05 \
  --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 0 $OUTBASE/v5_ainp_s10_cas05

CUDA_VISIBLE_DEVICES=0 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t02_cas05 \
  --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.2 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 0 $OUTBASE/v4_ainp_s10_t02_cas05

CUDA_VISIBLE_DEVICES=0 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s09_cas05 \
  --cas_threshold 0.5 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 0 $OUTBASE/v5_ainp_s09_cas05

CUDA_VISIBLE_DEVICES=0 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t03_cas05 \
  --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 0 $OUTBASE/v4_ainp_s10_t03_cas05
}

# ─── GPU 1 ──────────────────────────────────────────────────────────────────
gpu1_worker() {
CUDA_VISIBLE_DEVICES=1 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s09_cas06 \
  --cas_threshold 0.6 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 1 $OUTBASE/v4_ainp_s09_cas06

CUDA_VISIBLE_DEVICES=1 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s10_pm01_cas06 \
  --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1 $COMMON >> "$LOG" 2>&1
run_eval 1 $OUTBASE/v5_ainp_s10_pm01_cas06

CUDA_VISIBLE_DEVICES=1 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s10_cas06 \
  --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 1 $OUTBASE/v5_ainp_s10_cas06

CUDA_VISIBLE_DEVICES=1 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t02_cas06 \
  --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.2 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 1 $OUTBASE/v4_ainp_s10_t02_cas06

CUDA_VISIBLE_DEVICES=1 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s09_cas06 \
  --cas_threshold 0.6 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 1 $OUTBASE/v5_ainp_s09_cas06

CUDA_VISIBLE_DEVICES=1 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t03_cas06 \
  --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 1 $OUTBASE/v4_ainp_s10_t03_cas06
}

# ─── GPU 2 ──────────────────────────────────────────────────────────────────
gpu2_worker() {
CUDA_VISIBLE_DEVICES=2 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s09_cas07 \
  --cas_threshold 0.7 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 2 $OUTBASE/v4_ainp_s09_cas07

CUDA_VISIBLE_DEVICES=2 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s10_pm01_cas07 \
  --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1 $COMMON >> "$LOG" 2>&1
run_eval 2 $OUTBASE/v5_ainp_s10_pm01_cas07

CUDA_VISIBLE_DEVICES=2 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s10_cas07 \
  --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 2 $OUTBASE/v5_ainp_s10_cas07

CUDA_VISIBLE_DEVICES=2 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t02_cas07 \
  --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.2 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 2 $OUTBASE/v4_ainp_s10_t02_cas07

CUDA_VISIBLE_DEVICES=2 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s09_cas07 \
  --cas_threshold 0.7 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 2 $OUTBASE/v5_ainp_s09_cas07
}

# ─── GPU 3 ──────────────────────────────────────────────────────────────────
gpu3_worker() {
CUDA_VISIBLE_DEVICES=3 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t03_cas07 \
  --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 3 $OUTBASE/v4_ainp_s10_t03_cas07

CUDA_VISIBLE_DEVICES=3 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t05_cas05 \
  --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.5 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 3 $OUTBASE/v4_ainp_s10_t05_cas05

CUDA_VISIBLE_DEVICES=3 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t05_cas06 \
  --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.5 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 3 $OUTBASE/v4_ainp_s10_t05_cas06

CUDA_VISIBLE_DEVICES=3 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s10_t05_cas07 \
  --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.5 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 3 $OUTBASE/v4_ainp_s10_t05_cas07

CUDA_VISIBLE_DEVICES=3 $PY3 --prompts $PROMPTS --outdir $OUTBASE/v3_dag_s3_cas05 \
  --cas_threshold 0.5 --safety_scale 3.0 --spatial_threshold 0.3 --guide_mode dag_adaptive $COMMON >> "$LOG" 2>&1
run_eval 3 $OUTBASE/v3_dag_s3_cas05
}

# ─── GPU 4 ──────────────────────────────────────────────────────────────────
gpu4_worker() {
CUDA_VISIBLE_DEVICES=4 $PY3 --prompts $PROMPTS --outdir $OUTBASE/v3_dag_s3_cas06 \
  --cas_threshold 0.6 --safety_scale 3.0 --spatial_threshold 0.3 --guide_mode dag_adaptive $COMMON >> "$LOG" 2>&1
run_eval 4 $OUTBASE/v3_dag_s3_cas06

CUDA_VISIBLE_DEVICES=4 $PY3 --prompts $PROMPTS --outdir $OUTBASE/v3_dag_s3_cas07 \
  --cas_threshold 0.7 --safety_scale 3.0 --spatial_threshold 0.3 --guide_mode dag_adaptive $COMMON >> "$LOG" 2>&1
run_eval 4 $OUTBASE/v3_dag_s3_cas07

CUDA_VISIBLE_DEVICES=4 $PY3 --prompts $PROMPTS --outdir $OUTBASE/v3_dag_s5_cas05 \
  --cas_threshold 0.5 --safety_scale 5.0 --spatial_threshold 0.3 --guide_mode dag_adaptive $COMMON >> "$LOG" 2>&1
run_eval 4 $OUTBASE/v3_dag_s5_cas05

CUDA_VISIBLE_DEVICES=4 $PY3 --prompts $PROMPTS --outdir $OUTBASE/v3_dag_s5_cas06 \
  --cas_threshold 0.6 --safety_scale 5.0 --spatial_threshold 0.3 --guide_mode dag_adaptive $COMMON >> "$LOG" 2>&1
run_eval 4 $OUTBASE/v3_dag_s5_cas06

CUDA_VISIBLE_DEVICES=4 $PY3 --prompts $PROMPTS --outdir $OUTBASE/v3_dag_s5_cas07 \
  --cas_threshold 0.7 --safety_scale 5.0 --spatial_threshold 0.3 --guide_mode dag_adaptive $COMMON >> "$LOG" 2>&1
run_eval 4 $OUTBASE/v3_dag_s5_cas07
}

# ─── GPU 5 ──────────────────────────────────────────────────────────────────
gpu5_worker() {
CUDA_VISIBLE_DEVICES=5 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s08_cas05 \
  --cas_threshold 0.5 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 5 $OUTBASE/v4_ainp_s08_cas05

CUDA_VISIBLE_DEVICES=5 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s08_cas06 \
  --cas_threshold 0.6 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 5 $OUTBASE/v4_ainp_s08_cas06

CUDA_VISIBLE_DEVICES=5 $PY4 --prompts $PROMPTS --outdir $OUTBASE/v4_ainp_s08_cas07 \
  --cas_threshold 0.7 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint $COMMON >> "$LOG" 2>&1
run_eval 5 $OUTBASE/v4_ainp_s08_cas07

CUDA_VISIBLE_DEVICES=5 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_pt_m02_cas05 \
  --cas_threshold 0.5 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.2 $COMMON >> "$LOG" 2>&1
run_eval 5 $OUTBASE/v5_ainp_pt_m02_cas05

CUDA_VISIBLE_DEVICES=5 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_pt_m02_cas06 \
  --cas_threshold 0.6 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.2 $COMMON >> "$LOG" 2>&1
run_eval 5 $OUTBASE/v5_ainp_pt_m02_cas06
}

# ─── GPU 6 ──────────────────────────────────────────────────────────────────
gpu6_worker() {
CUDA_VISIBLE_DEVICES=6 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_pt_m02_cas07 \
  --cas_threshold 0.7 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.2 $COMMON >> "$LOG" 2>&1
run_eval 6 $OUTBASE/v5_ainp_pt_m02_cas07

CUDA_VISIBLE_DEVICES=6 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_sld_s10_cas05 \
  --cas_threshold 0.5 --safety_scale 10.0 --spatial_threshold 0.3 --guide_mode sld $COMMON >> "$LOG" 2>&1
run_eval 6 $OUTBASE/v5_sld_s10_cas05

CUDA_VISIBLE_DEVICES=6 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_sld_s10_cas06 \
  --cas_threshold 0.6 --safety_scale 10.0 --spatial_threshold 0.3 --guide_mode sld $COMMON >> "$LOG" 2>&1
run_eval 6 $OUTBASE/v5_sld_s10_cas06

CUDA_VISIBLE_DEVICES=6 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_sld_s10_cas07 \
  --cas_threshold 0.7 --safety_scale 10.0 --spatial_threshold 0.3 --guide_mode sld $COMMON >> "$LOG" 2>&1
run_eval 6 $OUTBASE/v5_sld_s10_cas07

CUDA_VISIBLE_DEVICES=6 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s08_cas05 \
  --cas_threshold 0.5 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 6 $OUTBASE/v5_ainp_s08_cas05
}

# ─── GPU 7 ──────────────────────────────────────────────────────────────────
gpu7_worker() {
CUDA_VISIBLE_DEVICES=7 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s08_cas06 \
  --cas_threshold 0.6 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 7 $OUTBASE/v5_ainp_s08_cas06

CUDA_VISIBLE_DEVICES=7 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_s08_cas07 \
  --cas_threshold 0.7 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0 $COMMON >> "$LOG" 2>&1
run_eval 7 $OUTBASE/v5_ainp_s08_cas07

CUDA_VISIBLE_DEVICES=7 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_pt_m01_cas05 \
  --cas_threshold 0.5 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1 $COMMON >> "$LOG" 2>&1
run_eval 7 $OUTBASE/v5_ainp_pt_m01_cas05

CUDA_VISIBLE_DEVICES=7 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_pt_m01_cas06 \
  --cas_threshold 0.6 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1 $COMMON >> "$LOG" 2>&1
run_eval 7 $OUTBASE/v5_ainp_pt_m01_cas06

CUDA_VISIBLE_DEVICES=7 $PY5 --prompts $PROMPTS --outdir $OUTBASE/v5_ainp_pt_m01_cas07 \
  --cas_threshold 0.7 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1 $COMMON >> "$LOG" 2>&1
run_eval 7 $OUTBASE/v5_ainp_pt_m01_cas07
}

# ─── LAUNCH ALL ─────────────────────────────────────────────────────────────
echo "[$(date)] Starting top15 CAS sweep on 8 GPUs (42 jobs)" | tee -a "$LOG"
gpu0_worker &
gpu1_worker &
gpu2_worker &
gpu3_worker &
gpu4_worker &
gpu5_worker &
gpu6_worker &
gpu7_worker &
wait
echo "[$(date)] All generation + eval DONE" | tee -a "$LOG"
