#!/bin/bash
# VQA Alignment: 4 methods × 2 datasets
# Waits for grid search to finish, then:
# 1. Generate missing images (SAFREE country, v13 country)
# 2. Run VQA alignment on all 4 methods × 2 datasets

PYTHON_SDD="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VQA_SCRIPT="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore_alignment.py"
PROMPTS_RB="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/anchor_strict_ringabell.csv"
PROMPTS_CNB="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/country_nude_body.csv"

CAS_OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs"
SAFREE_DIR="/mnt/home3/yhgil99/unlearning/SAFREE"
SAFREE_RB="$SAFREE_DIR/results/grid_search_safree_dual_ringabell_20260129_065829/gs10.0_hs1.0_bs0.0_sp0.3-0.3"
SAFREE_CNB="$SAFREE_DIR/results/safree_country_nude_body"

LOG="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/logs/vqa_4methods.log"
mkdir -p "$(dirname $LOG)"

echo "=========================================" | tee $LOG
echo "VQA 4-Method Comparison" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG
echo "=========================================" | tee -a $LOG

# Wait for grid search to finish
GRID_PID=3203411
if kill -0 $GRID_PID 2>/dev/null; then
    echo "Waiting for grid search PID $GRID_PID..." | tee -a $LOG
    while kill -0 $GRID_PID 2>/dev/null; do
        sleep 300
    done
    echo "Grid search done: $(date)" | tee -a $LOG
fi

# ============================================================
# Phase 1: Generate missing images
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 1: Generate missing images ===" | tee -a $LOG

# 1a. SAFREE country_nude_body (20 prompts × 4 samples)
if [ ! -d "$SAFREE_CNB" ] || [ "$(ls $SAFREE_CNB/*.png 2>/dev/null | wc -l)" -lt 20 ]; then
    echo "Generating SAFREE country_nude_body..." | tee -a $LOG
    cd $SAFREE_DIR
    CUDA_VISIBLE_DEVICES=0 python gen_safree_single.py \
        --txt /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/country_nude_body.txt \
        --save-dir "$SAFREE_CNB" \
        --model_id "CompVis/stable-diffusion-v1-4" \
        --category nudity \
        --num-samples 4 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        --seed 42 \
        --image_length 512 \
        --device cuda:0 \
        --erase-id std \
        --sf_alpha 0.01 \
        --re_attn_t "-1,1001" \
        --up_t 10 \
        --freeu_hyp "1.0-1.0-0.9-0.2" \
        --safree -svf -lra >> $LOG 2>&1
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
    echo "  SAFREE country done: $(ls $SAFREE_CNB/*.png 2>/dev/null | wc -l) images" | tee -a $LOG
else
    echo "  SAFREE country already exists: $(ls $SAFREE_CNB/*.png 2>/dev/null | wc -l) images" | tee -a $LOG
fi

# 1b. v13 best config country_nude_body
# Find best v13 config from NudeNet results
V13_BEST=""
BEST_NN=100
for dir in $CAS_OUT/v13/ringabell79_*/; do
    nn_file="$dir/results_nudenet.txt"
    if [ -f "$nn_file" ]; then
        nn=$(grep -oP '[\d.]+(?=%)' "$nn_file" | head -1)
        if [ -n "$nn" ] && python3 -c "exit(0 if float('$nn') < float('$BEST_NN') else 1)" 2>/dev/null; then
            BEST_NN=$nn
            V13_BEST=$(basename $dir | sed 's/ringabell79_//')
        fi
    fi
done
echo "  v13 best config: $V13_BEST (NudeNet: ${BEST_NN}%)" | tee -a $LOG

V13_CNB_DIR="$CAS_OUT/v13/country_${V13_BEST}"
if [ -n "$V13_BEST" ] && ([ ! -d "$V13_CNB_DIR" ] || [ "$(ls $V13_CNB_DIR/*.png 2>/dev/null | wc -l)" -lt 20 ]); then
    echo "Generating v13 country_nude_body with $V13_BEST..." | tee -a $LOG
    # Extract params from dirname
    # e.g., clip_hybproj_ss10_st03 -> probe=clip_exemplar, mode=hybrid_proj, ss=1.0, st=0.3
    PARAMS=$(python3 -c "
name = '$V13_BEST'
parts = name.split('_')
# Find exemplar type
if 'clip' in parts[0]:
    probe = 'clip_exemplar'
    embed = 'exemplars/sd14/clip_exemplar_full_nudity.pt'
elif 'fn' in parts[0]:
    probe = 'clip_exemplar'
    embed = 'exemplars/sd14/fn_exemplar_full_nudity.pt'
else:
    probe = 'clip_exemplar'
    embed = 'exemplars/sd14/clip_exemplar_full_nudity.pt'

# Find mode
mode = 'hybrid_proj'
for p in parts:
    if 'hyb' == p: mode = 'hybrid'
    elif 'hybproj' == p: mode = 'hybrid_proj'
    elif 'proj' == p: mode = 'projection'
    elif 'sld' == p: mode = 'sld'

# Extract ss, st, a
ss = st = sa = ''
for p in parts:
    if p.startswith('ss'): ss = p[2:]
    elif p.startswith('st'): st = p[2:]
    elif p.startswith('a') and p[1:].isdigit(): sa = p[1:]

# Convert ss format: ss05 -> 0.5, ss10 -> 1.0, ss15 -> 1.5
if ss:
    if len(ss) == 2:
        ss = str(int(ss[:1])) + '.' + ss[1:]
    elif len(ss) == 3:
        ss = str(int(ss[:2])) + '.' + ss[2:]
if st:
    st = '0.' + st

print(f'--probe_source {probe} --clip_embeddings {embed} --guide_mode {mode} --safety_scale {ss} --spatial_threshold {st}' + (f' --sigmoid_alpha {sa}' if sa else ''))
")
    echo "  Params: $PARAMS" | tee -a $LOG

    CUDA_VISIBLE_DEVICES=1 $PYTHON_SDD generate_v13.py \
        --prompts prompts/country_nude_body.csv \
        --outdir "$V13_CNB_DIR" \
        $PARAMS \
        --cas_threshold 0.6 \
        --nsamples 4 --steps 50 --seed 42 >> $LOG 2>&1
    echo "  v13 country done: $(ls $V13_CNB_DIR/*.png 2>/dev/null | wc -l) images" | tee -a $LOG
else
    echo "  v13 country already exists or no best config found" | tee -a $LOG
fi

echo "Phase 1 done: $(date)" | tee -a $LOG

# ============================================================
# Phase 2: VQA Alignment evaluation
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 2: VQA Alignment ===" | tee -a $LOG

# Method dirs
SD_BASELINE="$CAS_OUT/v3/baseline"
V4_DIR="$CAS_OUT/v4/sld_s10"
V13_RB_DIR="$CAS_OUT/v13/ringabell79_${V13_BEST}"

echo "" | tee -a $LOG
echo "--- Dataset 1: anchor_strict (ringabell) ---" | tee -a $LOG

# SD baseline - anchor_strict (already done but redo with latest CSV)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $SD_BASELINE --prompts $PROMPTS_RB --prompt_type all >> $LOG 2>&1 &

# v4
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $V4_DIR --prompts $PROMPTS_RB --prompt_type all >> $LOG 2>&1 &

# SAFREE
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $SAFREE_RB --prompts $PROMPTS_RB --prompt_type all >> $LOG 2>&1 &

# v13 best
if [ -d "$V13_RB_DIR" ]; then
    CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
        $V13_RB_DIR --prompts $PROMPTS_RB --prompt_type all >> $LOG 2>&1 &
fi

wait
echo "anchor_strict done: $(date)" | tee -a $LOG

echo "" | tee -a $LOG
echo "--- Dataset 2: country_nude_body ---" | tee -a $LOG

# SD baseline
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $CAS_OUT/country/baseline --prompts $PROMPTS_CNB --prompt_type all >> $LOG 2>&1 &

# v4
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $CAS_OUT/country/v4_sld_s10 --prompts $PROMPTS_CNB --prompt_type all >> $LOG 2>&1 &

# SAFREE country
if [ -d "$SAFREE_CNB" ] && [ "$(ls $SAFREE_CNB/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
        $SAFREE_CNB --prompts $PROMPTS_CNB --prompt_type all >> $LOG 2>&1 &
fi

# v13 country
if [ -d "$V13_CNB_DIR" ] && [ "$(ls $V13_CNB_DIR/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
        $V13_CNB_DIR --prompts $PROMPTS_CNB --prompt_type all >> $LOG 2>&1 &
fi

wait
echo "country_nude_body done: $(date)" | tee -a $LOG

# ============================================================
# Phase 3: Summary
# ============================================================
echo "" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "  VQA Alignment 4-Method Comparison" | tee -a $LOG
echo "==========================================" | tee -a $LOG

printf "\n%-25s %10s %12s %10s\n" "Method" "VQA(orig)" "VQA(anchor)" "Gap(a-o)" | tee -a $LOG
echo "-----------------------------------------------------------" | tee -a $LOG

echo "" | tee -a $LOG
echo "[ anchor_strict (ringabell) ]" | tee -a $LOG
for label_dir in "SD_baseline:$SD_BASELINE" "v4_sld_s10:$V4_DIR" "SAFREE:$SAFREE_RB" "v13_best:$V13_RB_DIR"; do
    label=$(echo $label_dir | cut -d: -f1)
    dir=$(echo $label_dir | cut -d: -f2)
    json="$dir/results_vqascore_alignment.json"
    if [ -f "$json" ]; then
        python3 -c "
import json
with open('$json') as f:
    d = json.load(f)
s = d.get('summary', {})
o = s.get('original', {}).get('mean', 0)
a = s.get('anchor', {}).get('mean', 0)
g = a - o
m = ' ***' if g > 0 else ''
print(f'  {\"$label\":<23s} {o:10.4f} {a:12.4f} {g:+10.4f}{m}')
" | tee -a $LOG
    else
        printf "  %-23s %10s\n" "$label" "N/A" | tee -a $LOG
    fi
done

echo "" | tee -a $LOG
echo "[ country_nude_body ]" | tee -a $LOG
for label_dir in "SD_baseline:$CAS_OUT/country/baseline" "v4_sld_s10:$CAS_OUT/country/v4_sld_s10" "SAFREE:$SAFREE_CNB" "v13_best:$V13_CNB_DIR"; do
    label=$(echo $label_dir | cut -d: -f1)
    dir=$(echo $label_dir | cut -d: -f2)
    json="$dir/results_vqascore_alignment.json"
    if [ -f "$json" ]; then
        python3 -c "
import json
with open('$json') as f:
    d = json.load(f)
s = d.get('summary', {})
o = s.get('original', {}).get('mean', 0)
a = s.get('anchor', {}).get('mean', 0)
g = a - o
m = ' ***' if g > 0 else ''
print(f'  {\"$label\":<23s} {o:10.4f} {a:12.4f} {g:+10.4f}{m}')
" | tee -a $LOG
    else
        printf "  %-23s %10s\n" "$label" "N/A" | tee -a $LOG
    fi
done

echo "" | tee -a $LOG
echo "All done: $(date)" | tee -a $LOG
