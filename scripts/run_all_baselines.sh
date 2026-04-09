#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# COMPREHENSIVE BASELINE EXPERIMENTS — All Methods × All Concepts
# =============================================================================
# Baselines: SD v1.4 Baseline, RECE, SDErasure, SLD-Max, SAFREE
# Ours: v27 anchor_inpaint / hybrid × img / txt / img+txt
# Concepts: nudity, violence, harassment, hate, shocking, illegal_activity, self-harm
# Datasets: I2P concept-specific CSVs
# =============================================================================

SERVER=${1:-siml-02}
GPU=${2:-0}

# === Paths ===
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
I2P=$REPO/SAFREE/datasets/i2p_categories
OUTBASE=$REPO/unlearning-baselines/outputs
V27=$REPO/CAS_SpatialCFG/generate_v27.py
CLIP=$REPO/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
mkdir -p $OUTBASE

# === I2P Concept Datasets ===
declare -A PROMPTS=(
  [nudity_ringabell]="$REPO/CAS_SpatialCFG/prompts/ringabell.txt"
  [nudity_unlearndiff]="$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv"
  [nudity_mma]="$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv"
  [nudity_i2p]="$I2P/i2p_sexual.csv"
  [violence]="$I2P/i2p_violence.csv"
  [harassment]="$I2P/i2p_harassment.csv"
  [hate]="$I2P/i2p_hate.csv"
  [shocking]="$I2P/i2p_shocking.csv"
  [illegal_activity]="$I2P/i2p_illegal_activity.csv"
  [self_harm]="$I2P/i2p_self-harm.csv"
)

# === Qwen eval concept mapping ===
declare -A EVAL_CONCEPT=(
  [nudity_ringabell]="nudity"
  [nudity_unlearndiff]="nudity"
  [nudity_mma]="nudity"
  [nudity_i2p]="nudity"
  [violence]="violence"
  [harassment]="harassment"
  [hate]="hate"
  [shocking]="shocking"
  [illegal_activity]="illegal"
  [self_harm]="self_harm"
)

GEN_COMMON="--nsamples 1 --steps 50 --seed 42 --cfg_scale 7.5"

# === Helper Functions ===
gen_done() { [ -f "${1}/generation_stats.json" ] || [ $(find "$1" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) -ge 50 ]; }

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  [ -f "${dir}/categories_qwen3_vl_${concept}.json" ] && return
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -lt 10 ] && return
  echo "[$(date +%H:%M)] GPU $gpu: Eval $(basename $dir) / $concept ($n imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -1
}

# =============================================================================
# METHOD 1: SD v1.4 Baseline (no guidance, no safety)
# =============================================================================
gen_baseline() {
  local gpu=$1 dataset=$2 prompt_file=$3
  local outdir=$OUTBASE/baseline/$dataset
  gen_done "$outdir" && return
  mkdir -p "$outdir"
  echo "[$(date +%H:%M)] GPU $gpu: Baseline $dataset"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 \
    --prompts "$prompt_file" --outdir "$outdir" \
    --cas_threshold 99.0 --safety_scale 0.0 \
    --how_mode anchor_inpaint --probe_mode text --attn_threshold 0.1 \
    $GEN_COMMON 2>&1 | tail -1
}

# =============================================================================
# METHOD 2: SLD-Max (Training-free, via RECE repo)
# =============================================================================
gen_sld_max() {
  local gpu=$1 dataset=$2 prompt_file=$3
  local outdir=$OUTBASE/sld_max/$dataset
  gen_done "$outdir" && return
  mkdir -p "$outdir"
  echo "[$(date +%H:%M)] GPU $gpu: SLD-Max $dataset"

  # SLD-Max uses diffusers StableDiffusionPipelineSafe
  CUDA_VISIBLE_DEVICES=$gpu $P - << 'SLDEOF'
import sys, os, torch
sys.path.insert(0, "/mnt/home3/yhgil99/unlearning/unlearning-baselines/RECE")
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
import csv, json

prompt_file = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROMPT_FILE")
outdir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUTDIR")

# Load prompts
prompts = []
if prompt_file.endswith('.csv'):
    import pandas as pd
    df = pd.read_csv(prompt_file)
    col = 'prompt' if 'prompt' in df.columns else df.columns[0]
    prompts = df[col].tolist()
else:
    with open(prompt_file) as f:
        prompts = [l.strip() for l in f if l.strip()]

# Load pipeline with SLD
from diffusers import StableDiffusionPipelineSafe
pipe = StableDiffusionPipelineSafe.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

sld_config = SafetyConfig.MAX
gen = torch.Generator("cuda").manual_seed(42)

os.makedirs(outdir, exist_ok=True)
for i, prompt in enumerate(prompts):
    outpath = os.path.join(outdir, f"{i:04d}_00.png")
    if os.path.exists(outpath):
        continue
    try:
        img = pipe(
            prompt, num_inference_steps=50, guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42),
            **sld_config,
        ).images[0]
        img.save(outpath)
    except Exception as e:
        print(f"  Error {i}: {e}")
    if (i+1) % 50 == 0:
        print(f"  SLD-Max: {i+1}/{len(prompts)}")

print(f"Done! {len(os.listdir(outdir))} images")
with open(os.path.join(outdir, "generation_stats.json"), "w") as f:
    json.dump({"method": "SLD-Max", "total": len(prompts)}, f)
SLDEOF
  "$prompt_file" "$outdir" 2>&1 | tail -3
}

# =============================================================================
# METHOD 3: SAFREE (Training-free)
# =============================================================================
gen_safree() {
  local gpu=$1 dataset=$2 prompt_file=$3 concept=$4
  local outdir=$OUTBASE/safree/$dataset
  gen_done "$outdir" && return
  mkdir -p "$outdir"
  echo "[$(date +%H:%M)] GPU $gpu: SAFREE $dataset"

  cd $REPO/SAFREE
  CUDA_VISIBLE_DEVICES=$gpu $P generate_safree.py \
    --data "$prompt_file" \
    --config configs/sd_config.json \
    --save-dir "$outdir" \
    --model_id CompVis/stable-diffusion-v1-4 \
    --category "$concept" \
    --num-samples 1 \
    --device cuda:0 \
    --safree -svf -lra \
    --sf_alpha 0.01 \
    --re_attn_t "-1,4" \
    --up_t 10 2>&1 | tail -3
  cd $REPO
}

# =============================================================================
# MASTER RUN: Generate + Evaluate per GPU
# =============================================================================
run_gpu() {
  local gpu=$1
  shift
  local datasets=("$@")

  for ds in "${datasets[@]}"; do
    local pf="${PROMPTS[$ds]}"
    local ec="${EVAL_CONCEPT[$ds]}"

    # 1. Baseline
    gen_baseline $gpu "$ds" "$pf"
    eval_qwen $gpu "$OUTBASE/baseline/$ds" "$ec"

    # 2. SLD-Max
    gen_sld_max $gpu "$ds" "$pf"
    eval_qwen $gpu "$OUTBASE/sld_max/$ds" "$ec"

    # 3. SAFREE (only for nudity and supported concepts)
    if [[ "$ds" == nudity_* ]] || [[ "$ec" == "nudity" ]]; then
      gen_safree $gpu "$ds" "$pf" "nudity"
      eval_qwen $gpu "$OUTBASE/safree/$ds" "$ec"
    fi
  done

  echo "GPU $gpu ALL DONE — $(date)"
}

# =============================================================================
# GPU Assignment (6 GPUs: 0,1,2,3,6,7)
# =============================================================================
echo "=============================================="
echo "  BASELINE EXPERIMENTS — $SERVER — $(date)"
echo "=============================================="

# GPU 0: nudity datasets (ringabell, unlearndiff)
(run_gpu 0 nudity_ringabell nudity_unlearndiff) &

# GPU 1: nudity datasets (mma, i2p)
(run_gpu 1 nudity_mma nudity_i2p) &

# GPU 2: violence + harassment
(run_gpu 2 violence harassment) &

# GPU 3: hate + shocking
(run_gpu 3 hate shocking) &

# GPU 6: illegal_activity + self_harm
(run_gpu 6 illegal_activity self_harm) &

# GPU 7: RECE + SDErasure for nudity (pre-trained checkpoints)
(
echo "=== GPU 7: RECE + SDErasure nudity ==="

# RECE nudity (download checkpoint first if needed)
RECE_DIR=$REPO/unlearning-baselines/RECE
RECE_CKPT="$RECE_DIR/ckpts/nudity"

# Check if RECE checkpoint exists, if not download
if [ ! -d "$RECE_CKPT" ]; then
  echo "Downloading RECE nudity checkpoint from HuggingFace..."
  mkdir -p "$RECE_DIR/ckpts"
  cd "$RECE_DIR/ckpts"
  # Try to download from HF
  $P -c "
from huggingface_hub import snapshot_download
snapshot_download('ChaoGong/RECE', local_dir='.', allow_patterns=['nudity/*'])
" 2>&1 | tail -3
  cd $REPO
fi

# Generate RECE images for nudity datasets
for ds in nudity_ringabell nudity_unlearndiff nudity_mma; do
  pf="${PROMPTS[$ds]}"
  outdir=$OUTBASE/rece/$ds
  if ! gen_done "$outdir"; then
    mkdir -p "$outdir"
    echo "[$(date +%H:%M)] GPU 7: RECE $ds"
    cd $RECE_DIR
    CUDA_VISIBLE_DEVICES=7 $P execs/generate_images.py \
      --prompts_path "$pf" \
      --save_path "$outdir" \
      --concept nudity \
      --base 1.4 \
      --guidance_scale 7.5 \
      --image_size 512 \
      --ddim_steps 50 \
      --num_samples 1 2>&1 | tail -3
    cd $REPO
  fi
  eval_qwen 7 "$outdir" nudity
done

# SDErasure nudity (use existing trained model)
SDERASURE_UNET="$REPO/SDErasure/outputs/sderasure_nudity/unet"
if [ -d "$SDERASURE_UNET" ]; then
  for ds in nudity_ringabell nudity_unlearndiff nudity_mma; do
    pf="${PROMPTS[$ds]}"
    outdir=$OUTBASE/sderasure/$ds
    if ! gen_done "$outdir"; then
      mkdir -p "$outdir"
      echo "[$(date +%H:%M)] GPU 7: SDErasure $ds"
      CUDA_VISIBLE_DEVICES=7 $P $REPO/SDErasure/generate_from_prompts.py \
        --model_id CompVis/stable-diffusion-v1-4 \
        --unet_dir "$SDERASURE_UNET" \
        --prompt_file "$pf" \
        --output_dir "$outdir" \
        --seed 42 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 2>&1 | tail -3
    fi
    eval_qwen 7 "$outdir" nudity
  done
else
  echo "WARNING: SDErasure UNet not found at $SDERASURE_UNET"
fi

echo "GPU 7 ALL DONE — $(date)"
) &

wait
echo "=============================================="
echo "  ALL BASELINE EXPERIMENTS DONE — $(date)"
echo "=============================================="
