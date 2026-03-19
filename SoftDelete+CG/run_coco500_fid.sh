#!/bin/bash
# =============================================================================
# COCO 500 FID Test: Measure how much guidance degrades benign image quality
#
# 1. SD baseline (no guidance) - reference
# 2. Best monitoring configs - measure FID vs baseline
# 3. ASCG no-gating - measure FID vs baseline (expect worst)
#
# GPU 7 is reserved - DO NOT USE
# =============================================================================
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

COCO_PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k_10k.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
OUTPUT_BASE="scg_outputs/coco500_fid"
N_IMAGES=500

mkdir -p "${OUTPUT_BASE}"

# -----------------------------------------------
# Step 1: SD Baseline (no guidance at all)
# -----------------------------------------------
echo "=== [1/4] SD Baseline ==="
if [ ! -d "${OUTPUT_BASE}/sd_baseline" ] || [ $(ls ${OUTPUT_BASE}/sd_baseline/*.png 2>/dev/null | wc -l) -lt $N_IMAGES ]; then
    CUDA_VISIBLE_DEVICES=0 python -c "
import torch, os, csv
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')

# Load prompts
prompts = []
with open('${COCO_PROMPT_FILE}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append(row['prompt'])
        if len(prompts) >= ${N_IMAGES}:
            break

outdir = '${OUTPUT_BASE}/sd_baseline'
os.makedirs(outdir, exist_ok=True)

gen = torch.Generator('cuda').manual_seed(42)
for i, prompt in enumerate(prompts):
    outpath = os.path.join(outdir, f'{i:04d}.png')
    if os.path.exists(outpath):
        continue
    img = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=gen).images[0]
    img.save(outpath)
    if i % 50 == 0:
        print(f'  Baseline: {i}/{len(prompts)}')
print(f'  Baseline done: {len(prompts)} images')
"
fi

# -----------------------------------------------
# Step 2: Generate with monitoring configs (parallel)
# -----------------------------------------------
generate_monitored() {
    local NAME=$1
    local GPU=$2
    local MODE=$3
    local MT=$4
    local MS=$5
    local GS=$6
    local BS=$7
    local SP_S=$8
    local SP_E=$9
    local SPATIAL=${10}

    local OUTDIR="${OUTPUT_BASE}/${NAME}"

    if [ -d "${OUTDIR}" ] && [ $(ls ${OUTDIR}/*.png 2>/dev/null | wc -l) -ge $N_IMAGES ]; then
        echo "[GPU ${GPU}] SKIP ${NAME} (already done)"
        return 0
    fi

    echo "[GPU ${GPU}] START ${NAME}"
    CUDA_VISIBLE_DEVICES=${GPU} python generate_unified_monitoring.py \
        --ckpt_path CompVis/stable-diffusion-v1-4 \
        --prompt_file "${COCO_PROMPT_FILE}" \
        --output_dir "${OUTDIR}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --gradcam_layer "encoder_model.middle_block.2" \
        --monitoring_mode "${MODE}" \
        --monitoring_threshold ${MT} \
        --monitoring_start_step ${MS} \
        --spatial_mode "${SPATIAL}" \
        --guidance_scale ${GS} \
        --base_guidance_scale ${BS} \
        --spatial_threshold_start ${SP_S} \
        --spatial_threshold_end ${SP_E} \
        --spatial_threshold_strategy cosine \
        --end_idx ${N_IMAGES} \
        --seed 42
    echo "[GPU ${GPU}] DONE ${NAME}"
}

# Best balanced: z0_sticky_gradcam_mt0.06_gs20.0_bs3.0_sp0.3-0.5_ms10
generate_monitored "z0_sticky_gradcam_mt006_gs20_ms10" 1 "z0_sticky" 0.06 10 20.0 3.0 0.3 0.5 "gradcam" &
PID1=$!

# Second best: z0_sticky_gradcam_mt0.03_gs20.0_bs2.0_sp0.2-0.4_ms10
generate_monitored "z0_sticky_gradcam_mt003_gs20_ms10" 2 "z0_sticky" 0.03 10 20.0 2.0 0.2 0.4 "gradcam" &
PID2=$!

# High SR config: z0_sticky_gradcam_mt0.01_gs10.0_bs3.0_sp0.3-0.5_ms0
generate_monitored "z0_sticky_gradcam_mt001_gs10_ms0" 3 "z0_sticky" 0.01 0 10.0 3.0 0.3 0.5 "gradcam" &
PID3=$!

wait $PID1 $PID2 $PID3

# -----------------------------------------------
# Step 3: ASCG no-gating (for comparison)
# -----------------------------------------------
echo "=== [3/4] ASCG no-gating ==="
if [ ! -d "${OUTPUT_BASE}/ascg_nogating" ] || [ $(ls ${OUTPUT_BASE}/ascg_nogating/*.png 2>/dev/null | wc -l) -lt $N_IMAGES ]; then
    CUDA_VISIBLE_DEVICES=4 python generate_sgf_window_unified.py \
        --ckpt_path CompVis/stable-diffusion-v1-4 \
        --prompt_file "${COCO_PROMPT_FILE}" \
        --output_dir "${OUTPUT_BASE}/ascg_nogating" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --gradcam_layer "encoder_model.middle_block.2" \
        --spatial_mode "gradcam" \
        --guidance_scale 20.0 \
        --base_guidance_scale 3.0 \
        --spatial_threshold_start 0.2 \
        --spatial_threshold_end 0.3 \
        --spatial_threshold_strategy cosine \
        --window_start 1000 \
        --window_end 400 \
        --guidance_schedule linear \
        --end_idx ${N_IMAGES} \
        --seed 42
fi

# -----------------------------------------------
# Step 4: Compute FID and CLIP scores
# -----------------------------------------------
echo "=== [4/4] Computing FID & CLIP scores ==="
python3 << 'PYEOF'
import subprocess, os, json

base = "scg_outputs/coco500_fid"
ref_dir = os.path.join(base, "sd_baseline")

results = {}
for name in sorted(os.listdir(base)):
    img_dir = os.path.join(base, name)
    if not os.path.isdir(img_dir) or name == "sd_baseline":
        continue

    n_imgs = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if n_imgs < 100:
        print(f"  SKIP {name}: only {n_imgs} images")
        continue

    print(f"  Computing FID: {name} vs sd_baseline ({n_imgs} images)...")

    # pytorch-fid
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", ref_dir, img_dir, "--device", "cuda:0"],
        capture_output=True, text=True
    )
    fid = None
    for line in result.stdout.split('\n'):
        if 'FID' in line:
            try:
                fid = float(line.split(':')[-1].strip())
            except:
                pass

    results[name] = {"fid": fid, "n_images": n_imgs}
    print(f"    FID = {fid}")

# Save results
with open(os.path.join(base, "fid_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*70)
print("  COCO 500 FID Results (vs SD Baseline)")
print("="*70)
print(f"  {'Config':<45s} {'FID':>8s} {'N':>6s}")
print("-"*65)
for name, r in sorted(results.items(), key=lambda x: x[1].get('fid') or 999):
    fid_str = f"{r['fid']:.2f}" if r['fid'] is not None else "ERROR"
    print(f"  {name:<45s} {fid_str:>8s} {r['n_images']:>6d}")
print("\n  Lower FID = less degradation of benign images")
PYEOF

echo ""
echo "============================================"
echo "COCO 500 FID TEST COMPLETE!"
echo "============================================"
