#!/bin/bash
# =============================================================================
# COCO 500 FID Test - PARALLEL version using GPUs 0-6
# GPU 7 is RESERVED - DO NOT USE
#
# 1. SD baseline (no guidance) split across GPUs
# 2. Best monitoring configs in parallel
# 3. ASCG no-gating
# 4. FID computation
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
# Step 1: SD Baseline - parallel across 7 GPUs
# -----------------------------------------------
echo "=== [1/4] SD Baseline (parallel, 7 GPUs) ==="
BASELINE_DIR="${OUTPUT_BASE}/sd_baseline"
mkdir -p "${BASELINE_DIR}"

# Split 500 images across 7 GPUs (~72 each)
CHUNK=72
for GPU in 0 1 2 3 4 5 6; do
    START=$((GPU * CHUNK))
    END=$(( (GPU + 1) * CHUNK ))
    if [ $GPU -eq 6 ]; then
        END=$N_IMAGES
    fi
    if [ $START -ge $N_IMAGES ]; then
        break
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python -c "
import torch, os, csv
from diffusers import StableDiffusionPipeline, DDIMScheduler

pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')

prompts = []
with open('${COCO_PROMPT_FILE}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append(row['prompt'])
        if len(prompts) >= ${N_IMAGES}:
            break

outdir = '${BASELINE_DIR}'
for i in range(${START}, min(${END}, len(prompts))):
    outpath = os.path.join(outdir, f'{i:04d}.png')
    if os.path.exists(outpath):
        continue
    gen = torch.Generator('cuda').manual_seed(42 + i)
    img = pipe(prompts[i], num_inference_steps=50, guidance_scale=7.5, generator=gen).images[0]
    img.save(outpath)
print(f'[GPU ${GPU}] Baseline done: {${START}}-{${END}}')
" &
done
echo "Waiting for baseline generation..."
wait
echo "Baseline done! $(ls ${BASELINE_DIR}/*.png | wc -l) images"

# -----------------------------------------------
# Step 2: Monitoring configs (parallel on GPUs 0-4)
# -----------------------------------------------
echo "=== [2/4] Monitoring configs ==="

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
        echo "[GPU ${GPU}] SKIP ${NAME}"
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

# Best balanced: z0_sticky_gradcam_mt0.06 gs20 bs3 sp0.3-0.5 ms10
generate_monitored "z0_sticky_gradcam_mt006" 0 "z0_sticky" 0.06 10 20.0 3.0 0.3 0.5 "gradcam" &

# Second: z0_sticky_gradcam_mt0.03 gs20 bs2 sp0.2-0.4 ms10
generate_monitored "z0_sticky_gradcam_mt003" 1 "z0_sticky" 0.03 10 20.0 2.0 0.2 0.4 "gradcam" &

# High SR: z0_sticky_gradcam_mt0.01 gs10 bs3 sp0.3-0.5 ms0
generate_monitored "z0_sticky_gradcam_mt001" 2 "z0_sticky" 0.01 0 10.0 3.0 0.3 0.5 "gradcam" &

# Attention based: gradcam_attention_mt0.05 gs10 bs2 sp0.2-0.3 ms10
generate_monitored "gradcam_attention_mt005" 3 "gradcam" 0.05 10 10.0 2.0 0.2 0.3 "attention" &

# -----------------------------------------------
# Step 3: ASCG no-gating (worst case for FP)
# -----------------------------------------------
echo "=== [3/4] ASCG no-gating ==="
ASCG_DIR="${OUTPUT_BASE}/ascg_nogating"
if [ ! -d "${ASCG_DIR}" ] || [ $(ls ${ASCG_DIR}/*.png 2>/dev/null | wc -l) -lt $N_IMAGES ]; then
    CUDA_VISIBLE_DEVICES=4 python generate_sgf_window_unified.py \
        --ckpt_path CompVis/stable-diffusion-v1-4 \
        --prompt_file "${COCO_PROMPT_FILE}" \
        --output_dir "${ASCG_DIR}" \
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
        --seed 42 &
fi

wait
echo "All generation done!"

# -----------------------------------------------
# Step 4: FID computation
# -----------------------------------------------
echo "=== [4/4] Computing FID ==="
python3 << 'PYEOF'
import subprocess, os, json

base = "scg_outputs/coco500_fid"
ref_dir = os.path.join(base, "sd_baseline")

n_ref = len([f for f in os.listdir(ref_dir) if f.endswith('.png')])
print(f"Reference (sd_baseline): {n_ref} images")

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
    if fid is None:
        print(f"    STDOUT: {result.stdout[:200]}")
        print(f"    STDERR: {result.stderr[:200]}")

    # Also get generation stats if available
    sf = os.path.join(img_dir, 'generation_stats.json')
    no_guid = None
    if os.path.isfile(sf):
        s = json.load(open(sf))
        no_guid = s['overall'].get('no_guidance_count', 0)
        total = s['overall'].get('total_images', n_imgs)
        fp_rate = 100 * (1 - no_guid / total)
    else:
        fp_rate = None

    results[name] = {"fid": fid, "n_images": n_imgs, "fp_rate": fp_rate}
    print(f"    FID = {fid}, FP = {fp_rate}%")

with open(os.path.join(base, "fid_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*75)
print("  COCO 500 FID Results (vs SD Baseline = no guidance)")
print("="*75)
print(f"  {'Config':<40s} {'FID':>8s} {'COCO FP%':>10s} {'N':>6s}")
print("-"*70)
for name, r in sorted(results.items(), key=lambda x: x[1].get('fid') or 999):
    fid_str = f"{r['fid']:.2f}" if r['fid'] is not None else "ERROR"
    fp_str = f"{r['fp_rate']:.0f}%" if r['fp_rate'] is not None else "N/A"
    print(f"  {name:<40s} {fid_str:>8s} {fp_str:>10s} {r['n_images']:>6d}")
print()
print("  FID↓ = less degradation of benign images")
print("  FP% = % of COCO prompts where guidance was triggered (should be 0%)")
PYEOF

echo ""
echo "============================================"
echo "COCO 500 FID TEST COMPLETE!"
echo "============================================"
