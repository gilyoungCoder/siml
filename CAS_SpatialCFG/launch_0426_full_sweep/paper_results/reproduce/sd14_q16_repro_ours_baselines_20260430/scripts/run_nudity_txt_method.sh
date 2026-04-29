#!/bin/bash
set -uo pipefail
METHOD=   # safree
DATASET=  # rab|unlearndiff|p4dn|mma
GPU=
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
case "" in
  rab) PROMPTS=/CAS_SpatialCFG/prompts/ringabell.txt ;;
  unlearndiff) PROMPTS=/CAS_SpatialCFG/prompts/unlearndiff.txt ;;
  p4dn) PROMPTS=/CAS_SpatialCFG/prompts/p4dn.txt ;;
  mma) PROMPTS=/CAS_SpatialCFG/prompts/mma.txt ;;
  *) echo "unknown dataset "; exit 2 ;;
esac
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs//nudity/
mkdir -p ""
EXPECTED=
COUNT=0
if [ "" -ge "" ]; then echo "[SKIP /nudity/] count= expected="; exit 0; fi
case "" in
  safree)
    cd "/SAFREE"
    CUDA_VISIBLE_DEVICES= "" gen_safree_single.py       --txt "" --save-dir ""       --model_id CompVis/stable-diffusion-v1-4 --category nudity       --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5       --seed 42 --image_length 512 --device cuda:0 --erase-id std       --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2"       --safree -svf -lra --linear_per_prompt_seed
    ;;
  *) echo "unknown nudity txt method "; exit 2 ;;
esac
