#!/bin/bash
set -uo pipefail
METHOD=   # safedenoiser|sgf
DATASET=  # rab|unlearndiff|p4dn|mma
GPU=
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
DATA=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/nudity_csv/.csv
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs//nudity/
EXPECTED=-1
ALLDIR=/all
COUNT=0
if [ "" -ge "" ]; then echo "[SKIP /nudity/] all_count= expected="; exit 0; fi
rm -rf ""
mkdir -p "/safe" "/unsafe" "/all"
case "" in
  safedenoiser)
    OREPO=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/Safe_Denoiser
    SCRIPT=run_copro.py
    TASK=configs/nudity/safe_denoiser.yaml
    ERASE=safree_neg_prompt_rep_threshold_time
    ;;
  sgf)
    OREPO=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/SGF/nudity_sdv1
    SCRIPT=generate_unsafe_sgf.py
    TASK=configs/sgf/sgf.yaml
    ERASE=safree_neg_prompt_rep_time
    ;;
  *) echo "unknown official method "; exit 2 ;;
esac
cd ""
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 "" ""   --nudenet-path=pretrained/classifier_model.onnx   --nudity_thr=0.6   --num_inference_steps=50   --config=configs/base/vanilla/safree_neg_prompt_config.json   --safe_level=MEDIUM   --data=""   --category=all   --task_config=""   --save-dir=""   --erase_id=""   --guidance_scale=7.5   --seed=42   --valid_case_numbers=0,100000
