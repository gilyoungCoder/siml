#!/usr/bin/env bash
set -euo pipefail

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_dag_adaptive_ss1.0_st0.3"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_dag_adaptive_ss1.0_st0.3" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_dag_adaptive_ss1.0_st0.3"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_dag_adaptive_ss5.0_st0.2"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_dag_adaptive_ss5.0_st0.2" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_dag_adaptive_ss5.0_st0.2"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_hybrid_ss2.0_st0.4"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_hybrid_ss2.0_st0.4" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_both_hybrid_ss2.0_st0.4"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss1.0_st0.3"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss1.0_st0.3" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss1.0_st0.3"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss5.0_st0.2"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss5.0_st0.2" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_dag_adaptive_ss5.0_st0.2"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_hybrid_ss2.0_st0.4"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_hybrid_ss2.0_st0.4" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_image_hybrid_ss2.0_st0.4"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_dag_adaptive_ss1.0_st0.3"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_dag_adaptive_ss1.0_st0.3" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_dag_adaptive_ss1.0_st0.3"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_dag_adaptive_ss5.0_st0.2"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_dag_adaptive_ss5.0_st0.2" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_dag_adaptive_ss5.0_st0.2"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_hybrid_ss2.0_st0.4"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_hybrid_ss2.0_st0.4" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v14/ringabell_text_hybrid_ss2.0_st0.4"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss1.0_st0.3_np16"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss1.0_st0.3_np16" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss1.0_st0.3_np16"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss5.0_st0.2_np16"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss5.0_st0.2_np16" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_dag_adaptive_ss5.0_st0.2_np16"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_hybrid_ss2.0_st0.4_np16"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_hybrid_ss2.0_st0.4_np16" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v15/ringabell_text_hybrid_ss2.0_st0.4_np16"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss1.0_st0.2_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss1.0_st0.2_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss1.0_st0.2_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss2.0_st0.3_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss2.0_st0.3_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss2.0_st0.3_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss3.0_st0.4_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss3.0_st0.4_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_dag_adaptive_ss3.0_st0.4_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss1.0_st0.2_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss1.0_st0.2_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss1.0_st0.2_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss2.0_st0.3_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss2.0_st0.3_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss2.0_st0.3_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss3.0_st0.4_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss3.0_st0.4_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_both_hybrid_ss3.0_st0.4_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss1.0_st0.2_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss1.0_st0.2_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss1.0_st0.2_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss2.0_st0.3_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss2.0_st0.3_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss2.0_st0.3_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss3.0_st0.4_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss3.0_st0.4_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_dag_adaptive_ss3.0_st0.4_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss1.0_st0.2_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss1.0_st0.2_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss1.0_st0.2_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss2.0_st0.3_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss2.0_st0.3_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss2.0_st0.3_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss3.0_st0.4_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss3.0_st0.4_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_image_hybrid_ss3.0_st0.4_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss1.0_st0.2_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss1.0_st0.2_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss1.0_st0.2_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss2.0_st0.3_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss2.0_st0.3_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss2.0_st0.3_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss3.0_st0.4_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss3.0_st0.4_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_dag_adaptive_ss3.0_st0.4_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss1.0_st0.2_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss1.0_st0.2_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss1.0_st0.2_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss2.0_st0.3_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss2.0_st0.3_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss2.0_st0.3_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss3.0_st0.4_probe_only"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss3.0_st0.4_probe_only" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v17/ringabell_text_hybrid_ss3.0_st0.4_probe_only"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss2.0_st0.2_cosine_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss2.0_st0.2_cosine_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss2.0_st0.2_cosine_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss2.0_st0.3_linear_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss2.0_st0.3_linear_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss2.0_st0.3_linear_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss3.0_st0.2_linear_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss3.0_st0.2_linear_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss3.0_st0.2_linear_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss3.0_st0.3_none_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss3.0_st0.3_none_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss3.0_st0.3_none_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss5.0_st0.2_step_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss5.0_st0.2_step_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss5.0_st0.2_step_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss5.0_st0.3_step_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss5.0_st0.3_step_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_dag_adaptive_ss5.0_st0.3_step_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss2.0_st0.3_cosine_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss2.0_st0.3_cosine_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss2.0_st0.3_cosine_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss3.0_st0.2_cosine_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss3.0_st0.2_cosine_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss3.0_st0.2_cosine_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss3.0_st0.3_linear_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss3.0_st0.3_linear_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss3.0_st0.3_linear_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss5.0_st0.2_linear_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss5.0_st0.2_linear_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss5.0_st0.2_linear_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss5.0_st0.3_none_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss5.0_st0.3_none_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_both_hybrid_ss5.0_st0.3_none_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss2.0_st0.2_step_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss2.0_st0.2_step_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss2.0_st0.2_step_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss2.0_st0.3_step_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss2.0_st0.3_step_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss2.0_st0.3_step_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss3.0_st0.3_cosine_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss3.0_st0.3_cosine_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss3.0_st0.3_cosine_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss5.0_st0.2_cosine_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss5.0_st0.2_cosine_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss5.0_st0.2_cosine_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss5.0_st0.3_linear_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss5.0_st0.3_linear_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_dag_adaptive_ss5.0_st0.3_linear_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss2.0_st0.2_linear_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss2.0_st0.2_linear_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss2.0_st0.2_linear_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss2.0_st0.3_none_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss2.0_st0.3_none_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss2.0_st0.3_none_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss3.0_st0.2_step_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss3.0_st0.2_step_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss3.0_st0.2_step_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss3.0_st0.3_step_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss3.0_st0.3_step_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss3.0_st0.3_step_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss5.0_st0.3_cosine_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss5.0_st0.3_cosine_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_image_hybrid_ss5.0_st0.3_cosine_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss2.0_st0.2_cosine_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss2.0_st0.2_cosine_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss2.0_st0.2_cosine_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss2.0_st0.3_linear_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss2.0_st0.3_linear_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss2.0_st0.3_linear_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss3.0_st0.2_linear_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss3.0_st0.2_linear_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss3.0_st0.2_linear_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss3.0_st0.3_none_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss3.0_st0.3_none_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss3.0_st0.3_none_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss5.0_st0.2_step_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss5.0_st0.2_step_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss5.0_st0.2_step_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss5.0_st0.3_step_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss5.0_st0.3_step_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_dag_adaptive_ss5.0_st0.3_step_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss2.0_st0.3_cosine_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss2.0_st0.3_cosine_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss2.0_st0.3_cosine_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss3.0_st0.2_cosine_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss3.0_st0.2_cosine_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss3.0_st0.2_cosine_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss3.0_st0.3_linear_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss3.0_st0.3_linear_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss3.0_st0.3_linear_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss5.0_st0.2_linear_sb1.0"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss5.0_st0.2_linear_sb1.0" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss5.0_st0.2_linear_sb1.0"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss5.0_st0.3_none_sb0.5"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss5.0_st0.3_none_sb0.5" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v18/ringabell_text_hybrid_ss5.0_st0.3_none_sb0.5"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_all_hybrid_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_diverse_hybrid_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_both_top_k_hybrid_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_all_hybrid_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_diverse_hybrid_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_image_top_k_hybrid_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_dag_adaptive_ss5.0_st0.3_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss1.0_st0.4_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss1.0_st0.4_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss1.0_st0.4_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss3.0_st0.2_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss3.0_st0.2_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss3.0_st0.2_multi_probe"

echo "[$(date +%H:%M:%S)] EVAL: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss5.0_st0.3_multi_probe"
cd /mnt/home3/yhgil99/unlearning/vlm && CUDA_VISIBLE_DEVICES=1 /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 opensource_vlm_i2p_all.py "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss5.0_st0.3_multi_probe" nudity qwen 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] DONE: /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v19/ringabell_text_hybrid_ss5.0_st0.3_multi_probe"

echo "GPU 1 ALL COMPLETE (104 folders)"
