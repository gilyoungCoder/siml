#!/usr/bin/env python3
"""
Launch SoftDelete+CG grid search on 8 GPUs.
Generates worker configs and spawns scg_grid_worker.py on each GPU.

Usage:
    PYTHONNOUSERSITE=1 python launch_scg_grid.py --phase rab
    PYTHONNOUSERSITE=1 python launch_scg_grid.py --phase coco
    PYTHONNOUSERSITE=1 python launch_scg_grid.py --phase both
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/mnt/home/yhgil99/.conda/envs/safree/bin/python"
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "scg_grid_worker.py")
NUM_GPUS = 8
GPU_IDS = list(range(NUM_GPUS))

# Paths
SCG_DIR = "/mnt/home/yhgil99/unlearning/SoftDelete+CG"
CLASSIFIER_CKPT = f"{SCG_DIR}/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR = f"{SCG_DIR}/gradcam_stats/nudity_4class_ringabell"
RAB_TEST = "/mnt/home/yhgil99/unlearning/rab_grid_search/data/ringabell_test.txt"
COCO_10K = "/mnt/home/yhgil99/unlearning/prompts/coco/coco_10k.txt"
OUTPUT_BASE = "/mnt/home/yhgil99/unlearning/rab_grid_search/results"

# Grid search parameters
PARAM_GRID = {
    "guidance_scale": [5, 10, 20, 50],
    "spatial_threshold_start": [0.2, 0.3, 0.5, 0.7],
    "spatial_threshold_end": [0.1],  # Not used for constant strategy
    "threshold_strategy": ["constant"],
    "harmful_scale": [0.5, 1.0, 2.0],
    "base_guidance_scale": [0.0],  # Guidance outside mask region
}


def generate_experiments():
    """Generate all experiment combinations."""
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]

    experiments = []
    for combo in itertools.product(*values):
        exp = dict(zip(keys, combo))
        experiments.append(exp)

    # Deduplicate: different thresholds don't matter if strategy doesn't use them
    seen = set()
    unique = []
    for e in experiments:
        # For constant strategy, threshold_end doesn't matter
        key_parts = [
            f"gs{e['guidance_scale']}",
            f"st{e['spatial_threshold_start']}",
            f"{e['threshold_strategy']}",
            f"hs{e['harmful_scale']}",
        ]
        key = "_".join(key_parts)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def run_phase(phase_name, prompt_file, output_root, max_prompts=None):
    """Run grid search for one phase (RAB or COCO)."""
    experiments = generate_experiments()
    print(f"\n{'='*60}")
    print(f"SCG Grid Search: {phase_name}")
    print(f"{'='*60}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Prompt file: {prompt_file}")
    print(f"  Output: {output_root}")

    # Create output directory with timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    grid_dir = os.path.join(output_root, f"grid_{ts}")
    os.makedirs(grid_dir, exist_ok=True)

    # Split experiments across GPUs
    exps_per_gpu = [[] for _ in range(NUM_GPUS)]
    for i, exp in enumerate(experiments):
        exps_per_gpu[i % NUM_GPUS].append(exp)

    # Common config
    common = {
        "ckpt_path": "CompVis/stable-diffusion-v1-4",
        "classifier_ckpt": CLASSIFIER_CKPT,
        "gradcam_stats_dir": GRADCAM_STATS_DIR,
        "prompt_file": prompt_file,
        "output_dir": grid_dir,
        "seed": 42,
        "num_inference_steps": 50,
        "cfg_scale": 7.5,
        "nsamples": 1,
    }
    if max_prompts:
        common["max_prompts"] = max_prompts

    # Create worker configs and launch
    processes = []
    for gpu_id in GPU_IDS:
        if not exps_per_gpu[gpu_id]:
            continue

        worker_config = {
            "gpu_id": gpu_id,
            "common": common,
            "experiments": exps_per_gpu[gpu_id],
        }

        config_path = os.path.join(grid_dir, f"worker_gpu{gpu_id}.json")
        with open(config_path, "w") as f:
            json.dump(worker_config, f, indent=2)

        log_path = os.path.join(grid_dir, f"worker_gpu{gpu_id}.log")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONNOUSERSITE"] = "1"

        cmd = [PYTHON, WORKER_SCRIPT, "--config", config_path]
        print(f"  [GPU {gpu_id}] {len(exps_per_gpu[gpu_id])} experiments -> {config_path}")

        with open(log_path, "w") as log_f:
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            processes.append((gpu_id, p, log_path))

    print(f"\n  Launched {len(processes)} workers. Waiting for completion...")
    print(f"  Grid dir: {grid_dir}")

    # Wait for all workers
    for gpu_id, p, log_path in processes:
        retcode = p.wait()
        if retcode != 0:
            print(f"  [GPU {gpu_id}] FAILED (exit code {retcode}). Check: {log_path}")
        else:
            print(f"  [GPU {gpu_id}] Done.")

    print(f"\n  {phase_name} complete! Results: {grid_dir}")
    return grid_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="both",
                        choices=["rab", "coco", "both"])
    args = parser.parse_args()

    print("SoftDelete+CG Grid Search Launcher")
    print(f"  Classifier: {CLASSIFIER_CKPT}")
    print(f"  GradCAM stats: {GRADCAM_STATS_DIR}")
    print(f"  GPUs: {GPU_IDS}")

    if args.phase in ("rab", "both"):
        run_phase(
            "RAB Test (40 prompts)",
            RAB_TEST,
            os.path.join(OUTPUT_BASE, "scg_gen_rab_test"),
        )

    if args.phase in ("coco", "both"):
        # Create COCO 50 subset
        coco_50_path = "/tmp/coco_50_scg.txt"
        with open(COCO_10K) as f:
            lines = [l.strip() for l in f if l.strip()][:50]
        with open(coco_50_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        run_phase(
            "COCO (50 prompts)",
            coco_50_path,
            os.path.join(OUTPUT_BASE, "scg_gen_coco50"),
        )

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
