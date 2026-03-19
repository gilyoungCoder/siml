"""
SHGD Grid Search

Sweep key hyperparameters to find optimal safety-quality tradeoff.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime


# Default grid: strong guidance + critical window sweep
GRID = {
    "anchor_guidance_scale": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    "guide_end_frac": [0.70, 0.75, 0.78, 0.82, 0.85, 0.90],
    "heal_strength": [0.3, 0.4, 0.5, 0.6],
}

# Quick grid
GRID_QUICK = {
    "anchor_guidance_scale": [10.0, 15.0, 20.0, 25.0],
    "guide_end_frac": [0.75, 0.78, 0.82],
    "heal_strength": [0.3, 0.4, 0.5],
}

# Focused grid: zoomed in around promising region
GRID_FOCUSED = {
    "anchor_guidance_scale": [12.0, 15.0, 18.0, 20.0, 25.0],
    "guide_end_frac": [0.76, 0.78, 0.80, 0.82],
    "heal_strength": [0.35, 0.4, 0.45, 0.5],
}


def generate_configs(grid):
    """Generate all combinations of hyperparameters."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    return configs


def config_to_name(config):
    """Create a short name from config values."""
    parts = []
    for k, v in config.items():
        short_key = {
            "anchor_guidance_scale": "ags",
            "guide_start_frac": "gsf",
            "guide_end_frac": "gef",
            "heal_strength": "hs",
            "consistency_threshold": "ct",
        }.get(k, k[:3])
        parts.append(f"{short_key}{v}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="SHGD Grid Search")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Base config file")
    parser.add_argument("--prompt_file", type=str,
                        default="../rab_grid_search/data/ringabell_full.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs for parallel jobs")
    parser.add_argument("--grid", type=str, default="quick",
                        choices=["full", "quick", "focused"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configs without running")

    args = parser.parse_args()

    grid = {"full": GRID, "quick": GRID_QUICK, "focused": GRID_FOCUSED}[args.grid]
    configs = generate_configs(grid)

    if args.output_dir is None:
        args.output_dir = f"outputs/grid_{args.grid}_{datetime.now():%Y%m%d_%H%M%S}"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Grid search: {len(configs)} configurations")
    print(f"Grid type: {args.grid}")
    print(f"Output: {args.output_dir}")
    print(f"GPUs: {args.gpu_ids}")

    if args.dry_run:
        for i, config in enumerate(configs):
            print(f"  [{i+1}] {config_to_name(config)}")
        print(f"\nTotal: {len(configs)} experiments")
        return

    # Save grid config
    with open(os.path.join(args.output_dir, "grid_config.json"), "w") as f:
        json.dump({"grid": grid, "total_configs": len(configs)}, f, indent=2)

    # Parse available GPUs
    gpu_list = args.gpu_ids.split(",")
    n_gpus = len(gpu_list)

    # Run experiments - distribute across GPUs
    processes = {}
    results = []

    for i, config in enumerate(configs):
        exp_name = config_to_name(config)
        exp_dir = os.path.join(args.output_dir, exp_name)

        # Wait for a GPU slot if all busy
        while len(processes) >= n_gpus:
            # Check for completed processes
            for gpu_id, (proc, ename) in list(processes.items()):
                if proc.poll() is not None:
                    print(f"  [GPU {gpu_id}] Completed: {ename} (exit={proc.returncode})")
                    del processes[gpu_id]

            if len(processes) >= n_gpus:
                import time
                time.sleep(5)

        # Find free GPU
        free_gpu = None
        for gpu in gpu_list:
            if gpu not in processes:
                free_gpu = gpu
                break

        print(f"[{i+1}/{len(configs)}] Launching {exp_name} on GPU {free_gpu}")

        python_exe = os.environ.get(
            "SHGD_PYTHON",
            "/mnt/home/yhgil99/.conda/envs/sfgd/bin/python",
        )
        cmd = [
            python_exe, "generate.py",
            "--config", args.config,
            "--prompt_file", args.prompt_file,
            "--save_dir", exp_dir,
            "--device", "cuda:0",
            "--skip_eval",
        ]
        for key, val in config.items():
            cmd.extend([f"--{key}", str(val)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = free_gpu
        env["PYTHONNOUSERSITE"] = "1"

        proc = subprocess.Popen(
            cmd, env=env,
            stdout=open(os.path.join(exp_dir + ".log"), "w") if os.path.exists(args.output_dir) else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        processes[free_gpu] = (proc, exp_name)

    # Wait for remaining
    for gpu_id, (proc, ename) in processes.items():
        proc.wait()
        print(f"  [GPU {gpu_id}] Completed: {ename}")

    # Collect results
    print("\n" + "=" * 70)
    print("Collecting results...")
    print("=" * 70)

    all_results = []
    for config in configs:
        exp_name = config_to_name(config)
        exp_dir = os.path.join(args.output_dir, exp_name)
        result_file = os.path.join(exp_dir, "results.json")

        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            all_results.append({
                "config": config,
                "name": exp_name,
                "safety_rate": summary.get("safety_rate", -1),
                "total": summary.get("total", 0),
                "safe": summary.get("safe", 0),
                "unsafe": summary.get("unsafe", 0),
            })

    # Sort by safety rate
    all_results.sort(key=lambda x: x["safety_rate"], reverse=True)

    print(f"\nTop 10 configurations (by safety rate):")
    print(f"{'Rank':<6} {'Safety%':<10} {'Safe/Total':<12} {'Config'}")
    print("-" * 80)
    for i, r in enumerate(all_results[:10]):
        print(f"{i+1:<6} {r['safety_rate']:.2%}    "
              f"{r['safe']}/{r['total']}       {r['name']}")

    # Save aggregated results
    with open(os.path.join(args.output_dir, "grid_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull results: {args.output_dir}/grid_results.json")


if __name__ == "__main__":
    main()
