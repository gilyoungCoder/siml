#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-GPU Grid Search for Sample-Level Monitoring + Spatial CG

Distributes experiments across multiple GPUs for efficient parameter search.

Usage:
    # Dry run to see all experiments
    python grid_search_multi_gpu.py --prompt_file sexual.csv --dry_run

    # Run on 8 GPUs
    python grid_search_multi_gpu.py --prompt_file sexual.csv --num_gpus 8

    # Run specific subset
    python grid_search_multi_gpu.py --prompt_file sexual.csv --start_idx 0 --end_idx 100
"""

import os
import sys
import json
import subprocess
import itertools
import multiprocessing
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Prompt file (txt or csv). Can be absolute path or filename in i2p dir")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth")
    parser.add_argument("--gradcam_stats_dir", type=str, default="./gradcam_stats/nudity_4class")
    parser.add_argument("--output_base_dir", type=str, default="./scg_outputs/grid_search_multi")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    # Multi-GPU settings
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Specific GPU IDs to use (comma-separated, e.g., '0,1,2,3')")

    # Grid search parameters (comma-separated)
    parser.add_argument("--monitoring_thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated monitoring thresholds")
    parser.add_argument("--guidance_scales", type=str, default="7.5,10,12.5,15",
                        help="Comma-separated guidance scales")
    parser.add_argument("--spatial_threshold_starts", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated spatial threshold starts")
    parser.add_argument("--spatial_threshold_ends", type=str, default="0.3,0.5",
                        help="Comma-separated spatial threshold ends")
    parser.add_argument("--base_guidance_scales", type=str, default="0.0,1.0,2.0",
                        help="Comma-separated base guidance scales")

    # Fixed parameter
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine",
                        help="Threshold annealing strategy (fixed to cosine)")

    # Experiment range (for partial runs)
    parser.add_argument("--start_idx", type=int, default=0, help="Start experiment index")
    parser.add_argument("--end_idx", type=int, default=-1, help="End experiment index (-1 for all)")

    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def resolve_prompt_file(prompt_file: str) -> str:
    """Resolve prompt file path. Check i2p directory if not absolute."""
    if os.path.isabs(prompt_file) and os.path.exists(prompt_file):
        return prompt_file

    # Check current directory
    if os.path.exists(prompt_file):
        return os.path.abspath(prompt_file)

    # Check i2p directory
    i2p_dir = "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"
    i2p_path = os.path.join(i2p_dir, prompt_file)
    if os.path.exists(i2p_path):
        return i2p_path

    # Check with .csv extension
    if not prompt_file.endswith(".csv"):
        i2p_csv = os.path.join(i2p_dir, prompt_file + ".csv")
        if os.path.exists(i2p_csv):
            return i2p_csv

    raise FileNotFoundError(f"Prompt file not found: {prompt_file}")


def run_experiment(config: dict) -> dict:
    """Run a single experiment and return result."""
    gpu_id = config["gpu_id"]
    exp_name = config["exp_name"]
    output_dir = config["output_dir"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "generate_nudity_4class_sample_level_monitoring.py",
        "--ckpt_path", config["ckpt_path"],
        "--prompt_file", config["prompt_file"],
        "--output_dir", output_dir,
        "--nsamples", str(config["nsamples"]),
        "--cfg_scale", str(config["cfg_scale"]),
        "--num_inference_steps", str(config["num_inference_steps"]),
        "--classifier_ckpt", config["classifier_ckpt"],
        "--gradcam_stats_dir", config["gradcam_stats_dir"],
        "--monitoring_threshold", str(config["monitoring_threshold"]),
        "--guidance_scale", str(config["guidance_scale"]),
        "--base_guidance_scale", str(config["base_guidance_scale"]),
        "--spatial_threshold_start", str(config["spatial_threshold_start"]),
        "--spatial_threshold_end", str(config["spatial_threshold_end"]),
        "--spatial_threshold_strategy", config["spatial_threshold_strategy"],
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=config["cwd"],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        # Load stats if successful
        stats = None
        stats_file = Path(output_dir) / "generation_stats.json"
        if success and stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)

        return {
            "exp_name": exp_name,
            "exp_idx": config["exp_idx"],
            "gpu_id": gpu_id,
            "success": success,
            "elapsed": elapsed,
            "returncode": result.returncode,
            "stats": stats,
            "stderr": result.stderr[-500:] if result.stderr else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "exp_name": exp_name,
            "exp_idx": config["exp_idx"],
            "gpu_id": gpu_id,
            "success": False,
            "elapsed": 7200,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "exp_name": exp_name,
            "exp_idx": config["exp_idx"],
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
        }


def main():
    args = parse_args()

    # Resolve prompt file
    prompt_file = resolve_prompt_file(args.prompt_file)
    print(f"Using prompt file: {prompt_file}")

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))
    num_gpus = len(gpu_ids)

    # Parse grid search parameters
    mon_thrs = [float(x) for x in args.monitoring_thresholds.split(",")]
    guid_scales = [float(x) for x in args.guidance_scales.split(",")]
    spat_starts = [float(x) for x in args.spatial_threshold_starts.split(",")]
    spat_ends = [float(x) for x in args.spatial_threshold_ends.split(",")]
    base_scales = [float(x) for x in args.base_guidance_scales.split(",")]

    # Generate all combinations
    combinations = list(itertools.product(mon_thrs, guid_scales, spat_starts, spat_ends, base_scales))
    total_experiments = len(combinations)

    # Apply range filter
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else total_experiments
    combinations = combinations[start_idx:end_idx]
    num_experiments = len(combinations)

    print("=" * 70)
    print("MULTI-GPU GRID SEARCH")
    print("=" * 70)
    print(f"monitoring_thresholds: {mon_thrs}")
    print(f"guidance_scales: {guid_scales}")
    print(f"spatial_threshold_starts: {spat_starts}")
    print(f"spatial_threshold_ends: {spat_ends}")
    print(f"base_guidance_scales: {base_scales}")
    print(f"spatial_threshold_strategy: {args.spatial_threshold_strategy} (fixed)")
    print(f"\nTotal combinations: {total_experiments}")
    print(f"Running experiments: {start_idx} to {end_idx} ({num_experiments} experiments)")
    print(f"GPUs: {gpu_ids}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:\n")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_base_dir) / f"grid_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_info = {
        "timestamp": timestamp,
        "prompt_file": prompt_file,
        "total_experiments": total_experiments,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_experiments": num_experiments,
        "gpu_ids": gpu_ids,
        "parameters": {
            "monitoring_thresholds": mon_thrs,
            "guidance_scales": guid_scales,
            "spatial_threshold_starts": spat_starts,
            "spatial_threshold_ends": spat_ends,
            "base_guidance_scales": base_scales,
            "spatial_threshold_strategy": args.spatial_threshold_strategy,
        },
        "args": vars(args),
    }
    with open(base_dir / "grid_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    # Prepare experiment configs
    experiment_configs = []
    for idx, (mon_thr, guid_scale, spat_start, spat_end, base_scale) in enumerate(combinations):
        global_idx = start_idx + idx
        exp_name = f"mon{mon_thr}_gs{guid_scale}_sp{spat_start}-{spat_end}_bs{base_scale}"
        output_dir = str(base_dir / exp_name)
        gpu_id = gpu_ids[idx % num_gpus]

        config = {
            "exp_idx": global_idx,
            "exp_name": exp_name,
            "gpu_id": gpu_id,
            "output_dir": output_dir,
            "ckpt_path": args.ckpt_path,
            "prompt_file": prompt_file,
            "nsamples": args.nsamples,
            "cfg_scale": args.cfg_scale,
            "num_inference_steps": args.num_inference_steps,
            "classifier_ckpt": args.classifier_ckpt,
            "gradcam_stats_dir": args.gradcam_stats_dir,
            "monitoring_threshold": mon_thr,
            "guidance_scale": guid_scale,
            "base_guidance_scale": base_scale,
            "spatial_threshold_start": spat_start,
            "spatial_threshold_end": spat_end,
            "spatial_threshold_strategy": args.spatial_threshold_strategy,
            "cwd": str(Path(__file__).parent.absolute()),
        }
        experiment_configs.append(config)

        if args.dry_run:
            print(f"[{global_idx:03d}] GPU {gpu_id}: {exp_name}")
            if args.verbose:
                print(f"       Output: {output_dir}")

    if args.dry_run:
        print(f"\nTotal: {num_experiments} experiments on {num_gpus} GPUs")
        print(f"Output directory: {base_dir}")
        return

    # Run experiments in parallel
    print(f"\nStarting experiments... (output: {base_dir})")
    results = []
    completed = 0

    # Use process pool with num_gpus workers
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(run_experiment, cfg): cfg for cfg in experiment_configs}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            status = "OK" if result.get("success") else "FAIL"
            elapsed = result.get("elapsed", 0)
            print(f"[{completed}/{num_experiments}] {result['exp_name']} - {status} ({elapsed:.1f}s)")

            # Save intermediate results
            with open(base_dir / "results_partial.json", "w") as f:
                json.dump(results, f, indent=2)

    # Final results
    successful = sum(1 for r in results if r.get("success"))
    failed = num_experiments - successful

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE!")
    print("=" * 70)
    print(f"Successful: {successful}/{num_experiments}")
    print(f"Failed: {failed}/{num_experiments}")
    print(f"Output directory: {base_dir}")

    # Save final results
    final_results = {
        "config": config_info,
        "summary": {
            "successful": successful,
            "failed": failed,
            "total": num_experiments,
        },
        "experiments": results,
    }
    with open(base_dir / "results_final.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Create summary CSV
    try:
        import pandas as pd
        rows = []
        for r in results:
            row = {
                "exp_name": r["exp_name"],
                "success": r.get("success", False),
                "elapsed": r.get("elapsed", 0),
            }
            if r.get("stats") and r["stats"].get("overall"):
                o = r["stats"]["overall"]
                row.update({
                    "avg_guided_steps": o.get("avg_guided_steps", 0),
                    "avg_guidance_ratio": o.get("avg_guidance_ratio", 0),
                    "no_guidance": o.get("no_guidance_count", 0),
                    "light_guidance": o.get("light_guidance_count", 0),
                    "medium_guidance": o.get("medium_guidance_count", 0),
                    "heavy_guidance": o.get("heavy_guidance_count", 0),
                })
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(base_dir / "results_summary.csv", index=False)
        print(f"\nSummary CSV saved to: {base_dir / 'results_summary.csv'}")
    except ImportError:
        print("(pandas not available, skipping CSV export)")

    print("\nNext steps:")
    print("  1. Run NudeNet evaluation on each output directory")
    print("  2. Compute FID/CLIP scores")
    print("  3. Analyze results with: python analyze_grid_results.py --input_dir", base_dir)


if __name__ == "__main__":
    main()
