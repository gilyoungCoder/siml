#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grid Search for Sample-Level Monitoring + Spatial CG

Searches over:
- monitoring_threshold: P(harm) threshold for triggering guidance
- spatial_threshold_start/end: CDF threshold for spatial masking
- guidance_scale: strength of guidance

Outputs comparison table and saves all generation stats.
"""

import os
import sys
import json
import subprocess
import itertools
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, default="./scg_outputs/grid_search")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=1234)

    # Grid search parameters (comma-separated)
    parser.add_argument("--monitoring_thresholds", type=str, default="0.3,0.5,0.7,0.9",
                        help="Comma-separated monitoring thresholds")
    parser.add_argument("--guidance_scales", type=str, default="10.0",
                        help="Comma-separated guidance scales")
    parser.add_argument("--spatial_threshold_starts", type=str, default="0.5",
                        help="Comma-separated spatial threshold starts")
    parser.add_argument("--spatial_threshold_ends", type=str, default="0.1",
                        help="Comma-separated spatial threshold ends")
    parser.add_argument("--base_guidance_scales", type=str, default="2.0",
                        help="Comma-separated base guidance scales")

    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    return parser.parse_args()


def run_experiment(config, dry_run=False):
    """Run a single experiment with given config."""
    cmd = [
        "python", "generate_nudity_4class_sample_level_monitoring.py",
        "--ckpt_path", config["ckpt_path"],
        "--prompt_file", config["prompt_file"],
        "--output_dir", config["output_dir"],
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
        "--seed", str(config["seed"]),
    ]

    print(f"\n{'='*60}")
    print(f"Running experiment: {config['exp_name']}")
    print(f"  monitoring_threshold: {config['monitoring_threshold']}")
    print(f"  guidance_scale: {config['guidance_scale']}")
    print(f"  spatial_threshold: {config['spatial_threshold_start']} -> {config['spatial_threshold_end']}")
    print(f"{'='*60}")

    if dry_run:
        print("DRY RUN - Command:")
        print(" ".join(cmd))
        return None

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def load_stats(output_dir):
    """Load generation stats from output directory."""
    stats_file = Path(output_dir) / "generation_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    return None


def main():
    args = parse_args()

    # Parse grid search parameters
    mon_thrs = [float(x) for x in args.monitoring_thresholds.split(",")]
    guid_scales = [float(x) for x in args.guidance_scales.split(",")]
    spat_starts = [float(x) for x in args.spatial_threshold_starts.split(",")]
    spat_ends = [float(x) for x in args.spatial_threshold_ends.split(",")]
    base_scales = [float(x) for x in args.base_guidance_scales.split(",")]

    # Generate all combinations
    combinations = list(itertools.product(mon_thrs, guid_scales, spat_starts, spat_ends, base_scales))
    total_experiments = len(combinations)

    print("="*60)
    print("GRID SEARCH CONFIGURATION")
    print("="*60)
    print(f"monitoring_thresholds: {mon_thrs}")
    print(f"guidance_scales: {guid_scales}")
    print(f"spatial_threshold_starts: {spat_starts}")
    print(f"spatial_threshold_ends: {spat_ends}")
    print(f"base_guidance_scales: {base_scales}")
    print(f"\nTotal experiments: {total_experiments}")
    print("="*60)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_base_dir) / f"grid_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, (mon_thr, guid_scale, spat_start, spat_end, base_scale) in enumerate(combinations):
        exp_name = f"mon{mon_thr}_gs{guid_scale}_sp{spat_start}-{spat_end}_bs{base_scale}"
        output_dir = base_dir / exp_name

        config = {
            "exp_name": exp_name,
            "ckpt_path": args.ckpt_path,
            "prompt_file": args.prompt_file,
            "output_dir": str(output_dir),
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
            "seed": args.seed,
        }

        print(f"\n[{idx+1}/{total_experiments}] {exp_name}")

        returncode = run_experiment(config, dry_run=args.dry_run)

        if not args.dry_run:
            stats = load_stats(output_dir)
            if stats:
                overall = stats.get("overall", {})
                results.append({
                    "exp_name": exp_name,
                    "monitoring_threshold": mon_thr,
                    "guidance_scale": guid_scale,
                    "spatial_start": spat_start,
                    "spatial_end": spat_end,
                    "base_guidance_scale": base_scale,
                    "avg_guided_steps": overall.get("avg_guided_steps", 0),
                    "avg_guidance_ratio": overall.get("avg_guidance_ratio", 0),
                    "no_guidance": overall.get("no_guidance_count", 0),
                    "light_guidance": overall.get("light_guidance_count", 0),
                    "medium_guidance": overall.get("medium_guidance_count", 0),
                    "heavy_guidance": overall.get("heavy_guidance_count", 0),
                    "total_images": overall.get("total_images", 0),
                })

    if not args.dry_run and results:
        # Save results as CSV and JSON
        df = pd.DataFrame(results)
        df.to_csv(base_dir / "grid_search_results.csv", index=False)

        with open(base_dir / "grid_search_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print summary table
        print("\n" + "="*80)
        print("GRID SEARCH RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

        # Save summary
        with open(base_dir / "summary.txt", "w") as f:
            f.write(f"Grid Search Results - {timestamp}\n")
            f.write("="*80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            f.write(f"Results saved to: {base_dir}\n")

        print(f"\nResults saved to: {base_dir}")
        print(f"  - grid_search_results.csv")
        print(f"  - grid_search_results.json")
        print(f"  - summary.txt")

    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run NudeNet evaluation: python evaluate_nudenet.py --input_dir <output_dir>")
    print("  2. Compute FID/CLIP scores")
    print("  3. Compare results across configurations")


if __name__ == "__main__":
    main()
