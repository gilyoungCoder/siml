#!/usr/bin/env python3
"""
Analyze monitoring results and run targeted generation grid search.

After Phase 1 (monitoring), this script:
1. Analyzes COCO FP and RAB TP rates across thresholds
2. Identifies parameter ranges where COCO FP < 10% and RAB TP > 90%
3. Launches targeted generation grid search with those params

Usage:
    # After Phase 1 monitoring is done
    python analyze_and_generate.py --monitoring_dir <phase1_results> --num_gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = "/mnt/home/yhgil99/unlearning"
SD_MODEL = "CompVis/stable-diffusion-v1-4"
PYTHON = "/mnt/home/yhgil99/.conda/envs/safree/bin/python"
CLEAN_ENV_VARS = {"PYTHONNOUSERSITE": "1"}


def analyze_monitoring(monitoring_dir):
    """
    Analyze Phase 1 monitoring results.
    Returns dict of threshold -> {coco_fp, rab_tp}.
    """
    results = {}

    # Load results from monitoring test output
    for ds_name in ["z0_coco", "z0_rab_test"]:
        ds_dir = os.path.join(monitoring_dir, ds_name)
        if not os.path.exists(ds_dir):
            print(f"  WARNING: {ds_dir} not found")
            continue

        # Find JSON result files
        all_results = []
        for f in sorted(Path(ds_dir).glob("*.json")):
            if f.name.startswith("merged") or f.name.endswith("_merged.json"):
                with open(f) as fp:
                    data = json.load(fp)
                if "results" in data:
                    all_results = data["results"]
                break

        if not all_results:
            # Try individual GPU files
            for f in sorted(Path(ds_dir).glob("*gpu*.json")):
                with open(f) as fp:
                    data = json.load(fp)
                if "results" in data:
                    all_results.extend(data["results"])

        if all_results:
            results[ds_name] = [r["max_p_harm"] for r in all_results]
            print(f"  {ds_name}: {len(all_results)} prompts")

    # Compute threshold analysis
    thresholds = np.arange(0.05, 1.0, 0.05).tolist()
    threshold_analysis = {}

    coco_ps = results.get("z0_coco", [])
    rab_ps = results.get("z0_rab_test", [])

    if not coco_ps or not rab_ps:
        print("  WARNING: Missing monitoring data")
        return threshold_analysis

    print(f"\n{'Threshold':>12} | {'COCO FP':>10} | {'RAB TP':>10} | {'Gap':>8} | {'Status'}")
    print("-" * 65)

    best_gap = -1
    best_thr = None

    for thr in thresholds:
        coco_fp = sum(1 for p in coco_ps if p > thr) / len(coco_ps)
        rab_tp = sum(1 for p in rab_ps if p > thr) / len(rab_ps)
        gap = rab_tp - coco_fp

        status = ""
        if coco_fp <= 0.10 and rab_tp >= 0.90:
            status = "*** IDEAL ***"
        elif coco_fp <= 0.15 and rab_tp >= 0.85:
            status = "GOOD"
        elif coco_fp <= 0.20 and rab_tp >= 0.80:
            status = "ok"

        if gap > best_gap:
            best_gap = gap
            best_thr = thr

        threshold_analysis[thr] = {
            "coco_fp": coco_fp,
            "rab_tp": rab_tp,
            "gap": gap,
        }

        print(f"  {thr:>10.2f} | {coco_fp:>9.1%} | {rab_tp:>9.1%} | {gap:>7.1%} | {status}")

    print(f"\n  Best gap threshold: {best_thr:.2f} "
          f"(FP={threshold_analysis[best_thr]['coco_fp']:.1%}, "
          f"TP={threshold_analysis[best_thr]['rab_tp']:.1%})")

    return threshold_analysis


def run_targeted_z0_grid(num_gpus, gpu_ids, output_dir, prompt_file, ds_label):
    """
    Run z0 generation grid search with targeted parameters.
    Uses grid_search_spatial_cg.py from z0_clf_guidance.
    """
    clf_dir = f"{BASE_DIR}/z0_clf_guidance"
    clf_ckpt = f"{clf_dir}/work_dirs/z0_resnet18_4class_ringabell/checkpoint/step_15900/classifier.pth"
    harmful_stats = f"{clf_dir}/harmful_stats_4class_ringabell_layer2.pt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Two guidance mode configs
    configs = [
        {
            "mode": "safe_minus_harm",
            "safe_classes": [0, 1],
            "harm_classes": [2, 3],
            "target_class": 1,
        },
        {
            "mode": "target",
            "safe_classes": None,
            "harm_classes": None,
            "target_class": 1,
        },
    ]

    for gc in configs:
        mode_tag = gc["mode"]
        run_output = os.path.join(output_dir, f"z0_{ds_label}_{mode_tag}_{timestamp}")

        cmd = [
            PYTHON,
            os.path.join(clf_dir, "grid_search_spatial_cg.py"),
            "--ckpt_path", SD_MODEL,
            "--prompt_file", prompt_file,
            "--classifier_ckpt", clf_ckpt,
            "--harmful_stats_path", harmful_stats,
            "--output_root", run_output,
            "--architecture", "resnet18",
            "--num_classes", "4",
            "--space", "latent",
            "--guidance_mode", gc["mode"],
            "--target_class", str(gc["target_class"]),
            "--gradcam_layer", "layer2",
            "--num_inference_steps", "50",
            "--cfg_scale", "7.5",
            "--seed", "1234",
            "--nsamples", "1",
            "--num_gpus", str(num_gpus),
            "--gpu_ids", ",".join(str(g) for g in gpu_ids),
            # Aggressive grid
            "--guidance_scales", "3", "5", "7.5", "10", "15", "20", "30", "50",
            "--spatial_modes", "none", "gradcam",
            "--spatial_thresholds", "0.1", "0.2", "0.3", "0.5", "0.7",
            "--spatial_soft_options", "0", "1",
            "--threshold_schedules", "constant", "cosine",
            "--harm_ratios", "1.0",
        ]

        if gc["safe_classes"]:
            cmd += ["--safe_classes"] + [str(c) for c in gc["safe_classes"]]
        if gc["harm_classes"]:
            cmd += ["--harm_classes"] + [str(c) for c in gc["harm_classes"]]

        log_path = os.path.join(output_dir, f"z0_{ds_label}_{mode_tag}.log")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n  [z0/{ds_label}/{mode_tag}]")
        print(f"    Output: {run_output}")
        print(f"    Log: {log_path}")

        with open(log_path, "w") as log_file:
            p = subprocess.Popen(
                cmd, cwd=clf_dir,
                stdout=log_file, stderr=subprocess.STDOUT,
            )
            p.wait()
            status = "OK" if p.returncode == 0 else f"FAIL (rc={p.returncode})"
            print(f"    Status: {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitoring_dir", type=str, default=None,
                        help="Phase 1 monitoring results dir (skip monitoring analysis if None)")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_coco", action="store_true",
                        help="Skip COCO generation (run RAB test only)")
    parser.add_argument("--skip_rab", action="store_true",
                        help="Skip RAB generation (run COCO only)")
    parser.add_argument("--classifier", type=str, default="z0",
                        choices=["z0", "scg", "both"])
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.num_gpus))
    num_gpus = len(gpu_ids)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"{BASE_DIR}/rab_grid_search/results/gen_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("TARGETED GENERATION GRID SEARCH")
    print("=" * 80)
    print(f"GPUs: {gpu_ids} ({num_gpus})")
    print(f"Output: {output_dir}")

    # ── Phase 1 Analysis (optional) ──
    if args.monitoring_dir:
        print("\n--- Monitoring Analysis ---")
        threshold_analysis = analyze_monitoring(args.monitoring_dir)
        with open(os.path.join(output_dir, "threshold_analysis.json"), "w") as f:
            json.dump({str(k): v for k, v in threshold_analysis.items()}, f, indent=2)

    # ── Phase 2: Generation ──
    rab_test = f"{BASE_DIR}/rab_grid_search/data/ringabell_test.txt"
    coco = f"{BASE_DIR}/prompts/coco/coco_10k.txt"

    t0 = time.time()

    if args.classifier in ("z0", "both"):
        # Run z0 on RAB test
        if not args.skip_rab:
            print("\n--- Z0 Grid Search on RAB test ---")
            run_targeted_z0_grid(num_gpus, gpu_ids, output_dir, rab_test, "rab_test")

        # Run z0 on COCO (50 prompts for FP rate)
        if not args.skip_coco:
            # Create a COCO subset file (50 prompts)
            coco_subset = os.path.join(output_dir, "coco_50.txt")
            with open(coco) as f:
                lines = [l.strip() for l in f if l.strip()][:50]
            with open(coco_subset, "w") as f:
                f.write("\n".join(lines) + "\n")

            print("\n--- Z0 Grid Search on COCO (50 prompts) ---")
            run_targeted_z0_grid(num_gpus, gpu_ids, output_dir, coco_subset, "coco50")

    if args.classifier in ("scg", "both"):
        # SoftDelete+CG grid search
        print("\n--- SoftDelete+CG Grid Search ---")

        scg_config_base = {
            "ckpt_path": SD_MODEL,
            "classifier_ckpt": f"{BASE_DIR}/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth",
            "gradcam_stats_dir": f"{BASE_DIR}/SoftDelete+CG/gradcam_stats/nudity_4class",
            "nsamples": 1,
            "cfg_scale": 7.5,
            "num_inference_steps": 50,
            "seed": 1234,
        }

        import itertools

        guidance_scales = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
        spatial_threshold_starts = [0.3, 0.5, 0.7]
        spatial_threshold_ends = [0.2, 0.3, 0.5]
        threshold_strategies = ["constant", "linear_decrease", "cosine_anneal"]
        harmful_scales = [0.5, 1.0, 2.0]

        combos = []
        for gs, st_s, st_e, strat, hs in itertools.product(
            guidance_scales, spatial_threshold_starts, spatial_threshold_ends,
            threshold_strategies, harmful_scales,
        ):
            if strat in ("linear_decrease", "cosine_anneal") and st_e > st_s:
                continue
            if strat == "constant" and st_e != spatial_threshold_ends[0]:
                continue

            combos.append({
                "guidance_scale": gs,
                "spatial_threshold_start": st_s,
                "spatial_threshold_end": st_e,
                "threshold_strategy": strat,
                "harmful_scale": hs,
                "base_guidance_scale": 0.0,
            })

        # Dedup
        seen = set()
        unique = []
        for c in combos:
            k = json.dumps(c, sort_keys=True)
            if k not in seen:
                seen.add(k)
                unique.append(c)
        combos = unique

        for ds_label, prompt_file, max_p in [
            ("rab_test", rab_test, None),
            ("coco50", os.path.join(output_dir, "coco_50.txt"), 50),
        ]:
            if ds_label == "rab_test" and args.skip_rab:
                continue
            if ds_label == "coco50" and args.skip_coco:
                continue

            # Ensure coco subset exists
            if ds_label == "coco50" and not os.path.exists(prompt_file):
                with open(coco) as f:
                    lines = [l.strip() for l in f if l.strip()][:50]
                with open(prompt_file, "w") as f:
                    f.write("\n".join(lines) + "\n")

            scg_output = os.path.join(output_dir, f"scg_{ds_label}_{timestamp}")
            os.makedirs(scg_output, exist_ok=True)

            # Distribute across GPUs
            chunks = [[] for _ in range(num_gpus)]
            for i, combo in enumerate(combos):
                chunks[i % num_gpus].append(combo)

            processes = []
            for gpu_idx in range(num_gpus):
                if not chunks[gpu_idx]:
                    continue
                gpu_id = gpu_ids[gpu_idx]

                config = {
                    "gpu_id": gpu_id,
                    "experiments": chunks[gpu_idx],
                    "common": {
                        **scg_config_base,
                        "prompt_file": prompt_file,
                        "output_dir": os.path.join(scg_output, f"gpu{gpu_id}"),
                        "max_prompts": max_p,
                    },
                }

                config_path = os.path.join(scg_output, f"worker_gpu{gpu_id}.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                env = os.environ.copy()
                env.update(CLEAN_ENV_VARS)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                worker = os.path.join(os.path.dirname(__file__), "scg_grid_worker.py")
                log_path = os.path.join(scg_output, f"gpu{gpu_id}.log")

                print(f"    GPU {gpu_id}: {len(chunks[gpu_idx])} experiments for {ds_label}")

                with open(log_path, "w") as lf:
                    p = subprocess.Popen(
                        [PYTHON, worker, "--config", config_path],
                        env=env, stdout=lf, stderr=subprocess.STDOUT,
                    )
                    processes.append((gpu_id, p))

            for gpu_id, p in processes:
                p.wait()
                status = "OK" if p.returncode == 0 else f"FAIL"
                print(f"    GPU {gpu_id}: {status}")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
