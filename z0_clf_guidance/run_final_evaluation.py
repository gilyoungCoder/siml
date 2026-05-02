#!/usr/bin/env python3
"""
Full evaluation pipeline:
1. Wait for grid search Qwen3 evaluation to complete
2. Find best setting (highest SR%)
3. Generate images on UnlearnDiff, MMA-Diffusion, COCO with best setting
4. Evaluate safety datasets with Qwen3-VL
5. Compute FID/CLIP on COCO
6. Print final results table

Usage:
    python run_final_evaluation.py --num_gpus 8
"""
import os
import sys
import json
import re
import csv
import time
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = "/mnt/home/yhgil99/unlearning/z0_clf_guidance"
GRID_DIR = f"{BASE_DIR}/grid_search_output/grid_20260216_063441"
OUTPUT_BASE = f"{BASE_DIR}/final_eval_output"

SDD_PYTHON = "/mnt/home/yhgil99/.conda/envs/sdd/bin/python"
VLM_PYTHON = "/mnt/home/yhgil99/.conda/envs/vlm/bin/python"

CLASSIFIER_CKPT = f"{BASE_DIR}/work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
HARMFUL_STATS = f"{BASE_DIR}/harmful_stats_layer1.pt"
SD_MODEL = "CompVis/stable-diffusion-v1-4"

# Datasets
DATASETS = {
    "unlearndiff": {
        "prompt_file": "/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv",
        "csv_prompt_column": "prompt",
        "num_prompts": 142,
    },
    "ringabell": {
        "prompt_file": "/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt",
        "csv_prompt_column": None,
        "num_prompts": 79,
    },
    "mma": {
        "prompt_file": "/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv",
        "csv_prompt_column": "adv_prompt",
        "num_prompts": 1000,
    },
    "coco": {
        "prompt_file": "/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k_10k.csv",
        "csv_prompt_column": "prompt",
        "num_prompts": 9998,
    },
}

COCO_REF_PATH = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/final_coco/sd_baseline"
COCO_PROMPTS_CSV = "/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k_10k.csv"

EVAL_SCRIPT = f"{BASE_DIR}/eval_grid_qwen3.py"
GENERATE_SCRIPT = f"{BASE_DIR}/generate.py"
COCO_METRICS_SCRIPT = "/mnt/home/yhgil99/unlearning/scripts/evaluate_coco_metrics.py"


def wait_for_grid_eval(grid_dir, total_expected=200, poll_interval=60):
    """Wait until all grid search experiments have Qwen3 results."""
    print(f"\n{'='*60}")
    print("Step 1: Waiting for grid search evaluation to complete...")
    print(f"{'='*60}")

    while True:
        count = 0
        for d in Path(grid_dir).iterdir():
            if not d.is_dir() or d.name.startswith("worker_"):
                continue
            if (d / "results_qwen3_vl_nudity.txt").exists():
                count += 1

        print(f"  [{time.strftime('%H:%M:%S')}] {count}/{total_expected} evaluations complete")

        if count >= total_expected:
            print("  All evaluations complete!")
            return
        time.sleep(poll_interval)


def find_best_setting(grid_dir):
    """Parse all results and find the setting with highest SR%."""
    print(f"\n{'='*60}")
    print("Step 2: Finding best grid search setting...")
    print(f"{'='*60}")

    results = []
    for d in sorted(Path(grid_dir).iterdir()):
        if not d.is_dir() or d.name.startswith("worker_") or d.name == "grid_config.json":
            continue
        txt = d / "results_qwen3_vl_nudity.txt"
        if not txt.exists():
            continue

        content = txt.read_text()
        safe_m = re.search(r'^Safe: (\d+)', content, re.M)
        partial_m = re.search(r'^Partial: (\d+)', content, re.M)
        full_m = re.search(r'^Full: (\d+)', content, re.M)
        notrel_m = re.search(r'^NotRel: (\d+)', content, re.M)
        total_m = re.search(r'Total images: (\d+)', content)

        safe = int(safe_m.group(1)) if safe_m else 0
        partial = int(partial_m.group(1)) if partial_m else 0
        full = int(full_m.group(1)) if full_m else 0
        notrel = int(notrel_m.group(1)) if notrel_m else 0
        total = int(total_m.group(1)) if total_m else (safe + partial + full + notrel)

        sr = (safe + partial) / total * 100 if total > 0 else 0

        results.append({
            "name": d.name,
            "sr": sr,
            "safe": safe,
            "partial": partial,
            "full": full,
            "notrel": notrel,
            "total": total,
        })

    # Sort by SR% descending, then by full ascending (fewer nude = better)
    results.sort(key=lambda x: (-x["sr"], x["full"]))

    # Print top 10
    print(f"\n  Top 10 settings by SR%:")
    print(f"  {'Rank':>4}  {'SR%':>6}  {'Safe':>5}  {'Part':>5}  {'Full':>5}  {'NRel':>5}  {'Experiment'}")
    print(f"  {'-'*80}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>4}  {r['sr']:>5.1f}%  {r['safe']:>5}  {r['partial']:>5}  "
              f"{r['full']:>5}  {r['notrel']:>5}  {r['name']}")

    best = results[0]
    print(f"\n  Best setting: {best['name']} (SR={best['sr']:.1f}%)")

    # Parse experiment name to extract hyperparameters
    config = parse_experiment_name(best["name"])
    print(f"  Parsed config: {json.dumps(config, indent=4)}")

    return best, config


def parse_experiment_name(name):
    """Parse experiment folder name back to hyperparameters.

    Examples:
        gs10.0_none_hr0.5
        gs15.0_gradcam_st0.5_bin_cosine_hr1.5
        gs20.0_gradcam_st0.3_soft_hr2.0
    """
    config = {
        "guidance_scale": 10.0,
        "spatial_mode": "none",
        "spatial_threshold": 0.3,
        "spatial_soft": False,
        "threshold_schedule": "constant",
        "harm_ratio": 1.0,
    }

    # guidance_scale
    gs_m = re.match(r'gs([\d.]+)_', name)
    if gs_m:
        config["guidance_scale"] = float(gs_m.group(1))

    # spatial_mode
    if "_gradcam_" in name or name.endswith("_gradcam"):
        config["spatial_mode"] = "gradcam"
    elif "_none" in name:
        config["spatial_mode"] = "none"

    # spatial_threshold
    st_m = re.search(r'_st([\d.]+)', name)
    if st_m:
        config["spatial_threshold"] = float(st_m.group(1))

    # spatial_soft
    if "_soft" in name:
        config["spatial_soft"] = True
    elif "_bin" in name:
        config["spatial_soft"] = False

    # threshold_schedule
    if "_cosine" in name:
        config["threshold_schedule"] = "cosine"

    # harm_ratio
    hr_m = re.search(r'_hr([\d.]+)', name)
    if hr_m:
        config["harm_ratio"] = float(hr_m.group(1))

    return config


def generate_images_distributed(dataset_name, dataset_cfg, gen_config, output_dir,
                                 gpu_ids, seed=42):
    """Generate images for a dataset, distributed across GPUs."""
    print(f"\n  Generating images for {dataset_name}...")
    print(f"  Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Check if already generated
    existing_pngs = list(Path(output_dir).glob("*.png"))
    if len(existing_pngs) >= dataset_cfg["num_prompts"]:
        print(f"  Already have {len(existing_pngs)} images, skipping generation.")
        return

    num_gpus = len(gpu_ids)
    prompt_file = dataset_cfg["prompt_file"]
    csv_col = dataset_cfg["csv_prompt_column"]

    # For single-GPU datasets (small prompt count), use 1 GPU
    # For large datasets, use all GPUs
    if dataset_cfg["num_prompts"] <= 200:
        # Single GPU is fine
        cmd = build_generate_cmd(
            prompt_file, csv_col, output_dir, gen_config, gpu_ids[0], seed,
        )
        print(f"  Running on GPU {gpu_ids[0]}...")
        run_cmd(cmd, f"{output_dir}/generate.log")
    else:
        # Distribute across GPUs by splitting prompt file
        split_and_generate(
            prompt_file, csv_col, output_dir, gen_config, gpu_ids, seed,
            dataset_cfg["num_prompts"],
        )


def build_generate_cmd(prompt_file, csv_col, output_dir, config, gpu_id, seed,
                        prompt_start=None, prompt_end=None, split_file=None):
    """Build the generate.py command string."""
    actual_prompt_file = split_file if split_file else prompt_file

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} {SDD_PYTHON} {GENERATE_SCRIPT} "
        f"{SD_MODEL} "
        f"--prompt_file {actual_prompt_file} "
        f"--output_dir {output_dir} "
        f"--nsamples 1 "
        f"--cfg_scale 7.5 "
        f"--num_inference_steps 50 "
        f"--seed {seed} "
        f"--classifier_ckpt {CLASSIFIER_CKPT} "
        f"--architecture resnet18 "
        f"--num_classes 3 "
        f"--space latent "
        f"--guidance_scale {config['guidance_scale']} "
        f"--guidance_start_step 1 "
        f"--target_class 1 "
        f"--guidance_mode safe_minus_harm "
        f"--safe_classes 1 "
        f"--harm_classes 2 "
        f"--spatial_mode {config['spatial_mode']} "
        f"--spatial_threshold {config['spatial_threshold']} "
        f"--gradcam_layer layer1 "
        f"--harmful_stats_path {HARMFUL_STATS} "
        f"--harm_ratio {config['harm_ratio']} "
        f"--threshold_schedule {config['threshold_schedule']} "
    )

    if config["spatial_soft"]:
        cmd += "--spatial_soft "

    if csv_col:
        cmd += f"--csv_prompt_column {csv_col} "

    return cmd


def split_and_generate(prompt_file, csv_col, output_dir, config, gpu_ids, seed, total_prompts):
    """Split prompts and generate in parallel across GPUs."""
    num_gpus = len(gpu_ids)

    # Read all prompts
    if prompt_file.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(prompt_file)
        prompts = df[csv_col].tolist()
    else:
        with open(prompt_file) as f:
            prompts = [l.strip() for l in f if l.strip()]

    # Split prompts into chunks
    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(prompts))
        if start < len(prompts):
            chunks.append(prompts[start:end])

    # Write split files and launch
    split_dir = os.path.join(output_dir, "_splits")
    os.makedirs(split_dir, exist_ok=True)

    procs = []
    for i, (gpu, chunk) in enumerate(zip(gpu_ids[:len(chunks)], chunks)):
        split_file = os.path.join(split_dir, f"split_{i}.txt")
        with open(split_file, "w") as f:
            for p in chunk:
                f.write(p + "\n")

        chunk_output = os.path.join(output_dir, f"_gpu{gpu}")
        os.makedirs(chunk_output, exist_ok=True)

        cmd = build_generate_cmd(
            prompt_file, None, chunk_output, config, gpu, seed,
            split_file=split_file,
        )
        log_file = os.path.join(output_dir, f"generate_gpu{gpu}.log")
        print(f"  GPU {gpu}: {len(chunk)} prompts -> {chunk_output}")

        with open(log_file, "w") as lf:
            p = subprocess.Popen(
                cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
            )
        procs.append((gpu, p, chunk_output))

    # Wait for all
    for gpu, p, chunk_output in procs:
        p.wait()
        if p.returncode != 0:
            print(f"  WARNING: GPU {gpu} returned code {p.returncode}")

    # Merge all chunk outputs into main output dir
    idx = 1
    for gpu, p, chunk_output in procs:
        for img in sorted(Path(chunk_output).glob("*.png")):
            dst = os.path.join(output_dir, f"prompt_{idx:04d}_sample_1.png")
            os.rename(str(img), dst)
            idx += 1

    # Cleanup
    for gpu, p, chunk_output in procs:
        try:
            os.rmdir(chunk_output)
        except OSError:
            pass

    print(f"  Total generated: {idx - 1} images")


def run_cmd(cmd, log_file=None):
    """Run a command, optionally logging output."""
    if log_file:
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
            p.wait()
            return p.returncode
    else:
        return subprocess.call(cmd, shell=True)


def evaluate_qwen3(img_dirs, gpu_ids):
    """Run Qwen3-VL evaluation on multiple image directories using multiple GPUs."""
    print(f"\n  Evaluating {len(img_dirs)} directories with Qwen3-VL...")

    # Filter dirs that need evaluation
    to_eval = []
    for d in img_dirs:
        if not os.path.exists(os.path.join(d, "results_qwen3_vl_nudity.txt")):
            to_eval.append(d)
        else:
            print(f"  [SKIP] {os.path.basename(d)} - already evaluated")

    if not to_eval:
        print("  All directories already evaluated!")
        return

    # Create a temporary base dir structure for the eval script
    eval_base = os.path.join(OUTPUT_BASE, "_qwen3_eval_tmp")
    os.makedirs(eval_base, exist_ok=True)

    # Symlink dirs into eval_base
    for d in to_eval:
        link = os.path.join(eval_base, os.path.basename(d))
        if os.path.exists(link):
            os.unlink(link)
        os.symlink(os.path.abspath(d), link)

    # Launch evaluation
    cmd = (
        f"{VLM_PYTHON} {EVAL_SCRIPT} "
        f"--base_dir {eval_base} "
        f"--launch "
        f"--gpu_ids {','.join(str(g) for g in gpu_ids)}"
    )
    log_file = os.path.join(OUTPUT_BASE, "qwen3_eval.log")
    print(f"  Log: {log_file}")

    p = subprocess.Popen(cmd, shell=True,
                         stdout=open(log_file, "w"),
                         stderr=subprocess.STDOUT)
    p.wait()

    if p.returncode != 0:
        print(f"  WARNING: Qwen3 eval returned code {p.returncode}")
    else:
        print("  Qwen3-VL evaluation complete!")


def evaluate_coco_metrics(coco_dir, gpu_id=0):
    """Compute FID and CLIP score for COCO-generated images."""
    print(f"\n  Computing FID and CLIP for COCO images...")

    metrics_file = os.path.join(coco_dir, "eval_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"  Already computed: FID={metrics.get('fid', 'N/A')}, CLIP={metrics.get('clip_score', 'N/A')}")
        return metrics

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} {SDD_PYTHON} {COCO_METRICS_SCRIPT} "
        f"--img_dirs {coco_dir} "
        f"--prompts_path {COCO_PROMPTS_CSV} "
        f"--ref_path {COCO_REF_PATH} "
        f"--metrics all "
        f"--device cuda:0 "
        f"--batch_size 64"
    )
    log_file = os.path.join(coco_dir, "eval_metrics.log")
    print(f"  Log: {log_file}")
    run_cmd(cmd, log_file)

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"  FID={metrics.get('fid', 'N/A'):.2f}, CLIP={metrics.get('clip_score', 'N/A'):.4f}")
        return metrics
    else:
        print("  WARNING: eval_metrics.json not created!")
        return {}


def parse_qwen3_results(img_dir):
    """Parse Qwen3-VL results from a directory."""
    txt = os.path.join(img_dir, "results_qwen3_vl_nudity.txt")
    if not os.path.exists(txt):
        return None

    content = open(txt).read()
    safe_m = re.search(r'^Safe: (\d+)', content, re.M)
    partial_m = re.search(r'^Partial: (\d+)', content, re.M)
    full_m = re.search(r'^Full: (\d+)', content, re.M)
    notrel_m = re.search(r'^NotRel: (\d+)', content, re.M)
    total_m = re.search(r'Total images: (\d+)', content)

    safe = int(safe_m.group(1)) if safe_m else 0
    partial = int(partial_m.group(1)) if partial_m else 0
    full = int(full_m.group(1)) if full_m else 0
    notrel = int(notrel_m.group(1)) if notrel_m else 0
    total = int(total_m.group(1)) if total_m else (safe + partial + full + notrel)

    sr = (safe + partial) / total if total > 0 else 0
    return {
        "sr": sr,
        "safe": safe / total if total > 0 else 0,
        "partial": partial / total if total > 0 else 0,
        "full": full / total if total > 0 else 0,
        "notrel": notrel / total if total > 0 else 0,
        "safe_n": safe, "partial_n": partial, "full_n": full, "notrel_n": notrel,
        "total": total,
    }


def print_final_table(results):
    """Print the final results table."""
    print(f"\n\n{'='*120}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*120}")

    # Header
    datasets = ["unlearndiff", "ringabell", "mma"]
    print(f"\n{'':>20}", end="")
    for ds in datasets:
        print(f"  {'|':>1} {ds:^42}", end="")
    print(f"  | {'COCO':^16}")

    print(f"{'Method':>20}", end="")
    for ds in datasets:
        print(f"  | {'SR':>5} {'Safe':>5} {'Part':>5} {'Full':>5} {'NRel':>5}", end="")
    print(f"  | {'FID↓':>6} {'CLIP↑':>6}")

    print(f"{'-'*120}")

    # Our method
    row = results
    print(f"{'Ours (z0-CG)':>20}", end="")
    for ds in datasets:
        r = row.get(ds)
        if r:
            print(f"  | {r['sr']:>5.3f} {r['safe']:>5.3f} {r['partial']:>5.3f} "
                  f"{r['full']:>5.3f} {r['notrel']:>5.3f}", end="")
        else:
            print(f"  | {'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5}", end="")

    coco = row.get("coco_metrics", {})
    fid = coco.get("fid", -1)
    clip = coco.get("clip_score", -1)
    fid_str = f"{fid:.2f}" if fid > 0 else "N/A"
    clip_str = f"{clip:.3f}" if clip > 0 else "N/A"
    print(f"  | {fid_str:>6} {clip_str:>6}")

    print(f"{'='*120}")

    # Save to JSON
    output_json = os.path.join(OUTPUT_BASE, "final_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_wait", action="store_true",
                        help="Skip waiting for grid eval (use if already done)")
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip generation (use existing images)")
    parser.add_argument("--skip_qwen3", action="store_true",
                        help="Skip Qwen3 evaluation (use existing results)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ================================================================
    # Step 1: Wait for grid search evaluation
    # ================================================================
    if not args.skip_wait:
        wait_for_grid_eval(GRID_DIR, total_expected=200, poll_interval=30)

    # ================================================================
    # Step 2: Find best setting
    # ================================================================
    best, config = find_best_setting(GRID_DIR)

    # Save best config
    with open(os.path.join(OUTPUT_BASE, "best_config.json"), "w") as f:
        json.dump({"best_experiment": best, "config": config}, f, indent=2)

    # ================================================================
    # Step 3: Generate images for all datasets
    # ================================================================
    if not args.skip_generate:
        print(f"\n{'='*60}")
        print("Step 3: Generating images with best setting...")
        print(f"{'='*60}")

        # UnlearnDiff (142 prompts - 1 GPU sufficient)
        ud_dir = os.path.join(OUTPUT_BASE, "unlearndiff")
        generate_images_distributed(
            "UnlearnDiff", DATASETS["unlearndiff"], config, ud_dir,
            gpu_ids[:1], args.seed,
        )

        # Ring-A-Bell: copy from grid search (already generated)
        rb_dir = os.path.join(OUTPUT_BASE, "ringabell")
        rb_grid_dir = os.path.join(GRID_DIR, best["name"])
        if not os.path.exists(rb_dir):
            os.symlink(os.path.abspath(rb_grid_dir), rb_dir)
            print(f"\n  Ring-A-Bell: symlinked from grid search -> {rb_grid_dir}")
        else:
            print(f"\n  Ring-A-Bell: already exists at {rb_dir}")

        # MMA-Diffusion (1000 prompts - use all GPUs)
        mma_dir = os.path.join(OUTPUT_BASE, "mma")
        generate_images_distributed(
            "MMA-Diffusion", DATASETS["mma"], config, mma_dir,
            gpu_ids, args.seed,
        )

        # COCO (10k prompts - use all GPUs)
        coco_dir = os.path.join(OUTPUT_BASE, "coco")
        generate_images_distributed(
            "COCO", DATASETS["coco"], config, coco_dir,
            gpu_ids, args.seed,
        )

    # ================================================================
    # Step 4: Qwen3-VL evaluation on safety datasets
    # ================================================================
    if not args.skip_qwen3:
        print(f"\n{'='*60}")
        print("Step 4: Qwen3-VL evaluation on safety datasets...")
        print(f"{'='*60}")

        safety_dirs = [
            os.path.join(OUTPUT_BASE, "unlearndiff"),
            os.path.join(OUTPUT_BASE, "ringabell"),
            os.path.join(OUTPUT_BASE, "mma"),
        ]
        evaluate_qwen3(safety_dirs, gpu_ids)

    # ================================================================
    # Step 5: FID/CLIP on COCO
    # ================================================================
    print(f"\n{'='*60}")
    print("Step 5: Computing COCO metrics (FID, CLIP)...")
    print(f"{'='*60}")

    coco_dir = os.path.join(OUTPUT_BASE, "coco")
    coco_metrics = evaluate_coco_metrics(coco_dir, gpu_id=gpu_ids[0])

    # ================================================================
    # Step 6: Aggregate results
    # ================================================================
    print(f"\n{'='*60}")
    print("Step 6: Aggregating final results...")
    print(f"{'='*60}")

    final_results = {
        "best_setting": best["name"],
        "config": config,
    }

    for ds_name in ["unlearndiff", "ringabell", "mma"]:
        ds_dir = os.path.join(OUTPUT_BASE, ds_name)
        r = parse_qwen3_results(ds_dir)
        if r:
            final_results[ds_name] = r
            print(f"  {ds_name}: SR={r['sr']:.3f} Safe={r['safe']:.3f} "
                  f"Partial={r['partial']:.3f} Full={r['full']:.3f} NotRel={r['notrel']:.3f}")
        else:
            print(f"  {ds_name}: No results found!")

    final_results["coco_metrics"] = coco_metrics

    print_final_table(final_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
