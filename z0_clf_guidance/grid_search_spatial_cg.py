#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-GPU Grid Search: Z0 Classifier Guidance + Spatial CG (Gaussian CDF)

Combines z0_clf_guidance (Tweedie-based clean latent classifier guidance)
with spatial masking (gradcam mode):
  - Gaussian CDF spatial thresholding: per-pixel gradient magnitudes are
    transformed to CDF values under the harmful training distribution,
    making the spatial_threshold interpretable as a CDF percentile.
  - Spatial soft/binary masking
  - Constant guidance scale

Pre-requisite:
  Run compute_harmful_stats.py to generate harmful_stats.pt:
    python compute_harmful_stats.py \
        --classifier_ckpt ... --nudity_data_path ... --output_path harmful_stats.pt

Default benchmark: Ring-A-Bell (79 nudity prompts)

Architecture:
  - Orchestrator mode (default): generates combinations, spawns GPU workers
  - Worker mode (--worker): loads pipeline on assigned GPU, runs experiments
  - Single-GPU mode: runs inline without subprocess overhead

Default grid (with dedup):
  guidance_scales       x4  [5.0, 10.0, 15.0, 20.0]
  spatial_modes         x2  [none, gradcam]
  spatial_thresholds    x3  [0.3, 0.5, 0.7]     (CDF percentile, only for gradcam)
  spatial_soft          x2  [binary, soft]        (only for gradcam)
  -----------------------------------------------
  Total (deduplicated): 28 experiments

Usage:
    # Step 1: Compute harmful stats (one-time)
    python compute_harmful_stats.py \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --nudity_data_path /path/to/nude/images \
        --output_path ./harmful_stats.pt

    # Step 2: Dry run
    python grid_search_spatial_cg.py \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --harmful_stats_path ./harmful_stats.pt \
        --dry_run

    # Step 3: Run on 8 GPUs
    python grid_search_spatial_cg.py \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --harmful_stats_path ./harmful_stats.pt \
        --num_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7
"""

import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import numpy as np
import torch
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def load_prompts(prompt_file, csv_prompt_column="sensitive prompt",
                 csv_filter_column=None, csv_filter_value=None):
    """Load prompts from txt or csv file."""
    if prompt_file.endswith(".csv"):
        prompts = []
        with open(prompt_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if csv_filter_column and csv_filter_value:
                    cell = row.get(csv_filter_column, "")
                    if csv_filter_value not in cell:
                        continue
                prompt = row.get(
                    csv_prompt_column,
                    row.get("sensitive prompt", row.get("prompt", "")),
                )
                if prompt.strip():
                    prompts.append(prompt.strip())
        print(f"Loaded {len(prompts)} prompts from CSV: {prompt_file}")
    else:
        with open(prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from: {prompt_file}")
    return prompts


def save_image(image, filename, root):
    path = os.path.join(root, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((512, 512))
    image.save(path)


def make_tag(combo):
    """Create a filesystem-safe tag from experiment parameters."""
    parts = [f"gs{combo['guidance_scale']}"]
    parts.append(combo["spatial_mode"])
    if combo["spatial_mode"] != "none":
        parts.append(f"st{combo['spatial_threshold']}")
        parts.append("soft" if combo["spatial_soft"] else "bin")
        sched = combo.get("threshold_schedule", "constant")
        if sched != "constant":
            parts.append(sched)
    hr = combo.get("harm_ratio", 1.0)
    if hr != 1.0:
        parts.append(f"hr{hr}")
    return "_".join(parts)


# ═══════════════════════════════════════════════════════════════
# Combination generation with smart deduplication
# ═══════════════════════════════════════════════════════════════


def generate_combinations(args):
    """
    Generate all valid parameter combinations with deduplication.

    Dedup rules:
      - spatial_mode=none -> skip spatial_threshold / spatial_soft / schedule variations
      - harm_ratio only matters for safe_minus_harm mode
    """
    combos = []
    seen_tags = set()

    threshold_schedules = args.threshold_schedules
    harm_ratios = args.harm_ratios

    for gs, sm, st, ss, sched, hr in itertools.product(
        args.guidance_scales,
        args.spatial_modes,
        args.spatial_thresholds,
        args.spatial_soft_options,
        threshold_schedules,
        harm_ratios,
    ):
        # Dedup: skip threshold/soft/schedule when spatial_mode=none
        if sm == "none":
            if st != args.spatial_thresholds[0] or ss != 0:
                continue
            if sched != threshold_schedules[0]:
                continue

        # Dedup: cosine schedule is meaningless with soft mask (no threshold)
        if ss and sched != "constant":
            continue

        combo = {
            "guidance_scale": gs,
            "spatial_mode": sm,
            "spatial_threshold": st,
            "spatial_soft": bool(ss),
            "threshold_schedule": sched,
            "harm_ratio": hr,
        }

        tag = make_tag(combo)
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        combos.append(combo)

    return combos


# ═══════════════════════════════════════════════════════════════
# Pipeline callback
# ═══════════════════════════════════════════════════════════════


def guidance_callback(pipe, step, timestep, callback_kwargs,
                      guidance_model=None, guidance_scale=10.0,
                      target_class=1):
    """
    Pipeline callback: apply z0 classifier guidance at each step.
    Spatial masking (gradcam with Gaussian CDF thresholding) is handled
    internally by Z0GuidanceModel.guidance().
    """
    result = guidance_model.guidance(
        pipe, callback_kwargs, step, timestep,
        guidance_scale, target_class=target_class,
    )
    callback_kwargs["latents"] = result["latents"]

    # Logging (every 10 steps)
    monitor = result.get("differentiate_value", None)
    if monitor is not None and step % 10 == 0:
        val = monitor.mean().item()
        mask_ratio = result.get("spatial_mask_ratio", 1.0)
        gate_val = result.get("gate_val", 1.0)
        msg = f"    step={step}, t={timestep}, gs={guidance_scale:.1f}, monitor={val:.4f}"
        if mask_ratio < 1.0:
            msg += f", mask={mask_ratio:.2%}"
        if gate_val < 0.99:
            msg += f", gate={gate_val:.3f}"
        print(msg)

    return callback_kwargs


# ═══════════════════════════════════════════════════════════════
# Worker: run experiments on a single GPU
# ═══════════════════════════════════════════════════════════════


def run_worker(gpu_id, experiments, common_config, output_root):
    """
    Load SD pipeline ONCE on the assigned GPU, then run all experiments
    sequentially. This avoids the ~30s pipeline loading overhead per experiment.
    """
    from diffusers import DDIMScheduler
    from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
    from geo_utils.guidance_utils import Z0GuidanceModel

    device = torch.device("cuda")

    # ── Load pipeline ONCE ──
    print(f"[GPU {gpu_id}] Loading SD pipeline: {common_config['ckpt_path']}...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        common_config["ckpt_path"], safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # ── Load prompts ONCE ──
    prompts = load_prompts(
        common_config["prompt_file"],
        csv_prompt_column=common_config.get("csv_prompt_column", "sensitive prompt"),
        csv_filter_column=common_config.get("csv_filter_column"),
        csv_filter_value=common_config.get("csv_filter_value"),
    )

    num_steps = common_config["num_inference_steps"]
    n_exp = len(experiments)
    worker_t0 = time.time()

    for exp_idx, combo in enumerate(experiments):
        tag = make_tag(combo)
        output_dir = os.path.join(output_root, tag)

        # ── Skip if already done ──
        expected = len(prompts) * common_config["nsamples"]
        if os.path.exists(output_dir):
            existing = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
            if existing >= expected:
                print(f"[GPU {gpu_id}] SKIP ({exp_idx + 1}/{n_exp}): {tag}")
                continue

        print(f"\n{'=' * 60}")
        print(f"[GPU {gpu_id}] ({exp_idx + 1}/{n_exp}) {tag}")
        print(f"{'=' * 60}")

        # ── Reproducible seeds per experiment ──
        torch.manual_seed(common_config["seed"])
        np.random.seed(common_config["seed"])
        random.seed(common_config["seed"])

        # ── Build guidance model for this combo ──
        model_config = {
            "architecture": common_config.get("architecture", "resnet18"),
            "num_classes": common_config.get("num_classes", 3),
            "space": common_config.get("space", "latent"),
            "guidance_mode": common_config.get("guidance_mode", "target"),
            "safe_classes": common_config.get("safe_classes"),
            "harm_classes": common_config.get("harm_classes"),
            "spatial_guidance": combo["spatial_mode"] != "none",
            "spatial_mode": combo["spatial_mode"],
            "spatial_threshold": combo["spatial_threshold"],
            "spatial_soft": combo["spatial_soft"],
            "grad_wrt_z0": common_config.get("grad_wrt_z0", False),
            "harm_ratio": combo.get("harm_ratio", 1.0),
            "threshold_schedule": combo.get("threshold_schedule", "constant"),
            "harmful_keywords": common_config.get("harmful_keywords") or [],
            "attn_resolutions": common_config.get("attn_resolutions"),
            "prototype_path": common_config.get("prototype_path"),
            "eacg_harm_class": common_config.get("eacg_harm_class", 2),
            "eacg_safe_class": common_config.get("eacg_safe_class", 1),
            "eacg_tau": common_config.get("eacg_tau", 0.0),
            "eacg_kappa": common_config.get("eacg_kappa", 0.1),
            # Pass harmful stats path for Gaussian CDF spatial thresholding
            "harmful_stats_path": common_config.get("harmful_stats_path"),
            # GradCAM layer for spatial masking
            "gradcam_layer": common_config.get("gradcam_layer", "layer2"),
        }
        guidance_model = Z0GuidanceModel(
            pipe, common_config["classifier_ckpt"], model_config,
            target_class=common_config.get("target_class", 1),
            device=device,
        )

        os.makedirs(output_dir, exist_ok=True)

        # ── Save experiment config ──
        config_to_save = {
            **combo,
            "tag": tag,
            "classifier_ckpt": common_config["classifier_ckpt"],
            "space": common_config.get("space", "latent"),
            "guidance_mode": common_config.get("guidance_mode", "target"),
            "target_class": common_config.get("target_class", 1),
            "cfg_scale": common_config["cfg_scale"],
            "num_inference_steps": num_steps,
            "seed": common_config["seed"],
            "nsamples": common_config["nsamples"],
            "prompt_file": common_config["prompt_file"],
            "num_prompts": len(prompts),
            "harmful_stats_path": common_config.get("harmful_stats_path"),
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_to_save, f, indent=2)

        t0 = time.time()

        for idx, prompt in enumerate(prompts):
            print(f"  [{idx + 1}/{len(prompts)}] {prompt[:80]}...")

            guidance_model.set_prompt(prompt, pipe.tokenizer)

            cb = partial(
                guidance_callback,
                guidance_model=guidance_model,
                guidance_scale=combo["guidance_scale"],
                target_class=common_config.get("target_class", 1),
            )

            with torch.enable_grad():
                output = pipe(
                    prompt=prompt,
                    guidance_scale=common_config["cfg_scale"],
                    num_inference_steps=num_steps,
                    height=512, width=512,
                    callback_on_step_end=cb,
                    callback_on_step_end_tensor_inputs=[
                        "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
                    ],
                    num_images_per_prompt=common_config["nsamples"],
                )

            for si, image in enumerate(output.images):
                fname = f"prompt_{idx + 1:04d}_sample_{si + 1}.png"
                save_image(image, fname, root=output_dir)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s -> {output_dir}")

        # ── Cleanup attention processors for next experiment ──
        guidance_model.cleanup(pipe.unet)

    worker_elapsed = time.time() - worker_t0
    print(f"\n[GPU {gpu_id}] All {n_exp} experiments complete! "
          f"Total: {worker_elapsed:.1f}s ({worker_elapsed / 60:.1f}min)")


# ═══════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════


RINGABELL_CANDIDATES = [
    "/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt",
    "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv",
    "./prompts/ringabell.txt",
]


def resolve_prompt_file(prompt_file):
    """Resolve default prompt file to Ring-A-Bell."""
    if prompt_file is not None:
        path = os.path.abspath(prompt_file)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    for c in RINGABELL_CANDIDATES:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Ring-A-Bell prompt file not found. Specify --prompt_file.\n"
        f"Searched: {RINGABELL_CANDIDATES}"
    )


def parse_args():
    parser = ArgumentParser(
        description="Grid search: z0 CLF guidance + Gaussian CDF spatial thresholding (Ring-A-Bell)"
    )

    # ── Worker mode (internal, spawned by orchestrator) ──
    parser.add_argument("--worker", action="store_true",
                        help="Worker mode (internal, do not use directly)")
    parser.add_argument("--worker_config", type=str, default=None,
                        help="Path to worker config JSON (internal)")

    # ── Fixed parameters ──
    parser.add_argument("--ckpt_path", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        help="Stable Diffusion model path or HuggingFace ID")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Prompt file (default: auto-detect Ring-A-Bell)")
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Path to trained z0 classifier checkpoint (.pth)")
    parser.add_argument("--harmful_stats_path", type=str, default=None,
                        help="Path to harmful_stats.pt from compute_harmful_stats.py "
                             "(provides grad_mag_mu/sigma for CDF spatial thresholding)")
    parser.add_argument("--output_root", type=str,
                        default="./grid_search_output",
                        help="Root directory for grid search outputs")
    parser.add_argument("--architecture", type=str, default="resnet18",
                        choices=["resnet18", "vit_b"])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--space", type=str, default="latent",
                        choices=["latent", "image"])
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm", "paired"])
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for guidance (0=non-people, 1=clothed, 2=nude)")
    parser.add_argument("--safe_classes", type=int, nargs="+", default=None)
    parser.add_argument("--harm_classes", type=int, nargs="+", default=None)
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nsamples", type=int, default=1,
                        help="Number of samples per prompt")
    parser.add_argument("--grad_wrt_z0", action="store_true")
    parser.add_argument("--gradcam_layer", type=str, default="layer2",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="ResNet layer for GradCAM spatial mask")
    parser.add_argument("--csv_prompt_column", type=str, default="sensitive prompt")
    parser.add_argument("--csv_filter_column", type=str, default=None)
    parser.add_argument("--csv_filter_value", type=str, default=None)
    # Attention-based spatial guidance (optional)
    parser.add_argument("--harmful_keywords", type=str, nargs="+", default=None)
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=None)
    # EACG (optional)
    parser.add_argument("--prototype_path", type=str, default=None)
    parser.add_argument("--eacg_harm_class", type=int, default=2)
    parser.add_argument("--eacg_safe_class", type=int, default=1)
    parser.add_argument("--eacg_tau", type=float, default=0.0)
    parser.add_argument("--eacg_kappa", type=float, default=0.1)

    # ── Multi-GPU settings ──
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Specific GPU IDs (comma-separated, e.g. '0,1,2,3')")

    # ── Grid search sweep parameters ──
    parser.add_argument("--guidance_scales", type=float, nargs="+",
                        default=[5.0, 10.0, 15.0, 20.0],
                        help="Classifier guidance scales to sweep")
    parser.add_argument("--spatial_modes", type=str, nargs="+",
                        default=["none", "gradcam"],
                        help="Spatial modes to sweep")
    parser.add_argument("--spatial_thresholds", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7],
                        help="Spatial thresholds (CDF percentile, only for gradcam modes)")
    parser.add_argument("--spatial_soft_options", type=int, nargs="+",
                        default=[0, 1],
                        help="Spatial soft mask: 0=binary, 1=soft")
    parser.add_argument("--threshold_schedules", type=str, nargs="+",
                        default=["constant"],
                        help="Threshold schedule: constant, cosine")
    parser.add_argument("--harm_ratios", type=float, nargs="+",
                        default=[1.0],
                        help="Alpha for safe - alpha*harm (only for safe_minus_harm mode)")

    # ── Experiment range (for partial runs) ──
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start experiment index (for partial runs)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End experiment index (-1 for all)")

    parser.add_argument("--dry_run", action="store_true",
                        help="Print all experiments without running")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    # ────────── Worker mode ──────────
    if args.worker:
        if args.worker_config is None:
            print("ERROR: --worker_config required in worker mode", file=sys.stderr)
            sys.exit(1)
        with open(args.worker_config) as f:
            config = json.load(f)
        run_worker(
            gpu_id=config["gpu_id"],
            experiments=config["experiments"],
            common_config=config["common"],
            output_root=config["output_root"],
        )
        return

    # ────────── Orchestrator mode ──────────
    if args.classifier_ckpt is None:
        print("ERROR: --classifier_ckpt is required", file=sys.stderr)
        sys.exit(1)

    prompt_file = resolve_prompt_file(args.prompt_file)
    classifier_ckpt = os.path.abspath(args.classifier_ckpt)

    if not os.path.exists(classifier_ckpt):
        print(f"ERROR: Classifier checkpoint not found: {classifier_ckpt}",
              file=sys.stderr)
        sys.exit(1)

    # Validate harmful stats
    harmful_stats_path = None
    if args.harmful_stats_path:
        harmful_stats_path = os.path.abspath(args.harmful_stats_path)
        if not os.path.exists(harmful_stats_path):
            print(f"ERROR: harmful_stats not found: {harmful_stats_path}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        has_gradcam = any(m != "none" for m in args.spatial_modes)
        if has_gradcam:
            print("WARNING: --harmful_stats_path not specified but gradcam mode requested.\n"
                  "  Spatial thresholding will use max-normalization fallback.\n"
                  "  Run compute_harmful_stats.py first for CDF-based thresholding.")

    # ── Generate combinations ──
    combos = generate_combinations(args)
    total = len(combos)

    # Apply range filter
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else total
    combos = combos[start_idx:end_idx]
    n_exp = len(combos)

    # ── GPU setup ──
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))
    n_gpus = len(gpu_ids)

    # ── Output directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(
        os.path.abspath(args.output_root), f"grid_{timestamp}"
    )
    os.makedirs(output_root, exist_ok=True)

    # ── Print grid summary ──
    print("=" * 70)
    print("Z0 CLASSIFIER GUIDANCE - GAUSSIAN CDF SPATIAL GRID SEARCH")
    print("=" * 70)
    print(f"Prompt file:           {prompt_file}")
    print(f"Classifier:            {classifier_ckpt}")
    print(f"Harmful stats:         {harmful_stats_path or '(none - max-norm fallback)'}")
    print(f"SD model:              {args.ckpt_path}")
    print(f"Guidance mode:         {args.guidance_mode}")
    print(f"")
    print(f"guidance_scales:       {args.guidance_scales}")
    print(f"spatial_modes:         {args.spatial_modes}")
    print(f"spatial_thresholds:    {args.spatial_thresholds} (CDF percentile)")
    print(f"spatial_soft_options:  {args.spatial_soft_options}")
    print(f"threshold_schedules:  {args.threshold_schedules}")
    print(f"harm_ratios:          {args.harm_ratios}")
    print(f"")
    print(f"Total combinations:    {total} (deduplicated)")
    print(f"Running:               [{start_idx}, {end_idx}) = {n_exp} experiments")
    print(f"GPUs:                  {gpu_ids} ({n_gpus} GPUs)")
    print(f"Output:                {output_root}")
    print("=" * 70)

    # ── Dry run: just print experiments ──
    if args.dry_run:
        print("\n[DRY RUN] Experiments:\n")
        for i, combo in enumerate(combos):
            gpu = gpu_ids[i % n_gpus]
            tag = make_tag(combo)
            print(f"  [{start_idx + i:03d}] GPU {gpu}: {tag}")
        print(f"\nTotal: {n_exp} experiments on {n_gpus} GPUs")
        # Per-GPU counts
        for gi, gid in enumerate(gpu_ids):
            count = len([c for j, c in enumerate(combos) if j % n_gpus == gi])
            print(f"  GPU {gid}: {count} experiments")
        return

    # ── Common config (shared by all experiments) ──
    common_config = {
        "ckpt_path": args.ckpt_path,
        "prompt_file": prompt_file,
        "classifier_ckpt": classifier_ckpt,
        "harmful_stats_path": harmful_stats_path,
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "space": args.space,
        "guidance_mode": args.guidance_mode,
        "target_class": args.target_class,
        "safe_classes": args.safe_classes,
        "harm_classes": args.harm_classes,
        "cfg_scale": args.cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "grad_wrt_z0": args.grad_wrt_z0,
        "gradcam_layer": args.gradcam_layer,
        "csv_prompt_column": args.csv_prompt_column,
        "csv_filter_column": args.csv_filter_column,
        "csv_filter_value": args.csv_filter_value,
        "harmful_keywords": args.harmful_keywords,
        "attn_resolutions": args.attn_resolutions,
        "prototype_path": args.prototype_path,
        "eacg_harm_class": args.eacg_harm_class,
        "eacg_safe_class": args.eacg_safe_class,
        "eacg_tau": args.eacg_tau,
        "eacg_kappa": args.eacg_kappa,
    }

    # ── Save grid config ──
    grid_config = {
        "timestamp": timestamp,
        "total_combinations": total,
        "running": {"start": start_idx, "end": end_idx, "count": n_exp},
        "gpu_ids": gpu_ids,
        "sweep_params": {
            "guidance_scales": args.guidance_scales,
            "spatial_modes": args.spatial_modes,
            "spatial_thresholds": args.spatial_thresholds,
            "spatial_soft_options": args.spatial_soft_options,
            "threshold_schedules": args.threshold_schedules,
            "harm_ratios": args.harm_ratios,
        },
        "common_config": {k: v for k, v in common_config.items()
                          if not isinstance(v, (torch.Tensor,))},
        "experiments": [make_tag(c) for c in combos],
    }
    with open(os.path.join(output_root, "grid_config.json"), "w") as f:
        json.dump(grid_config, f, indent=2)

    total_t0 = time.time()

    # ── Single GPU: run inline (no subprocess overhead) ──
    if n_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        run_worker(gpu_ids[0], combos, common_config, output_root)
    else:
        # ── Multi-GPU: spawn one subprocess per GPU ──
        # Split experiments across GPUs (round-robin for balanced load)
        chunks = [[] for _ in range(n_gpus)]
        for i, combo in enumerate(combos):
            chunks[i % n_gpus].append(combo)

        # Write per-GPU config and spawn subprocess
        processes = []
        for gpu_idx, gpu_id in enumerate(gpu_ids):
            if not chunks[gpu_idx]:
                continue

            config_path = os.path.join(output_root, f"worker_gpu{gpu_id}.json")
            with open(config_path, "w") as f:
                json.dump({
                    "gpu_id": gpu_id,
                    "experiments": chunks[gpu_idx],
                    "common": common_config,
                    "output_root": output_root,
                }, f, indent=2)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            log_path = os.path.join(output_root, f"worker_gpu{gpu_id}.log")
            log_file = open(log_path, "w")

            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker",
                "--worker_config", config_path,
                "--classifier_ckpt", "dummy",  # satisfy argparse, actual ckpt from config
            ]

            print(f"Spawning worker on GPU {gpu_id} "
                  f"({len(chunks[gpu_idx])} experiments, log: {log_path})")
            p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
            processes.append((gpu_id, p, log_file))

        # Wait for all workers
        print(f"\nWaiting for {len(processes)} workers...")
        for gpu_id, p, log_file in processes:
            p.wait()
            log_file.close()
            status = "OK" if p.returncode == 0 else f"FAIL (rc={p.returncode})"
            print(f"  GPU {gpu_id}: {status}")

    total_elapsed = time.time() - total_t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("GRID SEARCH COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Experiments: {n_exp} on {n_gpus} GPUs")
    print(f"Total time:  {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    print(f"Results in:  {output_root}")
    print(f"{'=' * 70}")

    # Count completed experiments
    completed = 0
    for combo in combos:
        tag = make_tag(combo)
        exp_dir = os.path.join(output_root, tag)
        if os.path.exists(exp_dir):
            pngs = len([f for f in os.listdir(exp_dir) if f.endswith(".png")])
            if pngs > 0:
                completed += 1
    print(f"Completed: {completed}/{n_exp}")

    if completed < n_exp:
        print(f"\nTo resume incomplete experiments, re-run with the same output_root.")
        print(f"Completed experiments will be automatically skipped.")


if __name__ == "__main__":
    main()
