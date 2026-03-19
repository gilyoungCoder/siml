#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare multiple monitoring metrics for FP/TP gap analysis.

Runs denoising once per prompt, computes ALL monitoring metrics simultaneously:
  1. softmax_p:     softmax(classifier(z0))[nude]
  2. logit_raw:     classifier(z0)[nude]  (raw logit)
  3. logit_cdf:     Gaussian CDF of logit, normalized by harmful logit stats
  4. gcam_mean_cdf: Gaussian CDF of mean(GradCAM), sample-level stats
  5. gcam_top20:    mean of top 20% GradCAM pixels
  6. gcam_top20_cdf: CDF of gcam_top20, normalized by harmful top-pixel stats
  7. gcam_max:      max GradCAM pixel value
  8. combined:      softmax_p * gcam_mean_cdf  (product)

Reports per-threshold trigger rates for each metric, finds the metric
with the best FP/TP gap (= TP - FP).

Usage:
    # Multi-GPU worker
    CUDA_VISIBLE_DEVICES=$i python test_monitoring_compare.py \
        --prompt_file coco --gpu_id $i --num_gpus 8

    # Merge
    python test_monitoring_compare.py --prompt_file coco --merge
"""

import csv
import json
import math
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import DDIMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from models.latent_classifier import LatentResNet18Classifier
from utils.denoise_utils import predict_z0


PROMPT_CONFIGS = {
    "coco": {
        "path": "/mnt/home/yhgil99/unlearning/prompts/coco/coco_10k.txt",
        "type": "txt",
        "max_prompts": 50,
        "label": "benign",
    },
    "ringabell": {
        "path": "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv",
        "type": "csv",
        "max_prompts": None,
        "label": "harmful",
    },
    "unlearndiff": {
        "path": "/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv",
        "type": "csv",
        "max_prompts": None,
        "label": "harmful",
    },
}

METRIC_NAMES = [
    "softmax_p",
    "logit_cdf",
    "gcam_mean_cdf",
    "gcam_top20_cdf",
    "gcam_max",
    "combined_softmax_gcam",
]

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_prompts(name_or_path, max_prompts=None):
    if name_or_path in PROMPT_CONFIGS:
        cfg = PROMPT_CONFIGS[name_or_path]
        path, ftype = cfg["path"], cfg["type"]
        if max_prompts is None:
            max_prompts = cfg["max_prompts"]
    else:
        path = name_or_path
        ftype = "csv" if path.endswith(".csv") else "txt"

    if ftype == "csv":
        prompts = []
        with open(path) as f:
            reader = csv.DictReader(f)
            col = "prompt" if "prompt" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                prompts.append(row[col].strip())
    else:
        with open(path) as f:
            prompts = [l.strip() for l in f if l.strip()]

    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def gaussian_cdf(x, mu, sigma):
    z = (x - mu) / (sigma + 1e-8)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_trigger_stats(max_values, thresholds):
    """Given list of max values per prompt, compute per-threshold trigger rate."""
    n = len(max_values)
    stats = {}
    for thr in thresholds:
        triggered = sum(1 for v in max_values if v > thr)
        stats[f"{thr:.2f}"] = {
            "threshold": thr,
            "rate": triggered / n if n > 0 else 0,
            "count": triggered,
            "total": n,
        }
    return stats


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth")
    parser.add_argument("--harmful_stats_path", type=str, default="./harmful_stats.pt")
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--gradcam_layer", type=str, default="layer2")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./monitoring_compare")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompt_name = args.prompt_file if args.prompt_file in PROMPT_CONFIGS else Path(args.prompt_file).stem
    label = PROMPT_CONFIGS.get(args.prompt_file, {}).get("label", "unknown")

    # === MERGE MODE ===
    if args.merge:
        all_results = []
        for f in sorted(Path(args.output_dir).glob(f"{prompt_name}_gpu*.json")):
            with open(f) as fp:
                data = json.load(fp)
                all_results.extend(data["results"])

        all_results.sort(key=lambda x: x["idx"])

        if not all_results:
            print(f"No results for {prompt_name}")
            return

        n = len(all_results)
        print(f"\n{'='*80}")
        print(f"MONITORING COMPARISON: {prompt_name} ({label}, N={n})")
        print(f"{'='*80}")

        # Per-metric analysis
        metric_stats = {}
        for metric in METRIC_NAMES:
            max_vals = [r["max_metrics"][metric] for r in all_results]
            trigger = compute_trigger_stats(max_vals, THRESHOLDS)
            metric_stats[metric] = {
                "trigger": trigger,
                "distribution": {
                    "min": float(np.min(max_vals)),
                    "p25": float(np.percentile(max_vals, 25)),
                    "median": float(np.percentile(max_vals, 50)),
                    "p75": float(np.percentile(max_vals, 75)),
                    "p90": float(np.percentile(max_vals, 90)),
                    "p95": float(np.percentile(max_vals, 95)),
                    "max": float(np.max(max_vals)),
                },
            }

            dist = metric_stats[metric]["distribution"]
            print(f"\n--- {metric} ---")
            print(f"  min={dist['min']:.4f}  p25={dist['p25']:.4f}  "
                  f"median={dist['median']:.4f}  p75={dist['p75']:.4f}  "
                  f"p90={dist['p90']:.4f}  max={dist['max']:.4f}")
            print(f"  {'Thr':>6} | {'Rate':>8}")
            print(f"  {'-'*6}-+-{'-'*8}")
            for thr in THRESHOLDS:
                s = trigger[f"{thr:.2f}"]
                print(f"  {thr:>6.2f} | {s['rate']:>7.1%}")

        # Save merged
        out_path = os.path.join(args.output_dir, f"{prompt_name}_merged.json")
        with open(out_path, "w") as f:
            json.dump({
                "prompt_name": prompt_name,
                "label": label,
                "n": n,
                "metric_stats": metric_stats,
                "results": all_results,
            }, f, indent=2)
        print(f"\nSaved: {out_path}")
        return

    # === RUN MODE ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_prompts = load_prompts(args.prompt_file, args.max_prompts)
    total = len(all_prompts)
    chunk = (total + args.num_gpus - 1) // args.num_gpus
    start = args.gpu_id * chunk
    end = min(start + chunk, total)
    prompts = all_prompts[start:end]

    print(f"GPU {args.gpu_id}/{args.num_gpus}: [{start}:{end}] ({len(prompts)}/{total})")

    # Load harmful stats
    hs = torch.load(args.harmful_stats_path, map_location="cpu")
    logit_mu = hs["logit_mu"]
    logit_sigma = hs["logit_sigma"]
    sample_mu = hs.get("sample_level_mu", hs["gradcam_full_mu"])
    sample_sigma = hs.get("sample_level_sigma", hs["gradcam_full_sigma"])
    gcam_mu = hs["gradcam_mu"]  # top 30% pixel stats
    gcam_sigma = hs["gradcam_sigma"]
    harm_class = hs.get("harmful_class", 2)
    top_frac = hs.get("top_fraction", 0.3)

    print(f"Stats: logit mu={logit_mu:.3f} s={logit_sigma:.3f} | "
          f"sample mu={sample_mu:.3f} s={sample_sigma:.3f} | "
          f"gcam_top mu={gcam_mu:.3f} s={gcam_sigma:.3f}")

    # Load classifier
    classifier = LatentResNet18Classifier(
        num_classes=args.num_classes, pretrained_backbone=False
    ).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()

    # Load pipeline
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    results = []

    for idx, prompt in enumerate(tqdm(prompts, desc=f"GPU{args.gpu_id} {prompt_name}")):
        set_seed(args.seed + start + idx)
        step_metrics = []

        def callback_on_step_end(pipe_ref, step, timestep, callback_kwargs):
            prev_latents = callback_kwargs["prev_latents"]
            noise_pred_uncond = callback_kwargs.get("noise_pred_uncond",
                                                     callback_kwargs["noise_pred"])

            alpha_bar = pipe_ref.scheduler.alphas_cumprod.to(device)[timestep]
            alpha_bar_4d = alpha_bar.view(-1, 1, 1, 1)
            z0_hat = predict_z0(prev_latents, noise_pred_uncond.detach(), alpha_bar_4d)

            # 1. Classifier softmax & logit
            with torch.no_grad():
                logits = classifier(z0_hat)
                probs = F.softmax(logits, dim=-1)
                softmax_p = probs[:, harm_class].mean().item()
                logit_raw = logits[:, harm_class].mean().item()

            logit_cdf_val = gaussian_cdf(logit_raw, logit_mu, logit_sigma)

            # 2. GradCAM
            with torch.enable_grad():
                gradcam_map = classifier.compute_gradcam(
                    z0_hat.detach(), target_class=harm_class,
                    layer_name=args.gradcam_layer
                )  # (B, 1, H, W)

            # 2a. Sample-level mean
            gcam_mean = gradcam_map.mean().item()
            gcam_mean_cdf_val = gaussian_cdf(gcam_mean, sample_mu, sample_sigma)

            # 2b. Top-20% pixel mean
            flat = gradcam_map.flatten()
            k = max(1, int(flat.numel() * 0.2))
            topk_vals, _ = flat.topk(k)
            gcam_top20 = topk_vals.mean().item()
            gcam_top20_cdf_val = gaussian_cdf(gcam_top20, gcam_mu, gcam_sigma)

            # 2c. Max pixel
            gcam_max_val = gradcam_map.max().item()

            # 3. Combined
            combined = softmax_p * gcam_mean_cdf_val

            metrics = {
                "softmax_p": softmax_p,
                "logit_cdf": logit_cdf_val,
                "gcam_mean_cdf": gcam_mean_cdf_val,
                "gcam_top20_cdf": gcam_top20_cdf_val,
                "gcam_max": gcam_max_val,
                "combined_softmax_gcam": combined,
            }
            step_metrics.append(metrics)
            return callback_kwargs

        with torch.no_grad():
            pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=[
                    "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
                ],
                height=512, width=512,
            )

        # Max per metric across all steps
        max_metrics = {}
        for m in METRIC_NAMES:
            vals = [s[m] for s in step_metrics]
            max_metrics[m] = max(vals) if vals else 0

        results.append({
            "idx": start + idx,
            "prompt": prompt[:100],
            "max_metrics": max_metrics,
            "step_metrics": {m: [s[m] for s in step_metrics] for m in METRIC_NAMES},
        })

        if idx % 5 == 0:
            print(f"  [{start+idx:03d}] sfx={max_metrics['softmax_p']:.3f} "
                  f"logit_cdf={max_metrics['logit_cdf']:.3f} "
                  f"gcam_cdf={max_metrics['gcam_mean_cdf']:.3f} "
                  f"top20_cdf={max_metrics['gcam_top20_cdf']:.3f} "
                  f"| {prompt[:50]}")

    out_path = os.path.join(args.output_dir, f"{prompt_name}_gpu{args.gpu_id}.json")
    with open(out_path, "w") as f:
        json.dump({
            "gpu_id": args.gpu_id,
            "prompt_name": prompt_name,
            "label": label,
            "start": start,
            "end": end,
            "results": results,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
