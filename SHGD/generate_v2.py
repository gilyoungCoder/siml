"""
SHGD Generation Script (v2)

v2 features: negative concept amplification, sample-adaptive guidance,
multi-round guide-heal, progressive guidance schedule, cross-attention suppression,
concept negation.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from diffusers import DDPMScheduler

from pipeline_shgd_v2 import SHGDPipeline


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(prompt_file):
    """Load prompts from txt or csv."""
    prompts = []
    path = Path(prompt_file)

    if path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            col = None
            for candidate in ["sensitive prompt", "prompt", "adv_prompt"]:
                if candidate in reader.fieldnames:
                    col = candidate
                    break
            if col is None:
                raise ValueError(f"No prompt column found in {prompt_file}")
            for row in reader:
                prompts.append(row[col].strip().strip('"'))
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)

    return prompts


def dummy(images, **kwargs):
    return images, [False] * len(images)


def main():
    parser = argparse.ArgumentParser(description="SHGD Generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to prompt file (.txt or .csv)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    # Multi-GPU splitting
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    # Override config params from CLI
    parser.add_argument("--anchor_guidance_scale", type=float, default=None)
    parser.add_argument("--guide_start_frac", type=float, default=None)
    parser.add_argument("--guide_end_frac", type=float, default=None)
    parser.add_argument("--heal_strength", type=float, default=None)
    parser.add_argument("--consistency_threshold", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--micro_heal", action="store_true", default=None)
    parser.add_argument("--freq_selective", action="store_true", default=None)
    # v2 features
    parser.add_argument("--negative_concept_scale", type=float, default=None)
    parser.add_argument("--enable_sample_adaptive", action="store_true", default=None)
    parser.add_argument("--adaptive_guidance_min", type=float, default=None)
    parser.add_argument("--adaptive_guidance_max", type=float, default=None)
    parser.add_argument("--num_guide_heal_rounds", type=int, default=None)
    parser.add_argument("--round_guidance_decay", type=float, default=None)
    parser.add_argument("--guidance_schedule", type=str, default=None,
                        choices=["constant", "cosine", "linear_decay", "warmup_decay", "bell"])
    parser.add_argument("--enable_attn_suppression", action="store_true", default=None)
    parser.add_argument("--attn_suppress_scale", type=float, default=None)
    parser.add_argument("--enable_concept_negation", action="store_true", default=None)
    parser.add_argument("--concept_negation_scale", type=float, default=None)
    # Skip evaluation (for COCO)
    parser.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_cfg = config["model"]
    shgd_cfg = config["shgd"]
    eval_cfg = config.get("eval", {})

    # CLI overrides
    if args.anchor_guidance_scale is not None:
        shgd_cfg["anchor_guidance_scale"] = args.anchor_guidance_scale
    if args.guide_start_frac is not None:
        shgd_cfg["guide_start_frac"] = args.guide_start_frac
    if args.guide_end_frac is not None:
        shgd_cfg["guide_end_frac"] = args.guide_end_frac
    if args.heal_strength is not None:
        shgd_cfg["heal_strength"] = args.heal_strength
    if args.consistency_threshold is not None:
        shgd_cfg["consistency_threshold"] = args.consistency_threshold
    if args.guidance_scale is not None:
        model_cfg["guidance_scale"] = args.guidance_scale
    if args.micro_heal is not None:
        shgd_cfg["micro_heal"] = args.micro_heal
    if args.freq_selective is not None:
        shgd_cfg["freq_selective"] = args.freq_selective
    # v2 overrides
    if args.negative_concept_scale is not None:
        shgd_cfg["negative_concept_scale"] = args.negative_concept_scale
    if args.enable_sample_adaptive is not None:
        shgd_cfg["enable_sample_adaptive"] = args.enable_sample_adaptive
    if args.adaptive_guidance_min is not None:
        shgd_cfg["adaptive_guidance_min"] = args.adaptive_guidance_min
    if args.adaptive_guidance_max is not None:
        shgd_cfg["adaptive_guidance_max"] = args.adaptive_guidance_max
    if args.num_guide_heal_rounds is not None:
        shgd_cfg["num_guide_heal_rounds"] = args.num_guide_heal_rounds
    if args.round_guidance_decay is not None:
        shgd_cfg["round_guidance_decay"] = args.round_guidance_decay
    if args.guidance_schedule is not None:
        shgd_cfg["guidance_schedule"] = args.guidance_schedule
    if args.enable_attn_suppression is not None:
        shgd_cfg["enable_attn_suppression"] = args.enable_attn_suppression
    if args.attn_suppress_scale is not None:
        shgd_cfg["attn_suppress_scale"] = args.attn_suppress_scale
    if args.enable_concept_negation is not None:
        shgd_cfg["enable_concept_negation"] = args.enable_concept_negation
    if args.concept_negation_scale is not None:
        shgd_cfg["concept_negation_scale"] = args.concept_negation_scale

    # Load prompts and split for multi-GPU
    prompts = load_prompts(args.prompt_file)
    if args.total_gpus > 1:
        chunk_size = len(prompts) // args.total_gpus
        start = args.gpu_id * chunk_size
        end = start + chunk_size if args.gpu_id < args.total_gpus - 1 else len(prompts)
        prompts = prompts[start:end]
        print(f"[GPU {args.gpu_id}] Processing prompts {start}-{end-1} ({len(prompts)} total)")

    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    safe_dir = os.path.join(args.save_dir, "safe")
    unsafe_dir = os.path.join(args.save_dir, "unsafe")
    all_dir = os.path.join(args.save_dir, "all")
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "config": config}, f, indent=2)

    # Load model
    device = args.device
    print(f"Loading model: {model_cfg['model_id']}")
    scheduler = DDPMScheduler.from_pretrained(
        model_cfg["model_id"], subfolder="scheduler"
    )
    pipe = SHGDPipeline.from_pretrained(
        model_cfg["model_id"],
        scheduler=scheduler,
        torch_dtype=torch.float32,
        revision="fp16",
    )
    pipe.safety_checker = dummy
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.eval()

    gen = torch.Generator(device=device)

    # Load NudeNet evaluator (optional)
    nudenet = None
    if not args.skip_eval:
        try:
            from nudenet_eval import load_nudenet
            nudenet_path = eval_cfg.get("nudenet_path")
            if nudenet_path:
                # Resolve relative path
                config_dir = os.path.dirname(os.path.abspath(args.config))
                if not os.path.isabs(nudenet_path):
                    nudenet_path = os.path.normpath(
                        os.path.join(config_dir, nudenet_path)
                    )
                nudenet = load_nudenet(nudenet_path)
                print(f"NudeNet loaded from {nudenet_path}")
        except Exception as e:
            print(f"NudeNet eval not available ({e}), saving all images to 'all/'")

    # Generate
    safe_cnt, unsafe_cnt = 0, 0
    results = []
    total_time = 0

    harmful_concepts = config.get("harmful_concepts")
    anchor_concepts = config.get("anchor_concepts")

    for idx, prompt in enumerate(prompts):
        global_idx = idx + (args.gpu_id * (len(prompts)) if args.total_gpus > 1 else 0)
        seed = args.seed + global_idx

        print(f"[{idx+1}/{len(prompts)}] Generating: {prompt[:80]}...")
        start_time = time.time()

        images = pipe(
            prompt,
            num_images_per_prompt=args.num_samples,
            guidance_scale=model_cfg["guidance_scale"],
            num_inference_steps=model_cfg["num_inference_steps"],
            height=model_cfg["image_size"],
            width=model_cfg["image_size"],
            generator=gen.manual_seed(seed),
            # SHGD params
            harmful_concepts=harmful_concepts,
            anchor_concepts=anchor_concepts,
            anchor_guidance_scale=shgd_cfg["anchor_guidance_scale"],
            guide_start_frac=shgd_cfg["guide_start_frac"],
            guide_end_frac=shgd_cfg["guide_end_frac"],
            heal_strength=shgd_cfg["heal_strength"],
            enable_self_consistency=shgd_cfg["enable_self_consistency"],
            consistency_threshold=shgd_cfg["consistency_threshold"],
            consistency_check_interval=shgd_cfg.get("consistency_check_interval", 5),
            adaptive_heal=shgd_cfg.get("adaptive_heal", True),
            min_heal_strength=shgd_cfg.get("min_heal_strength", 0.1),
            max_heal_strength=shgd_cfg.get("max_heal_strength", 0.5),
            micro_heal=shgd_cfg.get("micro_heal", False),
            micro_heal_noise_scale=shgd_cfg.get("micro_heal_noise_scale", 0.05),
            freq_selective=shgd_cfg.get("freq_selective", False),
            freq_cutoff=shgd_cfg.get("freq_cutoff", 0.3),
            # v2 features
            negative_concept_scale=shgd_cfg.get("negative_concept_scale", 0.0),
            enable_sample_adaptive=shgd_cfg.get("enable_sample_adaptive", False),
            adaptive_guidance_min=shgd_cfg.get("adaptive_guidance_min", 1.0),
            adaptive_guidance_max=shgd_cfg.get("adaptive_guidance_max", 20.0),
            num_guide_heal_rounds=shgd_cfg.get("num_guide_heal_rounds", 1),
            round_guidance_decay=shgd_cfg.get("round_guidance_decay", 0.7),
            guidance_schedule=shgd_cfg.get("guidance_schedule", "constant"),
            enable_attn_suppression=shgd_cfg.get("enable_attn_suppression", False),
            attn_suppress_scale=shgd_cfg.get("attn_suppress_scale", 0.1),
            enable_concept_negation=shgd_cfg.get("enable_concept_negation", False),
            concept_negation_scale=shgd_cfg.get("concept_negation_scale", 3.0),
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        # Save images
        for img_idx, img in enumerate(images):
            fname = f"{global_idx:04d}.png"

            # Always save to all/
            img.save(os.path.join(all_dir, fname))

            # Evaluate with NudeNet if available
            if nudenet is not None:
                is_unsafe, pred_score = nudenet(img)
                if is_unsafe:
                    unsafe_cnt += 1
                    img.save(os.path.join(unsafe_dir, fname))
                else:
                    safe_cnt += 1
                    img.save(os.path.join(safe_dir, fname))

                results.append({
                    "idx": global_idx,
                    "prompt": prompt,
                    "unsafe": is_unsafe,
                    "score": float(pred_score),
                    "time": elapsed,
                })
            else:
                safe_cnt += 1  # assume safe if no evaluator
                results.append({
                    "idx": global_idx,
                    "prompt": prompt,
                    "time": elapsed,
                })

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(prompts)}, "
                  f"Safe: {safe_cnt}, Unsafe: {unsafe_cnt}, "
                  f"Avg time: {total_time/(idx+1):.1f}s/img")

    # Save results
    total = safe_cnt + unsafe_cnt
    summary = {
        "total": total,
        "safe": safe_cnt,
        "unsafe": unsafe_cnt,
        "safety_rate": safe_cnt / max(total, 1),
        "avg_time_per_image": total_time / max(len(prompts), 1),
        "config": shgd_cfg,
    }

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "per_image": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SHGD Generation Complete")
    print(f"{'='*60}")
    print(f"Total: {total}, Safe: {safe_cnt}, Unsafe: {unsafe_cnt}")
    print(f"Safety Rate: {summary['safety_rate']:.2%}")
    print(f"Avg Time: {summary['avg_time_per_image']:.1f}s/img")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
