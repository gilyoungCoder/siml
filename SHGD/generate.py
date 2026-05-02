"""
SHGD Generation Script

Strong guidance in critical window + heal.
Supports multi-GPU with --gpu_id and --total_gpus.
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

from pipeline_shgd import SHGDPipeline


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(prompt_file):
    prompts = []
    path = Path(prompt_file)
    if path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            col = next(
                (c for c in ["sensitive prompt", "prompt", "adv_prompt"]
                 if c in reader.fieldnames), None
            )
            if col is None:
                raise ValueError(f"No prompt column in {prompt_file}")
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
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    # CLI overrides
    parser.add_argument("--anchor_guidance_scale", type=float, default=None)
    parser.add_argument("--guide_start_frac", type=float, default=None)
    parser.add_argument("--guide_end_frac", type=float, default=None)
    parser.add_argument("--heal_strength", type=float, default=None)
    parser.add_argument("--consistency_threshold", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--micro_heal", action="store_true", default=None)
    parser.add_argument("--freq_selective", action="store_true", default=None)
    parser.add_argument("--enable_trigger", type=str, default=None,
                        choices=["true", "false"],
                        help="Enable content-based trigger (default: from config)")
    parser.add_argument("--trigger_sim_threshold", type=float, default=None)
    parser.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    shgd_cfg = config["shgd"]
    eval_cfg = config.get("eval", {})

    # CLI overrides
    for key in ["anchor_guidance_scale", "guide_start_frac", "guide_end_frac",
                "heal_strength", "consistency_threshold"]:
        val = getattr(args, key, None)
        if val is not None:
            shgd_cfg[key] = val
    if args.guidance_scale is not None:
        model_cfg["guidance_scale"] = args.guidance_scale
    if args.micro_heal is not None:
        shgd_cfg["micro_heal"] = args.micro_heal
    if args.freq_selective is not None:
        shgd_cfg["freq_selective"] = args.freq_selective
    if args.enable_trigger is not None:
        shgd_cfg["enable_trigger"] = args.enable_trigger == "true"
    if args.trigger_sim_threshold is not None:
        shgd_cfg["trigger_sim_threshold"] = args.trigger_sim_threshold

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    if args.total_gpus > 1:
        chunk = len(prompts) // args.total_gpus
        start = args.gpu_id * chunk
        end = start + chunk if args.gpu_id < args.total_gpus - 1 else len(prompts)
        prompts = prompts[start:end]
        print(f"[GPU {args.gpu_id}] Prompts {start}-{end-1} ({len(prompts)} total)")

    # Output dirs
    os.makedirs(args.save_dir, exist_ok=True)
    safe_dir = os.path.join(args.save_dir, "safe")
    unsafe_dir = os.path.join(args.save_dir, "unsafe")
    all_dir = os.path.join(args.save_dir, "all")
    for d in [safe_dir, unsafe_dir, all_dir]:
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "config": config}, f, indent=2)

    # Load model
    device = args.device
    print(f"Loading model: {model_cfg['model_id']}")
    pipe = SHGDPipeline.from_pretrained(
        model_cfg["model_id"],
        scheduler=DDPMScheduler.from_pretrained(
            model_cfg["model_id"], subfolder="scheduler"
        ),
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

    # NudeNet for post-hoc eval
    nudenet = None
    if not args.skip_eval:
        try:
            from nudenet_eval import load_nudenet
            nudenet_path = eval_cfg.get("nudenet_path")
            if nudenet_path:
                config_dir = os.path.dirname(os.path.abspath(args.config))
                if not os.path.isabs(nudenet_path):
                    nudenet_path = os.path.normpath(
                        os.path.join(config_dir, nudenet_path)
                    )
                nudenet = load_nudenet(nudenet_path)
                print(f"NudeNet loaded from {nudenet_path}")
        except Exception as e:
            print(f"NudeNet not available ({e})")

    # Generate
    safe_cnt, unsafe_cnt = 0, 0
    results = []
    total_time = 0
    harmful_concepts = config.get("harmful_concepts")
    anchor_concepts = config.get("anchor_concepts")

    for idx, prompt in enumerate(prompts):
        global_idx = idx + (args.gpu_id * len(prompts) if args.total_gpus > 1 else 0)
        seed = args.seed + global_idx

        print(f"[{idx+1}/{len(prompts)}] {prompt[:80]}...")
        t0 = time.time()

        images = pipe(
            prompt,
            num_images_per_prompt=args.num_samples,
            guidance_scale=model_cfg["guidance_scale"],
            num_inference_steps=model_cfg["num_inference_steps"],
            height=model_cfg["image_size"],
            width=model_cfg["image_size"],
            generator=gen.manual_seed(seed),
            harmful_concepts=harmful_concepts,
            anchor_concepts=anchor_concepts,
            anchor_guidance_scale=shgd_cfg["anchor_guidance_scale"],
            guide_start_frac=shgd_cfg["guide_start_frac"],
            guide_end_frac=shgd_cfg["guide_end_frac"],
            heal_strength=shgd_cfg["heal_strength"],
            enable_self_consistency=shgd_cfg.get("enable_self_consistency", True),
            consistency_threshold=shgd_cfg.get("consistency_threshold", 0.85),
            consistency_check_interval=shgd_cfg.get("consistency_check_interval", 3),
            adaptive_heal=shgd_cfg.get("adaptive_heal", True),
            min_heal_strength=shgd_cfg.get("min_heal_strength", 0.2),
            max_heal_strength=shgd_cfg.get("max_heal_strength", 0.6),
            micro_heal=shgd_cfg.get("micro_heal", False),
            micro_heal_noise_scale=shgd_cfg.get("micro_heal_noise_scale", 0.05),
            freq_selective=shgd_cfg.get("freq_selective", False),
            freq_cutoff=shgd_cfg.get("freq_cutoff", 0.3),
            enable_trigger=shgd_cfg.get("enable_trigger", True),
            trigger_sim_threshold=shgd_cfg.get("trigger_sim_threshold", 0.3),
        )

        elapsed = time.time() - t0
        total_time += elapsed

        for img_idx, img in enumerate(images):
            fname = f"{global_idx:04d}.png"
            img.save(os.path.join(all_dir, fname))

            if nudenet is not None:
                is_unsafe, pred_score = nudenet(img)
                if is_unsafe:
                    unsafe_cnt += 1
                    img.save(os.path.join(unsafe_dir, fname))
                else:
                    safe_cnt += 1
                    img.save(os.path.join(safe_dir, fname))
                results.append({
                    "idx": global_idx, "prompt": prompt,
                    "unsafe": is_unsafe, "score": float(pred_score),
                    "time": elapsed,
                })
            else:
                safe_cnt += 1
                results.append({
                    "idx": global_idx, "prompt": prompt, "time": elapsed,
                })

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(prompts)} | "
                  f"Safe:{safe_cnt} Unsafe:{unsafe_cnt} | "
                  f"{total_time/(idx+1):.1f}s/img")

    total = safe_cnt + unsafe_cnt
    summary = {
        "total": total, "safe": safe_cnt, "unsafe": unsafe_cnt,
        "safety_rate": safe_cnt / max(total, 1),
        "avg_time": total_time / max(len(prompts), 1),
        "config": shgd_cfg,
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "per_image": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done | Total:{total} Safe:{safe_cnt} Unsafe:{unsafe_cnt}")
    print(f"Safety Rate: {summary['safety_rate']:.2%} | "
          f"Avg: {summary['avg_time']:.1f}s/img")
    print(f"Output: {args.save_dir}")


if __name__ == "__main__":
    main()
