#!/usr/bin/env python3
"""
SoftDelete+CG Grid Search Worker

Loads the SD pipeline and SoftDelete+CG classifier ONCE, then iterates
over all parameter combinations assigned to this GPU.

Usage (called by run_grid_search.py):
    CUDA_VISIBLE_DEVICES=0 python scg_grid_worker.py --config worker_gpu0.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add SoftDelete+CG to path
SCG_DIR = "/mnt/home/yhgil99/unlearning/SoftDelete+CG"
sys.path.insert(0, SCG_DIR)

from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(prompt_file, max_prompts=None):
    """Load prompts from txt or csv file."""
    if prompt_file.endswith(".csv"):
        import csv
        prompts = []
        with open(prompt_file) as f:
            reader = csv.DictReader(f)
            col = "sensitive prompt" if "sensitive prompt" in reader.fieldnames else "prompt"
            for row in reader:
                prompts.append(row[col].strip())
    else:
        with open(prompt_file) as f:
            prompts = [l.strip() for l in f if l.strip()]

    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


def load_gradcam_stats(stats_dir):
    """Load per-class GradCAM statistics."""
    stats = {}
    mapping = {
        2: "gradcam_stats_harm_nude_class2.json",
        3: "gradcam_stats_harm_color_class3.json",
    }
    for cls, fname in mapping.items():
        path = os.path.join(stats_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            stats[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
    return stats


def make_tag(combo):
    """Create a filesystem-safe tag string."""
    parts = [f"gs{combo['guidance_scale']}"]
    parts.append(f"st{combo['spatial_threshold_start']}")
    parts.append(f"se{combo['spatial_threshold_end']}")
    parts.append(combo["threshold_strategy"][:3])
    parts.append(f"hs{combo['harmful_scale']}")
    if combo.get("base_guidance_scale", 0) > 0:
        parts.append(f"bgs{combo['base_guidance_scale']}")
    return "_".join(parts)


def apply_cdf_normalization(heatmap, mean, std):
    """Convert raw heatmap to CDF normalized [0,1] range."""
    z = (heatmap - mean) / (std + 1e-8)
    from torch.distributions import Normal
    normal = Normal(
        torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype),
        torch.tensor(1.0, device=heatmap.device, dtype=heatmap.dtype),
    )
    return normal.cdf(z)


def get_threshold(strategy, start, end, step, total_steps):
    """Get spatial threshold for current step."""
    if strategy == "constant":
        return start
    t = step / max(total_steps - 1, 1)
    if strategy == "linear_decrease":
        return start - (start - end) * t
    elif strategy == "cosine_anneal":
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
    return start


class SCGGuidanceRunner:
    """
    Runs SoftDelete+CG-style selective spatial classifier guidance.

    At each denoising step:
    1. Classify current noisy latent -> detect harm
    2. If harmful: compute GradCAM heatmap -> spatial mask
    3. Apply bidirectional gradient guidance only in harmful regions
    """

    def __init__(self, classifier, gradcam_stats, device):
        self.classifier = classifier
        self.gradcam_stats = gradcam_stats
        self.device = device
        self.dtype = next(classifier.parameters()).dtype

        # GradCAM setup
        from geo_utils.classifier_interpretability import ClassifierGradCAM
        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier,
            target_layer_name="encoder_model.middle_block.2"
        )

        # 4-class config
        self.safe_classes = [0, 1]
        self.harm_classes = [2, 3]
        self.target_safe = 1

    def detect_and_guide(self, latent, timestep, step, combo, total_steps):
        """
        Detect harm and apply guidance if needed.

        Returns:
            guided_latent: latent after guidance
            info: dict with detection details
        """
        B = latent.shape[0]

        # Prepare timestep
        if not isinstance(timestep, torch.Tensor):
            ts = torch.tensor([timestep], device=self.device, dtype=torch.long)
        else:
            ts = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep
        if ts.shape[0] != B:
            ts = ts.expand(B)

        norm_ts = ts.float() / 1000.0

        # Classify
        with torch.no_grad():
            latent_input = latent.to(dtype=self.dtype)
            logits = self.classifier(latent_input, norm_ts)
            probs = F.softmax(logits, dim=1)
            max_class = logits.argmax(dim=1)[0].item()

        is_harmful = max_class in self.harm_classes
        harm_class = max_class if is_harmful else None

        info = {
            "step": step,
            "is_harmful": is_harmful,
            "max_class": max_class,
            "probs": probs[0].detach().cpu().numpy().tolist(),
        }

        if not is_harmful:
            return latent, info

        # Get spatial threshold
        threshold = get_threshold(
            combo["threshold_strategy"],
            combo["spatial_threshold_start"],
            combo["spatial_threshold_end"],
            step, total_steps,
        )

        # Compute GradCAM heatmap
        use_cdf = (self.gradcam_stats and harm_class in self.gradcam_stats)
        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(
                latent=latent_input, timestep=norm_ts,
                target_class=harm_class, normalize=not use_cdf,
            )

        if use_cdf:
            stats = self.gradcam_stats[harm_class]
            heatmap = apply_cdf_normalization(heatmap, stats["mean"], stats["std"])

        # Binary spatial mask
        mask = (heatmap >= threshold).float()
        mask_ratio = mask.mean().item()
        info["mask_ratio"] = mask_ratio

        # Compute bidirectional gradient: safe_grad - harmful_scale * harm_grad
        gs = combo["guidance_scale"]
        hs = combo["harmful_scale"]
        bgs = combo.get("base_guidance_scale", 0.0)

        with torch.enable_grad():
            # Safe gradient
            lat_safe = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            logits_safe = self.classifier(lat_safe, norm_ts)
            safe_logit = logits_safe[:, self.target_safe].sum()
            grad_safe = torch.autograd.grad(safe_logit, lat_safe)[0]

            # Harmful gradient
            lat_harm = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            logits_harm = self.classifier(lat_harm, norm_ts)
            harm_logit = logits_harm[:, harm_class].sum()
            grad_harm = torch.autograd.grad(harm_logit, lat_harm)[0]

        # Bidirectional gradient
        grad = grad_safe - hs * grad_harm

        # Spatially-weighted guidance
        mask_4d = mask.unsqueeze(1)  # [B, 1, H, W]
        weight_map = mask_4d * gs + (1 - mask_4d) * bgs
        weighted_grad = grad.to(dtype=latent.dtype) * weight_map.to(dtype=latent.dtype)

        # Clip gradient magnitude
        wg_norm = weighted_grad.norm()
        lat_norm = latent.norm()
        max_ratio = 0.1  # at most 10% of latent magnitude
        if wg_norm > lat_norm * max_ratio and wg_norm > 0:
            weighted_grad = weighted_grad * (lat_norm * max_ratio / wg_norm)

        guided_latent = latent + weighted_grad.detach()
        info["guided"] = True
        return guided_latent, info


def run_experiments(config):
    """Run all experiments assigned to this GPU."""
    common = config["common"]
    experiments = config["experiments"]
    gpu_id = config["gpu_id"]

    device = torch.device("cuda")
    print(f"[GPU {gpu_id}] Loading SD pipeline...")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        common["ckpt_path"], safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load classifier
    print(f"[GPU {gpu_id}] Loading classifier: {common['classifier_ckpt']}")
    classifier_ckpt = common["classifier_ckpt"]

    # Load using load_discriminator
    from geo_models.classifier.classifier import discriminator_defaults, create_classifier
    disc_args = discriminator_defaults(num_classes=4)
    classifier = create_classifier(**disc_args)
    state_dict = torch.load(classifier_ckpt, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.to(device).eval()

    # Load gradcam stats
    gradcam_stats = {}
    if common.get("gradcam_stats_dir") and os.path.exists(common["gradcam_stats_dir"]):
        gradcam_stats = load_gradcam_stats(common["gradcam_stats_dir"])
        print(f"[GPU {gpu_id}] Loaded GradCAM stats: {list(gradcam_stats.keys())}")

    # Create guidance runner
    runner = SCGGuidanceRunner(classifier, gradcam_stats, device)

    # Load prompts
    prompts = load_prompts(common["prompt_file"], common.get("max_prompts"))
    print(f"[GPU {gpu_id}] Loaded {len(prompts)} prompts")

    num_steps = common["num_inference_steps"]
    n_exp = len(experiments)

    for exp_idx, combo in enumerate(experiments):
        tag = make_tag(combo)
        output_dir = os.path.join(common["output_dir"], tag)

        # Skip if done
        if os.path.exists(output_dir):
            existing = len(list(Path(output_dir).glob("*.png")))
            if existing >= len(prompts):
                print(f"[GPU {gpu_id}] SKIP ({exp_idx+1}/{n_exp}): {tag}")
                continue

        print(f"\n[GPU {gpu_id}] ({exp_idx+1}/{n_exp}) {tag}")
        os.makedirs(output_dir, exist_ok=True)

        # Save config
        cfg_save = {**combo, "tag": tag, **{k: v for k, v in common.items()
                    if k not in ("experiments",)}}
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(cfg_save, f, indent=2)

        set_seed(common["seed"])

        # Track per-prompt guidance stats
        prompt_stats = []
        t0 = time.time()

        for idx, prompt in enumerate(prompts):
            set_seed(common["seed"] + idx)

            step_infos = []

            def callback_on_step_end(pipe_ref, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                guided_latents, info = runner.detect_and_guide(
                    latents, timestep, step, combo, num_steps,
                )
                callback_kwargs["latents"] = guided_latents
                step_infos.append(info)
                return callback_kwargs

            with torch.enable_grad():
                output = pipe(
                    prompt=prompt,
                    guidance_scale=common["cfg_scale"],
                    num_inference_steps=num_steps,
                    height=512, width=512,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                    num_images_per_prompt=common["nsamples"],
                )

            # Save image
            for si, img in enumerate(output.images):
                fname = f"prompt_{idx+1:04d}_sample_{si+1}.png"
                img_path = os.path.join(output_dir, fname)
                img = np.asarray(img)
                Image.fromarray(img, mode="RGB").resize((512, 512)).save(img_path)

            # Record stats
            n_guided = sum(1 for s in step_infos if s.get("is_harmful", False))
            prompt_stats.append({
                "idx": idx,
                "prompt": prompt[:80],
                "n_guided_steps": n_guided,
                "total_steps": num_steps,
                "guidance_ratio": n_guided / max(num_steps, 1),
            })

            if idx % 5 == 0:
                print(f"  [{idx+1}/{len(prompts)}] guided={n_guided}/{num_steps} | {prompt[:50]}")

        elapsed = time.time() - t0

        # Save experiment stats
        exp_stats = {
            "tag": tag,
            "combo": combo,
            "n_prompts": len(prompts),
            "elapsed": elapsed,
            "avg_guidance_ratio": np.mean([s["guidance_ratio"] for s in prompt_stats]),
            "prompts_with_guidance": sum(1 for s in prompt_stats if s["n_guided_steps"] > 0),
            "prompt_stats": prompt_stats,
        }
        with open(os.path.join(output_dir, "experiment_stats.json"), "w") as f:
            json.dump(exp_stats, f, indent=2)

        print(f"  Done in {elapsed:.0f}s | "
              f"guided: {exp_stats['prompts_with_guidance']}/{len(prompts)} prompts | "
              f"avg ratio: {exp_stats['avg_guidance_ratio']:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to worker config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[SCG Worker] GPU {config['gpu_id']}: "
          f"{len(config['experiments'])} experiments")

    run_experiments(config)
    print(f"\n[SCG Worker] GPU {config['gpu_id']}: ALL DONE!")


if __name__ == "__main__":
    main()
