#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI interface for CG Delete MVP.

Minimal implementation based on genTCMMM.py structure.
Focuses on attention-based harmful concept deletion without maintain/ADD-LIST features.
"""

import os
import sys
from argparse import ArgumentParser
from functools import partial
from typing import Optional, List
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline

# Import local modules
from .utils import (
    set_seed,
    save_image,
    build_harm_vector,
    get_scheduler,
)
from .attention_delete import AttentionEraser, HarmConfig, build_sot_exempt_mask


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="CG Delete MVP: Attention-based harmful concept deletion")

    # Positional arguments
    parser.add_argument("model_id", type=str, help="HuggingFace model ID or local path")

    # Generation parameters
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompts file (one per line)")
    parser.add_argument("--outdir", type=str, default="./outputs/run1", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale (CFG)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")

    # Harmful concept deletion
    parser.add_argument("--harm_list", type=str, default=None, help="Path to harmful keywords file")
    parser.add_argument("--gamma", type=float, default=0.35, help="Deletion strength (γ)")
    parser.add_argument("--gamma_sched", type=str, default="cosine",
                        choices=["fixed", "linear", "cosine"],
                        help="Gamma scheduling strategy")
    parser.add_argument("--gamma_start", type=float, default=None,
                        help="Starting gamma (overrides --gamma for scheduling)")
    parser.add_argument("--gamma_end", type=float, default=None,
                        help="Ending gamma (overrides --gamma for scheduling)")
    parser.add_argument("--harm_tau", type=float, default=0.1, help="Cosine similarity threshold (τ)")
    parser.add_argument("--harm_layer", type=int, default=-1,
                        help="Text encoder layer index for harm vector")
    parser.add_argument("--harm_vec_mode", type=str, default="masked_mean",
                        choices=["masked_mean", "token"],
                        help="Harm vector construction mode")
    parser.add_argument("--include_special_tokens", action="store_true",
                        help="Include SOT/EOT/PAD in averaging")

    # Optional: negative prompt (documented as not recommended for deletion)
    parser.add_argument("--neg_prompt", type=str, default=None,
                        help="Negative prompt (not recommended for deletion purposes)")

    args = parser.parse_args()
    return args


def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from file.

    Args:
        prompt_file: Path to prompts file (one per line)

    Returns:
        List of prompt strings
    """
    prompt_file = os.path.expanduser(prompt_file)
    if not os.path.isfile(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    return prompts


def load_harm_concepts(harm_list_file: Optional[str]) -> List[str]:
    """Load harmful concepts from file.

    Args:
        harm_list_file: Path to harmful keywords file (one per line)

    Returns:
        List of harmful concept strings
    """
    if harm_list_file is None:
        return []

    harm_list_file = os.path.expanduser(harm_list_file)
    if not os.path.isfile(harm_list_file):
        raise FileNotFoundError(f"Harm list file not found: {harm_list_file}")

    with open(harm_list_file, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f if line.strip()]

    return concepts


def create_step_callback(
    eraser: AttentionEraser,
    scheduler_fn,
    gamma_start: float,
    gamma_end: float,
    num_steps: int,
):
    """Create step-end callback for gamma scheduling.

    Args:
        eraser: AttentionEraser instance
        scheduler_fn: Scheduling function
        gamma_start: Starting gamma value
        gamma_end: Ending gamma value
        num_steps: Total number of inference steps

    Returns:
        Callback function
    """
    def callback_on_step_end(pipe, step: int, timestep, callback_kwargs):
        # Update gamma based on schedule
        gamma = scheduler_fn(step, num_steps, gamma_start, gamma_end)
        eraser.set_harm_gamma(gamma)
        return callback_kwargs

    return callback_on_step_end


def main():
    """Main execution function."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    print(f"[main] Device: {device}")
    print(f"[main] Model: {args.model_id}")
    print(f"[main] Output: {args.outdir}")
    print(f"[main] Seed: {args.seed}")

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load pipeline
    print(f"[main] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    print(f"[main] Pipeline loaded on {pipe.device}")

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f"[main] Loaded {len(prompts)} prompts from {args.prompt_file}")

    # Load harmful concepts
    harm_concepts = load_harm_concepts(args.harm_list)
    if len(harm_concepts) == 0:
        print("[main] WARNING: No harmful concepts provided. Running without deletion.")
        harm_vec = None
        enable_harm = False
    else:
        print(f"[main] Loaded {len(harm_concepts)} harmful concepts: {harm_concepts}")
        enable_harm = True

        # Build harm vector
        print(f"[main] Building harm vector (mode={args.harm_vec_mode}, layer={args.harm_layer})...")
        harm_vec = build_harm_vector(
            pipe,
            harm_concepts,
            layer_idx=args.harm_layer,
            include_special=args.include_special_tokens,
            mode=args.harm_vec_mode,
            target_words=harm_concepts if args.harm_vec_mode == "token" else None,
        )

        if harm_vec is None:
            print("[main] WARNING: Failed to build harm vector. Running without deletion.")
            enable_harm = False

    # Setup gamma scheduling
    if args.gamma_start is not None and args.gamma_end is not None:
        gamma_start = args.gamma_start
        gamma_end = args.gamma_end
    else:
        gamma_start = args.gamma
        gamma_end = args.gamma

    scheduler_fn = get_scheduler(args.gamma_sched)
    print(f"[main] Gamma schedule: {args.gamma_sched} ({gamma_start} → {gamma_end})")

    # Create attention eraser
    harm_cfg = HarmConfig(
        enable=enable_harm,
        tau=args.harm_tau,
        gamma=gamma_start,  # Will be updated per step
    )

    eraser = AttentionEraser(harm_vec=harm_vec, harm_cfg=harm_cfg)
    pipe.unet.set_attn_processor(eraser)

    print(f"[main] Harm suppression: enabled={enable_harm}, tau={args.harm_tau}, "
          f"gamma={gamma_start}→{gamma_end}")

    # Create step callback
    callback = create_step_callback(
        eraser=eraser,
        scheduler_fn=scheduler_fn,
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        num_steps=args.steps,
    )

    # Generation loop
    print(f"\n[main] Starting generation...")
    print("=" * 80)

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")

        # Build SOT exempt mask for this prompt
        sot_mask = build_sot_exempt_mask(pipe, prompt)
        eraser.set_soft_exempt_mask(sot_mask)

        # Run inference
        with torch.enable_grad():  # Enable grad for potential guidance extensions
            output = pipe(
                prompt=prompt,
                negative_prompt=args.neg_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_images_per_prompt=args.nsamples,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        # Save images
        for i, image in enumerate(output.images):
            if args.nsamples == 1:
                filename = f"{idx+1:04d}.png"
            else:
                filename = f"{idx+1:04d}_{i+1}.png"

            save_path = os.path.join(args.outdir, filename)
            save_image(image, save_path)

    print("\n" + "=" * 80)
    print(f"[main] Generation complete! Images saved to: {args.outdir}")


if __name__ == "__main__":
    main()
