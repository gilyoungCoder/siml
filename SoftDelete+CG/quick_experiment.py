#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Experiment Script for Soft Spatial CG

빠르게 파라미터를 테스트할 수 있는 스크립트
"""

import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

from geo_utils.selective_guidance_utils import (
    SelectiveGuidanceMonitor,
    SpatiallyMaskedGuidance,
    WeightScheduler
)


def quick_test(
    prompt: str,
    # Soft masking parameters
    use_soft_mask: bool = True,
    soft_mask_temperature: float = 1.0,
    gaussian_sigma: float = 0.5,
    spatial_threshold: float = 0.5,
    # Weight scheduling
    strategy: str = "cosine_anneal",
    start_weight: float = 3.0,
    end_weight: float = 0.5,
    # Guidance parameters
    guidance_scale: float = 5.0,
    harmful_scale: float = 1.0,
    harmful_threshold: float = 0.5,
    # Normalization
    normalize_gradient: bool = True,
    gradient_norm_type: str = "l2",
    # Other parameters
    output_path: str = "output.png",
    seed: int = 42,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    **kwargs
):
    """
    빠른 테스트 함수

    Usage:
        python quick_experiment.py \
            --prompt "a nude woman on the beach" \
            --soft_mask_temperature 1.0 \
            --strategy cosine_anneal \
            --guidance_scale 5.0
    """
    print("="*60)
    print("SOFT SPATIAL CG - QUICK EXPERIMENT")
    print("="*60)
    print(f"\nPrompt: {prompt}\n")
    print("Configuration:")
    print(f"  Soft Masking:")
    print(f"    - use_soft_mask: {use_soft_mask}")
    print(f"    - temperature: {soft_mask_temperature}")
    print(f"    - gaussian_sigma: {gaussian_sigma}")
    print(f"    - spatial_threshold: {spatial_threshold}")
    print(f"\n  Weight Scheduling:")
    print(f"    - strategy: {strategy}")
    print(f"    - start_weight: {start_weight}")
    print(f"    - end_weight: {end_weight}")
    print(f"\n  Guidance:")
    print(f"    - guidance_scale: {guidance_scale}")
    print(f"    - harmful_scale: {harmful_scale}")
    print(f"    - harmful_threshold: {harmful_threshold}")
    print(f"\n  Normalization:")
    print(f"    - normalize_gradient: {normalize_gradient}")
    print(f"    - gradient_norm_type: {gradient_norm_type}")
    print("="*60 + "\n")

    # NOTE: 실제 구현시 classifier와 pipeline을 로드해야 합니다
    # 여기서는 skeleton만 제공

    # Load models (TODO: implement)
    # pipeline = StableDiffusionPipeline.from_pretrained(...)
    # classifier = load_classifier(...)

    # Create weight scheduler
    scheduler = WeightScheduler(
        strategy=strategy,
        start_step=0,
        end_step=num_steps,
        start_weight=start_weight,
        end_weight=end_weight
    )

    # Setup monitor
    # monitor = SelectiveGuidanceMonitor(
    #     classifier_model=classifier,
    #     harmful_threshold=harmful_threshold,
    #     spatial_threshold=spatial_threshold,
    #     use_soft_mask=use_soft_mask,
    #     soft_mask_temperature=soft_mask_temperature,
    #     soft_mask_gaussian_sigma=gaussian_sigma,
    #     device="cuda",
    #     debug=True
    # )

    # Setup guidance
    # guidance = SpatiallyMaskedGuidance(
    #     classifier_model=classifier,
    #     weight_scheduler=scheduler,
    #     normalize_gradient=normalize_gradient,
    #     gradient_norm_type=gradient_norm_type,
    #     device="cuda"
    # )

    # Generate callback
    # def callback(pipe, step_index, timestep, callback_kwargs):
    #     latents = callback_kwargs["latents"]
    #
    #     should_apply, spatial_mask, info = monitor.should_apply_guidance(
    #         latent=latents,
    #         timestep=timestep,
    #         step=step_index
    #     )
    #
    #     if should_apply and spatial_mask is not None:
    #         # Print weight schedule info
    #         if step_index % 10 == 0:
    #             weight = scheduler.get_weight(step_index)
    #             print(f"[Step {step_index}] Weight: {weight:.3f}")
    #
    #         latents = guidance.apply_guidance(
    #             latent=latents,
    #             timestep=timestep,
    #             spatial_mask=spatial_mask,
    #             guidance_scale=guidance_scale,
    #             harmful_scale=harmful_scale,
    #             current_step=step_index
    #         )
    #
    #     callback_kwargs["latents"] = latents
    #     return callback_kwargs

    # Generate
    # generator = torch.Generator(device="cuda").manual_seed(seed)
    # result = pipeline(
    #     prompt=prompt,
    #     num_inference_steps=num_steps,
    #     guidance_scale=cfg_scale,
    #     generator=generator,
    #     callback_on_step_end=callback,
    #     callback_on_step_end_tensor_inputs=["latents"]
    # )

    # Save
    # image = result.images[0]
    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # image.save(output_path)

    # Stats
    # stats = monitor.get_statistics()
    # print(f"\n{'='*60}")
    # print("RESULTS")
    # print(f"{'='*60}")
    # print(f"✓ Image saved to: {output_path}")
    # print(f"  Guidance applied: {stats.get('guidance_ratio', 0):.1%} of steps")
    # print(f"  Harmful detected: {stats.get('harmful_ratio', 0):.1%} of steps")
    # print(f"{'='*60}\n")

    print("[INFO] This is a skeleton script. Implement model loading to use.")


def main():
    parser = argparse.ArgumentParser(
        description="Quick experiment with soft spatial CG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt")
    parser.add_argument("--output_path", type=str, default="output.png",
                        help="Output image path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Soft masking
    parser.add_argument("--use_soft_mask", action="store_true", default=True,
                        help="Use soft masking")
    parser.add_argument("--soft_mask_temperature", type=float, default=1.0,
                        help="Soft mask temperature (0.1=sharp, 5.0=very soft)")
    parser.add_argument("--gaussian_sigma", type=float, default=0.5,
                        help="Gaussian smoothing sigma (0=none)")
    parser.add_argument("--spatial_threshold", type=float, default=0.5,
                        help="Spatial threshold for masking")

    # Weight scheduling
    parser.add_argument("--strategy", type=str, default="cosine_anneal",
                        choices=["constant", "linear_increase", "linear_decrease",
                                 "cosine_anneal", "exponential_decay"],
                        help="Weight scheduling strategy")
    parser.add_argument("--start_weight", type=float, default=3.0,
                        help="Starting weight")
    parser.add_argument("--end_weight", type=float, default=0.5,
                        help="Ending weight")

    # Guidance
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Guidance scale")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale")
    parser.add_argument("--harmful_threshold", type=float, default=0.5,
                        help="Harmful detection threshold")

    # Normalization
    parser.add_argument("--normalize_gradient", action="store_true", default=True,
                        help="Normalize gradients")
    parser.add_argument("--gradient_norm_type", type=str, default="l2",
                        choices=["l2", "layer"],
                        help="Gradient normalization type")

    # Generation
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")

    args = parser.parse_args()

    quick_test(**vars(args))


if __name__ == "__main__":
    main()
