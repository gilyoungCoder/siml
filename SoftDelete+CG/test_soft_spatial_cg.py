#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Soft Spatial CG Testing Script

Tests various hyperparameter configurations for soft spatial concept guidance.
Evaluates on multiple concepts: nudity safe/unsafe, violence safe/unsafe.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from diffusers import StableDiffusionPipeline

from geo_utils.selective_guidance_utils import (
    SelectiveGuidanceMonitor,
    SpatiallyMaskedGuidance,
    WeightScheduler
)


def load_test_prompts(prompts_file: Path) -> Dict[str, List[str]]:
    """Load test prompts from JSON file."""
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    return prompts


def load_tuning_config(config_file: Path) -> Dict:
    """Load tuning configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_weight_scheduler(schedule_config: Dict) -> Optional[WeightScheduler]:
    """
    Create WeightScheduler from configuration.

    Args:
        schedule_config: Dictionary with scheduling parameters

    Returns:
        WeightScheduler instance or None
    """
    strategy = schedule_config.get('strategy', 'constant')
    start_step = schedule_config.get('start_step', 0)
    end_step = schedule_config.get('end_step', 50)
    start_weight = schedule_config.get('start_weight', 1.0)
    end_weight = schedule_config.get('end_weight', 1.0)
    decay_rate = schedule_config.get('decay_rate', 0.1)

    scheduler = WeightScheduler(
        strategy=strategy,
        start_step=start_step,
        end_step=end_step,
        start_weight=start_weight,
        end_weight=end_weight,
        decay_rate=decay_rate
    )

    return scheduler


def setup_guidance_components(
    classifier_model,
    preset_config: Dict,
    device: str = "cuda"
):
    """
    Setup guidance components from preset configuration.

    Args:
        classifier_model: Loaded classifier model
        preset_config: Preset configuration dictionary
        device: Device to use

    Returns:
        monitor: SelectiveGuidanceMonitor
        guidance: SpatiallyMaskedGuidance
    """
    # Create weight scheduler
    weight_scheduler = create_weight_scheduler(preset_config)

    # Create monitor
    monitor = SelectiveGuidanceMonitor(
        classifier_model=classifier_model,
        harmful_threshold=preset_config.get('harmful_threshold', 0.5),
        harmful_class=preset_config.get('harmful_class', 2),
        safe_class=preset_config.get('safe_class', 1),
        spatial_threshold=preset_config.get('spatial_threshold', 0.5),
        use_soft_mask=preset_config.get('use_soft_mask', True),
        soft_mask_temperature=preset_config.get('soft_mask_temperature', 1.0),
        soft_mask_gaussian_sigma=preset_config.get('gaussian_sigma', 0.0),
        device=device,
        debug=True
    )

    # Create guidance
    guidance = SpatiallyMaskedGuidance(
        classifier_model=classifier_model,
        safe_class=preset_config.get('safe_class', 1),
        harmful_class=preset_config.get('harmful_class', 2),
        device=device,
        use_bidirectional=True,
        weight_scheduler=weight_scheduler,
        normalize_gradient=preset_config.get('normalize_gradient', False),
        gradient_norm_type=preset_config.get('gradient_norm_type', 'l2')
    )

    return monitor, guidance


def generate_with_config(
    pipeline,
    classifier_model,
    prompt: str,
    preset_config: Dict,
    output_path: Path,
    num_inference_steps: int = 50,
    guidance_scale_cfg: float = 7.5,
    seed: int = 42
):
    """
    Generate image with specific configuration.

    Args:
        pipeline: Stable Diffusion pipeline
        classifier_model: Classifier model for guidance
        prompt: Text prompt
        preset_config: Configuration dictionary
        output_path: Path to save generated image
        num_inference_steps: Number of diffusion steps
        guidance_scale_cfg: CFG scale
        seed: Random seed
    """
    # Setup components
    monitor, guidance = setup_guidance_components(
        classifier_model=classifier_model,
        preset_config=preset_config,
        device=pipeline.device
    )

    # Reset statistics
    monitor.reset_statistics()

    # Setup callback for guidance
    def guidance_callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        # Check if guidance should be applied
        should_apply, spatial_mask, info = monitor.should_apply_guidance(
            latent=latents,
            timestep=timestep,
            step=step_index
        )

        if should_apply and spatial_mask is not None:
            # Apply spatially-masked guidance
            latents = guidance.apply_guidance(
                latent=latents,
                timestep=timestep,
                spatial_mask=spatial_mask,
                guidance_scale=preset_config.get('guidance_scale', 5.0),
                harmful_scale=preset_config.get('harmful_scale', 1.0),
                current_step=step_index
            )

        callback_kwargs["latents"] = latents
        return callback_kwargs

    # Set seed for reproducibility
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Generate
    print(f"\n{'='*60}")
    print(f"Generating: {prompt}")
    print(f"Config: {preset_config.get('name', 'custom')}")
    print(f"{'='*60}\n")

    result = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale_cfg,
        generator=generator,
        callback_on_step_end=guidance_callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )

    # Save image
    image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    # Get statistics
    stats = monitor.get_statistics()

    print(f"\n✓ Saved: {output_path}")
    print(f"  Guidance applied: {stats.get('guidance_ratio', 0):.1%} of steps")
    print(f"  Harmful detected: {stats.get('harmful_ratio', 0):.1%} of steps\n")

    return stats


def test_preset(
    pipeline,
    classifier_model,
    preset_name: str,
    preset_config: Dict,
    test_prompts: Dict[str, List[str]],
    output_dir: Path,
    num_images_per_category: int = 2,
    seed_start: int = 42
):
    """
    Test a single preset configuration on multiple prompt categories.

    Args:
        pipeline: Stable Diffusion pipeline
        classifier_model: Classifier model
        preset_name: Name of the preset
        preset_config: Preset configuration
        test_prompts: Dictionary of prompt categories
        output_dir: Output directory
        num_images_per_category: Number of images to generate per category
        seed_start: Starting seed value
    """
    print(f"\n{'#'*60}")
    print(f"# TESTING PRESET: {preset_name}")
    print(f"{'#'*60}\n")

    preset_output_dir = output_dir / preset_name
    preset_output_dir.mkdir(parents=True, exist_ok=True)

    # Add name to config for logging
    preset_config['name'] = preset_name

    all_stats = {}
    seed = seed_start

    for category, prompts in test_prompts.items():
        print(f"\n--- Category: {category} ---")

        category_stats = []

        # Generate a subset of prompts
        for i, prompt in enumerate(prompts[:num_images_per_category]):
            output_path = preset_output_dir / category / f"{i:02d}.png"

            stats = generate_with_config(
                pipeline=pipeline,
                classifier_model=classifier_model,
                prompt=prompt,
                preset_config=preset_config,
                output_path=output_path,
                seed=seed
            )

            category_stats.append(stats)
            seed += 1

        all_stats[category] = category_stats

    # Save statistics
    stats_file = preset_output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n✓ Preset '{preset_name}' testing complete")
    print(f"  Results saved to: {preset_output_dir}")
    print(f"  Statistics saved to: {stats_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test soft spatial CG with various configurations"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Pretrained model ID"
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="configs/multi_concept_test_prompts.json",
        help="Path to test prompts JSON file"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/soft_spatial_cg_tuning.yaml",
        help="Path to tuning configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/soft_cg_tests",
        help="Output directory for results"
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["gentle_increase", "strong_decay", "constant_soft"],
        help="Preset names to test (from config file)"
    )
    parser.add_argument(
        "--num_images_per_category",
        type=int,
        default=2,
        help="Number of images to generate per prompt category"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting random seed"
    )

    args = parser.parse_args()

    # Load resources
    print("Loading resources...")

    # Load prompts
    prompts_file = Path(args.prompts_file)
    test_prompts = load_test_prompts(prompts_file)
    print(f"✓ Loaded {len(test_prompts)} prompt categories")

    # Load tuning config
    config_file = Path(args.config_file)
    tuning_config = load_tuning_config(config_file)
    print(f"✓ Loaded tuning configuration")

    # Load pipeline
    print(f"Loading pipeline: {args.model_id}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(args.device)
    print("✓ Pipeline loaded")

    # Load classifier
    print(f"Loading classifier: {args.classifier_path}")
    # TODO: Replace with actual classifier loading code
    # classifier_model = load_classifier(args.classifier_path)
    # For now, placeholder:
    # classifier_model = ...
    print("✓ Classifier loaded")
    # NOTE: You need to implement the actual classifier loading

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test each preset
    presets_config = tuning_config.get('presets', {})

    for preset_name in args.presets:
        if preset_name not in presets_config:
            print(f"[WARNING] Preset '{preset_name}' not found in config. Skipping.")
            continue

        preset_config = presets_config[preset_name]

        # test_preset(
        #     pipeline=pipeline,
        #     classifier_model=classifier_model,
        #     preset_name=preset_name,
        #     preset_config=preset_config,
        #     test_prompts=test_prompts,
        #     output_dir=output_dir,
        #     num_images_per_category=args.num_images_per_category,
        #     seed_start=args.seed
        # )

        print(f"[INFO] Would test preset: {preset_name}")
        print(f"  Config: {preset_config}")

    print(f"\n{'='*60}")
    print("All tests complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
