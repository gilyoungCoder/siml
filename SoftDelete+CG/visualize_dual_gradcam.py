#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dual Grad-CAM Visualization for Multi-Concept Detection

Visualizes overlapping regions where both nudity and violence classifiers
detect harmful content simultaneously.

Visualization:
  - Red channel: Nudity detection heatmap
  - Green channel: Violence detection heatmap
  - Yellow (Red + Green): Overlapping regions where both concepts detected
  - Computes overlap percentage at different timesteps
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import csv

from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_gradcam(
    classifier,
    latent,
    timestep,
    target_class,
    layer_name="encoder_model.middle_block.2"
):
    """
    Compute Grad-CAM heatmap for a given class.

    Args:
        classifier: Classifier model
        latent: Latent tensor [B, C, H, W]
        timestep: Timestep tensor
        target_class: Target class index
        layer_name: Layer name for Grad-CAM

    Returns:
        heatmap: Grad-CAM heatmap [H, W]
        logit: Class logit value
    """
    # Hook for feature maps
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    target_layer = dict(classifier.named_modules())[layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    latent_input = latent.detach().to(dtype=next(classifier.parameters()).dtype).requires_grad_(True)

    # Ensure timestep is on correct device
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
    elif timestep.dim() == 0:
        timestep = timestep.unsqueeze(0).to(latent.device)
    else:
        timestep = timestep.to(latent.device)

    logits = classifier(latent_input, timestep)
    target_logit = logits[0, target_class]

    # Backward pass
    classifier.zero_grad()
    target_logit.backward()

    # Compute Grad-CAM
    activation = activations[0]  # [B, C, H, W]
    gradient = gradients[0]      # [B, C, H, W]

    # Global average pooling of gradients
    weights = gradient.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

    # Weighted combination
    cam = (weights * activation).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    cam = F.relu(cam)  # ReLU

    # Normalize to [0, 1]
    cam = cam.squeeze()  # [H, W]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    # Upsample to latent size
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(latent.shape[2], latent.shape[3]),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return cam.cpu().numpy(), target_logit.item()


def visualize_dual_gradcam(
    nudity_cam,
    violence_cam,
    image,
    nudity_logit,
    violence_logit,
    nudity_threshold,
    violence_threshold,
    step_idx,
    overlap_percentage,
    save_path
):
    """
    Visualize dual Grad-CAM with overlap.

    Red: Nudity heatmap
    Green: Violence heatmap
    Yellow: Overlap (both concepts detected)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Resize heatmaps to image size for visualization
    h, w = 512, 512
    nudity_cam_resized = np.array(Image.fromarray((nudity_cam * 255).astype(np.uint8)).resize((w, h)))
    violence_cam_resized = np.array(Image.fromarray((violence_cam * 255).astype(np.uint8)).resize((w, h)))

    # Normalize to [0, 1]
    nudity_cam_resized = nudity_cam_resized / 255.0
    violence_cam_resized = violence_cam_resized / 255.0

    # Row 1: Individual heatmaps
    # Nudity heatmap
    axes[0, 0].imshow(image)
    nudity_overlay = axes[0, 0].imshow(nudity_cam_resized, cmap='Reds', alpha=0.6)
    axes[0, 0].set_title(f'Nudity Grad-CAM\nLogit: {nudity_logit:.2f} (threshold: {nudity_threshold:.2f})',
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(nudity_overlay, ax=axes[0, 0], fraction=0.046)

    # Violence heatmap
    axes[0, 1].imshow(image)
    violence_overlay = axes[0, 1].imshow(violence_cam_resized, cmap='Greens', alpha=0.6)
    axes[0, 1].set_title(f'Violence Grad-CAM\nLogit: {violence_logit:.2f} (threshold: {violence_threshold:.2f})',
                          fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(violence_overlay, ax=axes[0, 1], fraction=0.046)

    # Combined RGB overlay
    # Red channel: Nudity
    # Green channel: Violence
    # Yellow areas = overlap
    rgb_overlay = np.zeros((h, w, 3))
    rgb_overlay[:, :, 0] = nudity_cam_resized      # Red
    rgb_overlay[:, :, 1] = violence_cam_resized    # Green
    rgb_overlay[:, :, 2] = 0                        # Blue (unused)

    axes[0, 2].imshow(image)
    axes[0, 2].imshow(rgb_overlay, alpha=0.6)
    axes[0, 2].set_title(f'Combined Overlay\nRed=Nudity, Green=Violence, Yellow=Overlap',
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Binary masks and overlap
    # Binary nudity mask (threshold at 0.5)
    nudity_mask = (nudity_cam_resized > 0.5).astype(float)
    axes[1, 0].imshow(nudity_mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Nudity Binary Mask (>0.5)\nCoverage: {nudity_mask.mean()*100:.1f}%',
                         fontsize=12)
    axes[1, 0].axis('off')

    # Binary violence mask
    violence_mask = (violence_cam_resized > 0.5).astype(float)
    axes[1, 1].imshow(violence_mask, cmap='Greens', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Violence Binary Mask (>0.5)\nCoverage: {violence_mask.mean()*100:.1f}%',
                          fontsize=12)
    axes[1, 1].axis('off')

    # Overlap mask
    overlap_mask = (nudity_mask * violence_mask).astype(float)
    overlap_rgb = np.zeros((h, w, 3))
    overlap_rgb[:, :, 0] = nudity_mask
    overlap_rgb[:, :, 1] = violence_mask
    overlap_rgb[:, :, 2] = overlap_mask  # Blue for overlap

    axes[1, 2].imshow(overlap_rgb)
    axes[1, 2].set_title(f'Overlap Analysis\nOverlap: {overlap_percentage:.1f}%\n' +
                         f'Nudity only: {(nudity_mask.sum() - overlap_mask.sum()) / nudity_mask.size * 100:.1f}%\n' +
                         f'Violence only: {(violence_mask.sum() - overlap_mask.sum()) / violence_mask.size * 100:.1f}%',
                         fontsize=12)
    axes[1, 2].axis('off')

    # Legend
    red_patch = mpatches.Patch(color='red', label='Nudity only')
    green_patch = mpatches.Patch(color='green', label='Violence only')
    blue_patch = mpatches.Patch(color='blue', label='Overlap')
    axes[1, 2].legend(handles=[red_patch, green_patch, blue_patch],
                      loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.suptitle(f'Dual Classifier Grad-CAM Analysis - Step {step_idx}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_dual_gradcam(
    pipe,
    prompt,
    nudity_classifier,
    violence_classifier,
    output_dir,
    args
):
    """
    Analyze a single prompt with dual Grad-CAM visualization.
    """
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Analyzing prompt: {prompt}")
    print(f"{'='*80}")

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=""
    )

    # Initialize latent
    latent = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=device,
        dtype=text_embeddings.dtype
    )

    # Storage for statistics
    overlap_stats = []

    # Denoising loop
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipe.scheduler.timesteps

    # Select timesteps to visualize (early, middle, late)
    vis_steps = [0, args.num_inference_steps // 2, args.num_inference_steps - 1]

    for step_idx, timestep in enumerate(tqdm(timesteps, desc="Generating")):
        # Expand latent for classifier-free guidance
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings
            ).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous sample
        latent = pipe.scheduler.step(noise_pred, timestep, latent).prev_sample

        # Compute Grad-CAMs
        nudity_cam, nudity_logit = compute_gradcam(
            nudity_classifier, latent, timestep,
            target_class=args.nudity_harmful_class
        )

        violence_cam, violence_logit = compute_gradcam(
            violence_classifier, latent, timestep,
            target_class=args.violence_harmful_class
        )

        # Compute overlap percentage
        nudity_mask = (nudity_cam > 0.5).astype(float)
        violence_mask = (violence_cam > 0.5).astype(float)
        overlap_mask = nudity_mask * violence_mask
        overlap_percentage = (overlap_mask.sum() / (nudity_mask.size)) * 100

        overlap_stats.append({
            'step': step_idx,
            'nudity_logit': nudity_logit,
            'violence_logit': violence_logit,
            'nudity_coverage': nudity_mask.mean() * 100,
            'violence_coverage': violence_mask.mean() * 100,
            'overlap_percentage': overlap_percentage
        })

        # Visualize at selected steps
        if step_idx in vis_steps:
            # Decode current latent to image
            with torch.no_grad():
                temp_latent = 1 / 0.18215 * latent
                temp_image = pipe.vae.decode(temp_latent).sample
                temp_image = (temp_image / 2 + 0.5).clamp(0, 1)
                temp_image = temp_image.cpu().permute(0, 2, 3, 1).numpy()[0]
                temp_image = (temp_image * 255).astype(np.uint8)

            save_path = output_dir / f"step_{step_idx:03d}_gradcam.png"
            visualize_dual_gradcam(
                nudity_cam=nudity_cam,
                violence_cam=violence_cam,
                image=temp_image,
                nudity_logit=nudity_logit,
                violence_logit=violence_logit,
                nudity_threshold=args.nudity_harmful_threshold,
                violence_threshold=args.violence_harmful_threshold,
                step_idx=step_idx,
                overlap_percentage=overlap_percentage,
                save_path=save_path
            )
            print(f"  Saved visualization: {save_path}")

    # Decode final image
    with torch.no_grad():
        latent = 1 / 0.18215 * latent
        final_image = pipe.vae.decode(latent).sample
        final_image = (final_image / 2 + 0.5).clamp(0, 1)
        final_image = final_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        final_image = (final_image * 255).astype(np.uint8)

    # Save final image
    Image.fromarray(final_image).save(output_dir / "final_image.png")

    # Save statistics
    stats_path = output_dir / "overlap_statistics.csv"
    with open(stats_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=overlap_stats[0].keys())
        writer.writeheader()
        writer.writerows(overlap_stats)

    print(f"\n  Saved statistics: {stats_path}")

    # Plot overlap over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    steps = [s['step'] for s in overlap_stats]

    # Logits over time
    axes[0].plot(steps, [s['nudity_logit'] for s in overlap_stats],
                 'r-', linewidth=2, label='Nudity logit')
    axes[0].plot(steps, [s['violence_logit'] for s in overlap_stats],
                 'g-', linewidth=2, label='Violence logit')
    axes[0].axhline(y=args.nudity_harmful_threshold, color='r', linestyle='--',
                    alpha=0.5, label=f'Nudity threshold ({args.nudity_harmful_threshold})')
    axes[0].axhline(y=args.violence_harmful_threshold, color='g', linestyle='--',
                    alpha=0.5, label=f'Violence threshold ({args.violence_harmful_threshold})')
    axes[0].set_xlabel('Denoising Step', fontsize=12)
    axes[0].set_ylabel('Logit Value', fontsize=12)
    axes[0].set_title('Classifier Logits Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Coverage and overlap over time
    axes[1].plot(steps, [s['nudity_coverage'] for s in overlap_stats],
                 'r-', linewidth=2, label='Nudity coverage')
    axes[1].plot(steps, [s['violence_coverage'] for s in overlap_stats],
                 'g-', linewidth=2, label='Violence coverage')
    axes[1].plot(steps, [s['overlap_percentage'] for s in overlap_stats],
                 'y-', linewidth=3, label='Overlap')
    axes[1].set_xlabel('Denoising Step', fontsize=12)
    axes[1].set_ylabel('Coverage (%)', fontsize=12)
    axes[1].set_title('Spatial Coverage Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "overlap_over_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved time analysis: {output_dir / 'overlap_over_time.png'}")

    # Summary statistics
    avg_overlap = np.mean([s['overlap_percentage'] for s in overlap_stats])
    max_overlap = np.max([s['overlap_percentage'] for s in overlap_stats])

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average overlap across all steps: {avg_overlap:.2f}%")
    print(f"Maximum overlap: {max_overlap:.2f}%")
    print(f"Final overlap: {overlap_stats[-1]['overlap_percentage']:.2f}%")
    print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dual Grad-CAM Visualization")

    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Stable Diffusion checkpoint")
    parser.add_argument("--nudity_classifier", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Nudity classifier path")
    parser.add_argument("--violence_classifier", type=str,
                        default="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth",
                        help="Violence classifier path")

    parser.add_argument("--prompt", type=str,
                        default="a naked person in a violent fight",
                        help="Prompt to analyze")
    parser.add_argument("--output_dir", type=str, default="dual_gradcam_analysis",
                        help="Output directory")

    parser.add_argument("--nudity_harmful_class", type=int, default=2,
                        help="Nudity harmful class index")
    parser.add_argument("--violence_harmful_class", type=int, default=2,
                        help="Violence harmful class index")
    parser.add_argument("--nudity_harmful_threshold", type=float, default=-0.5,
                        help="Nudity harmful threshold")
    parser.add_argument("--violence_harmful_threshold", type=float, default=-0.5,
                        help="Violence harmful threshold")

    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # Load models
    print(f"✓ Loading Stable Diffusion from {args.ckpt_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"✓ Loading nudity classifier from {args.nudity_classifier}...")
    nudity_classifier = load_discriminator(
        ckpt_path=args.nudity_classifier,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3  # 3-class classifier: NotPeople, Safe, Full
    ).to(device)

    print(f"✓ Loading violence classifier from {args.violence_classifier}...")
    violence_classifier = load_discriminator(
        ckpt_path=args.violence_classifier,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3  # 3-class classifier: NotRelevant, Safe, Full
    ).to(device)

    # Analyze
    analyze_dual_gradcam(
        pipe=pipe,
        prompt=args.prompt,
        nudity_classifier=nudity_classifier,
        violence_classifier=violence_classifier,
        output_dir=args.output_dir,
        args=args
    )

    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
