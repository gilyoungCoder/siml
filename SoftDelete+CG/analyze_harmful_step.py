#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze at which diffusion step harmful content becomes detectable
by monitoring classifier logits throughout the generation process.

This script runs SD 1.4 with a harmful prompt and tracks when the
classifier starts detecting harmful content (class 1 logit increases).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from typing import List, Dict, Tuple
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Analyze harmful content detection by step")

    # Model paths
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Stable Diffusion model ID")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class_clothed/checkpoint/step_10700/classifier.pth",
                        help="Classifier checkpoint path")

    # Prompts
    parser.add_argument("--harmful_prompt", type=str,
                        default="nude woman on the beach",
                        help="Harmful prompt to test")
    parser.add_argument("--safe_prompt", type=str,
                        default="a woman walking on the beach",
                        help="Safe prompt for comparison")

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                        help="Output directory for plots and data")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classifier classes")

    args = parser.parse_args()
    return args


# =========================
# Classifier Wrapper
# =========================
class ClassifierWrapper:
    """Wrapper for the classifier to evaluate latents at each step"""

    def __init__(self, classifier_ckpt: str, num_classes: int = 3, device: str = "cuda", use_fp16: bool = False):
        self.device = device
        self.num_classes = num_classes
        self.use_fp16 = use_fp16

        # Load classifier
        print(f"Loading classifier from {classifier_ckpt}...")
        self.classifier = load_discriminator(
            ckpt_path=classifier_ckpt,
            condition=None,
            eval=True,
            channel=4,  # Latent channels
            num_classes=num_classes
        ).to(device)

        # Keep classifier in float32 to avoid timestep embedding issues
        # We'll convert inputs to float32 instead
        self.classifier.eval()

        print(f"Classifier loaded successfully on {device} (float32 for stability)")

    @torch.no_grad()
    def evaluate_latent(self, latent: torch.Tensor, timestep: int) -> Dict[str, float]:
        """
        Evaluate a latent representation with the classifier

        Args:
            latent: Latent tensor [B, C, H, W]
            timestep: Current timestep

        Returns:
            Dictionary with logits and probabilities for each class
        """
        # Convert latent to float32 for classifier (handles both fp16 and fp32 inputs)
        latent = latent.to(self.device).float()

        # Create timestep tensor
        t = torch.tensor([timestep], device=self.device)

        # Get classifier prediction
        logits = self.classifier(latent, t)  # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)

        # Convert to dict
        result = {
            "timestep": timestep,
            "logits": logits[0].cpu().numpy(),
            "probs": probs[0].cpu().numpy(),
        }

        # Add per-class logits and probs
        for i in range(self.num_classes):
            result[f"logit_class_{i}"] = logits[0, i].item()
            result[f"prob_class_{i}"] = probs[0, i].item()

        return result


# =========================
# Custom Callback for Latent Monitoring
# =========================
class LatentMonitorCallback:
    """Callback to monitor latents at each diffusion step"""

    def __init__(self, classifier: ClassifierWrapper):
        self.classifier = classifier
        self.step_data = []

    def __call__(self, step: int, timestep: int, latents: torch.Tensor) -> Dict:
        """Called at each diffusion step"""
        # Evaluate latents with classifier
        eval_result = self.classifier.evaluate_latent(latents, timestep)
        eval_result["step"] = step

        # Store results
        self.step_data.append(eval_result)

        return {}

    def get_results(self) -> List[Dict]:
        """Get all collected step data"""
        return self.step_data


# =========================
# Generation with Monitoring
# =========================
def generate_with_monitoring(
    pipe: StableDiffusionPipeline,
    classifier: ClassifierWrapper,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42
):
    """
    Generate image while monitoring classifier predictions at each step

    Returns:
        step_data: List of dictionaries containing classifier results per step
        image: Generated PIL image
    """
    # Set seed
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Create callback
    callback = LatentMonitorCallback(classifier)

    # Generate with callback
    print(f"\nGenerating with prompt: '{prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")

    output = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback=callback,
        callback_steps=1,  # Call at every step
        output_type="pil"  # Return PIL image directly
    )

    image = output.images[0]
    step_data = callback.get_results()

    return step_data, image


# =========================
# Visualization
# =========================
def plot_logit_progression(
    harmful_data: List[Dict],
    safe_data: List[Dict],
    output_path: str,
    num_classes: int = 3
):
    """
    Plot logit progression for harmful vs safe prompts

    Creates multiple subplots:
    1. Harmful Class logit comparison
    2. Safe Class logit comparison
    3. All classes logits for harmful prompt
    4. All classes logits for safe prompt
    5. Harmful Class probability comparison
    6. Safe Class probability comparison
    """
    # Class names mapping
    class_names = {
        0: 'Not-Relevant',
        1: 'Clothed People',
        2: 'Nude People'
    }

    # Color mapping for classes
    class_colors = {
        0: 'blue',     # Not-Relevant
        1: 'green',    # Clothed (Safe)
        2: 'red'       # Nude (Harmful)
    }

    # Set style
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # Extract data
    harmful_steps = [d["step"] for d in harmful_data]
    safe_steps = [d["step"] for d in safe_data]

    # === Plot 1: Nude Class (Class 2) Logit Comparison ===
    ax1 = axes[0, 0]
    harmful_class2_logits = [d["logit_class_2"] for d in harmful_data]
    safe_class2_logits = [d["logit_class_2"] for d in safe_data]

    ax1.plot(harmful_steps, harmful_class2_logits, 'r-o', linewidth=2,
             markersize=4, label='Harmful Prompt', alpha=0.8)
    ax1.plot(safe_steps, safe_class2_logits, 'g-s', linewidth=2,
             markersize=4, label='Safe Prompt', alpha=0.8)

    ax1.set_xlabel('Diffusion Step', fontsize=12)
    ax1.set_ylabel('Nude People Logit', fontsize=12)
    ax1.set_title('Nude People (Class 2) Logit Progression: Harmful vs Safe Prompt', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # Find crossover point (when harmful becomes detectable)
    threshold = 0.0  # Logit threshold
    harmful_above_threshold = [i for i, logit in enumerate(harmful_class2_logits) if logit > threshold]
    if harmful_above_threshold:
        first_detection = harmful_steps[harmful_above_threshold[0]]
        ax1.axvline(x=first_detection, color='orange', linestyle='--',
                   linewidth=2, label=f'First Detection (step {first_detection})')
        ax1.legend(fontsize=11)

    # === Plot 2: Clothed Class (Class 1) Logit Comparison ===
    ax2 = axes[0, 1]
    harmful_class1_logits = [d["logit_class_1"] for d in harmful_data]
    safe_class1_logits = [d["logit_class_1"] for d in safe_data]

    ax2.plot(harmful_steps, harmful_class1_logits, 'r-o', linewidth=2,
             markersize=4, label='Harmful Prompt', alpha=0.8)
    ax2.plot(safe_steps, safe_class1_logits, 'g-s', linewidth=2,
             markersize=4, label='Safe Prompt', alpha=0.8)

    ax2.set_xlabel('Diffusion Step', fontsize=12)
    ax2.set_ylabel('Clothed People Logit', fontsize=12)
    ax2.set_title('Clothed People (Class 1) Logit Progression: Harmful vs Safe Prompt', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # === Plot 3: All Classes for Harmful Prompt ===
    ax3 = axes[1, 0]

    for cls in range(num_classes):
        logits = [d[f"logit_class_{cls}"] for d in harmful_data]
        ax3.plot(harmful_steps, logits, marker='o', linewidth=2,
                markersize=4, label=class_names.get(cls, f'Class {cls}'),
                color=class_colors.get(cls, 'gray'), alpha=0.8)

    ax3.set_xlabel('Diffusion Step', fontsize=12)
    ax3.set_ylabel('Logit Value', fontsize=12)
    ax3.set_title('All Classes Logits - Harmful Prompt', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # === Plot 4: All Classes for Safe Prompt ===
    ax4 = axes[1, 1]

    for cls in range(num_classes):
        logits = [d[f"logit_class_{cls}"] for d in safe_data]
        ax4.plot(safe_steps, logits, marker='s', linewidth=2,
                markersize=4, label=class_names.get(cls, f'Class {cls}'),
                color=class_colors.get(cls, 'gray'), alpha=0.8)

    ax4.set_xlabel('Diffusion Step', fontsize=12)
    ax4.set_ylabel('Logit Value', fontsize=12)
    ax4.set_title('All Classes Logits - Safe Prompt', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # === Plot 5: Nude Class Probability Comparison ===
    ax5 = axes[2, 0]
    harmful_class2_probs = [d["prob_class_2"] for d in harmful_data]
    safe_class2_probs = [d["prob_class_2"] for d in safe_data]

    ax5.plot(harmful_steps, harmful_class2_probs, 'r-o', linewidth=2,
             markersize=4, label='Harmful Prompt', alpha=0.8)
    ax5.plot(safe_steps, safe_class2_probs, 'g-s', linewidth=2,
             markersize=4, label='Safe Prompt', alpha=0.8)

    ax5.set_xlabel('Diffusion Step', fontsize=12)
    ax5.set_ylabel('Nude People Probability', fontsize=12)
    ax5.set_title('Nude People (Class 2) Probability: Detection Confidence', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])

    # Add threshold line
    ax5.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='50% Threshold')
    ax5.legend(fontsize=11)

    # === Plot 6: Clothed Class Probability Comparison ===
    ax6 = axes[2, 1]
    harmful_class1_probs = [d["prob_class_1"] for d in harmful_data]
    safe_class1_probs = [d["prob_class_1"] for d in safe_data]

    ax6.plot(harmful_steps, harmful_class1_probs, 'r-o', linewidth=2,
             markersize=4, label='Harmful Prompt', alpha=0.8)
    ax6.plot(safe_steps, safe_class1_probs, 'g-s', linewidth=2,
             markersize=4, label='Safe Prompt', alpha=0.8)

    ax6.set_xlabel('Diffusion Step', fontsize=12)
    ax6.set_ylabel('Clothed People Probability', fontsize=12)
    ax6.set_title('Clothed People (Class 1) Probability: Detection Confidence', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])

    # Add threshold line
    ax6.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='50% Threshold')
    ax6.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_detection_heatmap(
    harmful_data: List[Dict],
    output_path: str,
    num_classes: int = 3
):
    """
    Create a heatmap showing class probabilities across steps
    """
    # Class names
    class_names = ['Not-Relevant', 'Clothed People', 'Nude People']

    plt.figure(figsize=(14, 6))

    # Prepare data
    steps = [d["step"] for d in harmful_data]
    probs_matrix = np.zeros((num_classes, len(steps)))

    for cls in range(num_classes):
        probs_matrix[cls, :] = [d[f"prob_class_{cls}"] for d in harmful_data]

    # Create heatmap
    sns.heatmap(
        probs_matrix,
        xticklabels=[f"{s}" if s % 5 == 0 else "" for s in steps],
        yticklabels=[class_names[i] if i < len(class_names) else f"Class {i}" for i in range(num_classes)],
        cmap="RdYlGn_r",
        annot=False,
        fmt=".2f",
        cbar_kws={'label': 'Probability'},
        vmin=0,
        vmax=1
    )

    plt.xlabel('Diffusion Step', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.title('Class Probability Evolution - Harmful Prompt', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def save_numerical_results(
    harmful_data: List[Dict],
    safe_data: List[Dict],
    output_path: str
):
    """Save numerical results to CSV for further analysis"""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        num_classes = len([k for k in harmful_data[0].keys() if k.startswith("logit_class_")])
        header = ["step", "prompt_type", "timestep"]
        for cls in range(num_classes):
            header.extend([f"logit_class_{cls}", f"prob_class_{cls}"])
        writer.writerow(header)

        # Write harmful data
        for d in harmful_data:
            row = [d["step"], "harmful", d["timestep"]]
            for cls in range(num_classes):
                row.extend([d[f"logit_class_{cls}"], d[f"prob_class_{cls}"]])
            writer.writerow(row)

        # Write safe data
        for d in safe_data:
            row = [d["step"], "safe", d["timestep"]]
            for cls in range(num_classes):
                row.extend([d[f"logit_class_{cls}"], d[f"prob_class_{cls}"]])
            writer.writerow(row)

    print(f"Numerical results saved to: {output_path}")


def print_analysis_summary(
    harmful_data: List[Dict],
    safe_data: List[Dict]
):
    """Print summary statistics about when harmful content is detected"""
    # Class names
    class_names = {
        0: 'Not-Relevant',
        1: 'Clothed People',
        2: 'Nude People'
    }

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    # Find when harmful (nude) logit becomes positive
    harmful_class2_logits = [d["logit_class_2"] for d in harmful_data]
    steps = [d["step"] for d in harmful_data]

    positive_steps = [s for s, logit in zip(steps, harmful_class2_logits) if logit > 0]

    print("\n" + "─"*80)
    print("NUDE PEOPLE (CLASS 2) DETECTION")
    print("─"*80)

    if positive_steps:
        first_positive = min(positive_steps)
        print(f"\n✓ Nude content first detected at step: {first_positive}")
        print(f"  (Nude People logit became positive: {harmful_data[first_positive]['logit_class_2']:.4f})")
    else:
        print("\n✗ Nude content never clearly detected (no positive logit)")

    # Find when probability exceeds 50%
    harmful_class2_probs = [d["prob_class_2"] for d in harmful_data]
    high_conf_steps = [s for s, prob in zip(steps, harmful_class2_probs) if prob > 0.5]

    if high_conf_steps:
        first_high_conf = min(high_conf_steps)
        print(f"\n✓ High confidence Nude People detection (>50% prob) at step: {first_high_conf}")
        print(f"  (Probability: {harmful_data[first_high_conf]['prob_class_2']:.4f})")
    else:
        print("\n✗ Never reached high confidence threshold (>50%)")

    # Clothed class analysis
    print("\n" + "─"*80)
    print("CLOTHED PEOPLE (CLASS 1) DETECTION")
    print("─"*80)

    harmful_class1_logits = [d["logit_class_1"] for d in harmful_data]
    safe_class1_logits = [d["logit_class_1"] for d in safe_data]

    # When does safe prompt have high clothed class probability?
    safe_class1_probs = [d["prob_class_1"] for d in safe_data]
    safe_high_conf_steps = [s for s, prob in zip(steps, safe_class1_probs) if prob > 0.5]

    if safe_high_conf_steps:
        first_safe_high_conf = min(safe_high_conf_steps)
        print(f"\n✓ Safe prompt classified as Clothed People (>50% prob) at step: {first_safe_high_conf}")
        print(f"  (Probability: {safe_data[first_safe_high_conf]['prob_class_1']:.4f})")
    else:
        print("\n✗ Safe prompt never strongly classified as Clothed People")

    # Final step statistics
    final_harmful = harmful_data[-1]
    final_safe = safe_data[-1]

    print(f"\n{'─'*80}")
    print("FINAL STEP COMPARISON (Step {})".format(steps[-1]))
    print(f"{'─'*80}")

    print("\nHarmful Prompt:")
    for cls in range(len([k for k in final_harmful.keys() if k.startswith("logit_class_")])):
        cls_name = class_names.get(cls, f"Class {cls}")
        print(f"  {cls_name:20s}: Logit={final_harmful[f'logit_class_{cls}']:>7.4f}  "
              f"Prob={final_harmful[f'prob_class_{cls}']:>6.4f}")

    print("\nSafe Prompt:")
    for cls in range(len([k for k in final_safe.keys() if k.startswith("logit_class_")])):
        cls_name = class_names.get(cls, f"Class {cls}")
        print(f"  {cls_name:20s}: Logit={final_safe[f'logit_class_{cls}']:>7.4f}  "
              f"Prob={final_safe[f'prob_class_{cls}']:>6.4f}")

    # Calculate differences for both harmful and safe classes
    print(f"\n{'─'*80}")
    print(f"DIFFERENCE (Harmful Prompt - Safe Prompt):")
    print(f"{'─'*80}")

    for cls in range(len([k for k in final_harmful.keys() if k.startswith("logit_class_")])):
        cls_name = class_names.get(cls, f"Class {cls}")
        logit_diff = final_harmful[f"logit_class_{cls}"] - final_safe[f"logit_class_{cls}"]
        prob_diff = final_harmful[f"prob_class_{cls}"] - final_safe[f"prob_class_{cls}"]
        print(f"  {cls_name:20s}: Logit={logit_diff:>+.4f}  Prob={prob_diff:>+.4f}")

    print("="*80 + "\n")


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("HARMFUL CONTENT DETECTION ANALYSIS")
    print("="*80)
    print(f"\nModel: {args.model_id}")
    print(f"Classifier: {args.classifier_ckpt}")
    print(f"Steps: {args.num_inference_steps}")
    print(f"Harmful prompt: '{args.harmful_prompt}'")
    print(f"Safe prompt: '{args.safe_prompt}'")
    print(f"Output directory: {args.output_dir}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"\nUsing device: {device}")
    print(f"Using FP16: {use_fp16}")

    # Load Stable Diffusion pipeline
    print("\nLoading Stable Diffusion 1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # Use DDIM scheduler for consistency
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print("✓ Pipeline loaded")

    # Load classifier with matching dtype
    classifier = ClassifierWrapper(
        classifier_ckpt=args.classifier_ckpt,
        num_classes=args.num_classes,
        device=device,
        use_fp16=use_fp16
    )

    # Generate with harmful prompt
    print("\n" + "─"*80)
    print("GENERATING WITH HARMFUL PROMPT")
    print("─"*80)
    harmful_data, harmful_image = generate_with_monitoring(
        pipe=pipe,
        classifier=classifier,
        prompt=args.harmful_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

    # Save harmful image
    harmful_img_path = os.path.join(args.output_dir, "harmful_generated.png")
    harmful_image.save(harmful_img_path)
    print(f"✓ Harmful image saved to: {harmful_img_path}")

    # Generate with safe prompt
    print("\n" + "─"*80)
    print("GENERATING WITH SAFE PROMPT")
    print("─"*80)
    safe_data, safe_image = generate_with_monitoring(
        pipe=pipe,
        classifier=classifier,
        prompt=args.safe_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

    # Save safe image
    safe_img_path = os.path.join(args.output_dir, "safe_generated.png")
    safe_image.save(safe_img_path)
    print(f"✓ Safe image saved to: {safe_img_path}")

    # Create visualizations
    print("\n" + "─"*80)
    print("CREATING VISUALIZATIONS")
    print("─"*80)

    plot_path = os.path.join(args.output_dir, "logit_progression.png")
    plot_logit_progression(harmful_data, safe_data, plot_path, args.num_classes)

    heatmap_path = os.path.join(args.output_dir, "detection_heatmap.png")
    plot_detection_heatmap(harmful_data, heatmap_path, args.num_classes)

    # Save numerical results
    csv_path = os.path.join(args.output_dir, "step_analysis.csv")
    save_numerical_results(harmful_data, safe_data, csv_path)

    # Print analysis summary
    print_analysis_summary(harmful_data, safe_data)

    print("\n✓ Analysis complete!")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
