"""
Run FK Steering with Nudity Classifier

Example usage:
    python run_fk_steering.py --prompt "a photo of a person at the beach" --target_class 1 --num_particles 4
"""

import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add paths
sys.path.insert(0, '../three_classificaiton')

from diffusers import StableDiffusionPipeline
from fk_steering import (
    fk_steering_pipeline,
    compare_fk_vs_baseline,
    NudityClassifierReward,
)
from geo_models.classifier.classifier import load_discriminator


def visualize_results(results, save_path=None):
    """Visualize comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Baseline
    axes[0].imshow(results['baseline']['image'])
    axes[0].set_title('Baseline (single sample)')
    axes[0].axis('off')

    # Best-of-n
    axes[1].imshow(results['best_of_n']['image'])
    axes[1].set_title(f"Best-of-n (n={len(results['best_of_n']['all_images'])})\n"
                     f"Reward: {max(results['best_of_n']['rewards']):.3f}")
    axes[1].axis('off')

    # FK steering
    axes[2].imshow(results['fk_steering']['images'])
    axes[2].set_title(f"FK Steering (k={len(results['fk_steering']['all_images'])})\n"
                     f"Reward: {max(results['fk_steering']['rewards']):.3f}")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_particles(results, save_path=None):
    """Visualize all particles from FK steering."""
    k = len(results['fk_steering']['all_images'])
    fig, axes = plt.subplots(1, k, figsize=(5*k, 5))

    if k == 1:
        axes = [axes]

    for i in range(k):
        axes[i].imshow(results['fk_steering']['all_images'][i])
        axes[i].set_title(f"Particle {i}\nReward: {results['fk_steering']['rewards'][i]:.3f}")
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved all particles to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_reward_history(results, save_path=None):
    """Plot reward evolution over time."""
    history = results['fk_steering']['particle_history']  # [num_steps, k]
    num_steps, k = history.shape

    plt.figure(figsize=(10, 6))
    for i in range(k):
        plt.plot(history[:, i], label=f'Particle {i}', alpha=0.7)

    plt.xlabel('Diffusion Step')
    plt.ylabel('Reward')
    plt.title('Particle Rewards Over Time (FK Steering)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reward history to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='FK Steering with Nudity Classifier')

    # Model paths
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4',
                       help='Stable Diffusion model ID')
    parser.add_argument('--classifier_ckpt', type=str,
                       default='./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth',
                       help='Path to classifier checkpoint')

    # Generation parameters
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt')
    parser.add_argument('--target_class', type=int, default=1,
                       help='Target class: 0=not people, 1=clothed, 2=nude')

    # FK steering parameters
    parser.add_argument('--num_particles', type=int, default=4,
                       help='Number of particles (k)')
    parser.add_argument('--potential_type', type=str, default='max',
                       choices=['max', 'difference', 'sum'],
                       help='Type of potential function')
    parser.add_argument('--lambda_scale', type=float, default=10.0,
                       help='Lambda scaling factor')
    parser.add_argument('--resampling_interval', type=int, default=10,
                       help='Resample every N steps')

    # Diffusion parameters
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='./fk_steering_outputs',
                       help='Output directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare FK vs baseline vs best-of-n')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading Stable Diffusion...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    ).to(device)

    print(f"Loading classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3,
    ).to(device)

    # Generate resampling schedule
    resampling_schedule = list(range(0, args.num_inference_steps, args.resampling_interval))

    # Create safe filename
    safe_prompt = args.prompt.replace(' ', '_')[:50]
    target_names = ['not_people', 'clothed', 'nude']
    base_filename = f"{safe_prompt}_target{args.target_class}_{target_names[args.target_class]}"

    if args.compare:
        print("\n" + "="*80)
        print("COMPARISON MODE: FK Steering vs Best-of-N vs Baseline")
        print("="*80 + "\n")

        results = compare_fk_vs_baseline(
            pipe=pipe,
            prompt=args.prompt,
            classifier=classifier,
            target_class=args.target_class,
            num_particles=args.num_particles,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )

        # Print results
        print("\n" + "-"*80)
        print("RESULTS:")
        print("-"*80)
        print(f"Best-of-N best reward: {max(results['best_of_n']['rewards']):.4f}")
        print(f"FK Steering best reward: {max(results['fk_steering']['rewards']):.4f}")
        print(f"FK Steering mean reward: {np.mean(results['fk_steering']['rewards']):.4f}")
        print("-"*80 + "\n")

        # Visualize
        comp_path = os.path.join(args.output_dir, f"{base_filename}_comparison.png")
        visualize_results(results, save_path=comp_path)

        particles_path = os.path.join(args.output_dir, f"{base_filename}_all_particles.png")
        visualize_all_particles(results, save_path=particles_path)

        history_path = os.path.join(args.output_dir, f"{base_filename}_reward_history.png")
        plot_reward_history(results, save_path=history_path)

        # Save best images
        results['baseline']['image'].save(
            os.path.join(args.output_dir, f"{base_filename}_baseline.png")
        )
        results['best_of_n']['image'].save(
            os.path.join(args.output_dir, f"{base_filename}_best_of_n.png")
        )
        results['fk_steering']['images'].save(
            os.path.join(args.output_dir, f"{base_filename}_fk_steering.png")
        )

    else:
        print("\n" + "="*80)
        print("FK STEERING ONLY MODE")
        print("="*80 + "\n")
        print(f"Prompt: {args.prompt}")
        print(f"Target class: {args.target_class} ({target_names[args.target_class]})")
        print(f"Num particles: {args.num_particles}")
        print(f"Potential type: {args.potential_type}")
        print(f"Lambda: {args.lambda_scale}")
        print(f"Resampling schedule: {resampling_schedule}")
        print("\n")

        results = fk_steering_pipeline(
            pipe=pipe,
            prompt=args.prompt,
            classifier=classifier,
            target_class=args.target_class,
            num_particles=args.num_particles,
            potential_type=args.potential_type,
            lambda_scale=args.lambda_scale,
            resampling_schedule=resampling_schedule,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_all_particles=True,
            verbose=True,
        )

        # Print results
        print("\n" + "-"*80)
        print("RESULTS:")
        print("-"*80)
        print(f"Best reward: {max(results['rewards']):.4f}")
        print(f"Mean reward: {np.mean(results['rewards']):.4f}")
        print(f"Best particle index: {results['best_idx']}")
        print("-"*80 + "\n")

        # Save all particles
        for i, img in enumerate(results['all_images']):
            img_path = os.path.join(
                args.output_dir,
                f"{base_filename}_particle{i}_reward{results['rewards'][i]:.3f}.png"
            )
            img.save(img_path)
            print(f"Saved particle {i} to {img_path}")

        # Plot reward history
        history_path = os.path.join(args.output_dir, f"{base_filename}_reward_history.png")
        plot_reward_history({'fk_steering': results}, save_path=history_path)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
