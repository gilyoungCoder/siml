#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weight Scheduling 시각화 도구

다양한 scheduling 전략을 비교하여 시각화합니다.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from geo_utils.selective_guidance_utils import WeightScheduler


def visualize_schedules(
    num_steps: int = 50,
    output_path: str = "weight_schedules.png"
):
    """
    다양한 weight scheduling 전략을 시각화합니다.
    """
    # Define strategies to visualize
    configs = [
        {
            "name": "Constant",
            "strategy": "constant",
            "start_weight": 1.0,
            "end_weight": 1.0,
            "color": "blue",
            "linestyle": "-"
        },
        {
            "name": "Linear Increase",
            "strategy": "linear_increase",
            "start_weight": 0.5,
            "end_weight": 3.0,
            "color": "green",
            "linestyle": "-"
        },
        {
            "name": "Linear Decrease",
            "strategy": "linear_decrease",
            "start_weight": 3.0,
            "end_weight": 0.5,
            "color": "red",
            "linestyle": "-"
        },
        {
            "name": "Cosine Anneal",
            "strategy": "cosine_anneal",
            "start_weight": 5.0,
            "end_weight": 0.5,
            "color": "purple",
            "linestyle": "-"
        },
        {
            "name": "Exponential Decay",
            "strategy": "exponential_decay",
            "start_weight": 10.0,
            "end_weight": 0.1,
            "color": "orange",
            "linestyle": "-",
            "decay_rate": 0.1
        }
    ]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Steps array
    steps = np.arange(0, num_steps)

    # Plot 1: All strategies together
    ax1 = axes[0]
    for config in configs:
        scheduler = WeightScheduler(
            strategy=config["strategy"],
            start_step=0,
            end_step=num_steps - 1,
            start_weight=config["start_weight"],
            end_weight=config["end_weight"],
            decay_rate=config.get("decay_rate", 0.1)
        )

        weights = [scheduler.get_weight(step) for step in steps]

        ax1.plot(
            steps,
            weights,
            label=config["name"],
            color=config["color"],
            linestyle=config["linestyle"],
            linewidth=2.5,
            marker='o',
            markersize=3,
            markevery=5
        )

    ax1.set_xlabel("Denoising Step", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Weight Multiplier", fontsize=13, fontweight='bold')
    ax1.set_title("Weight Scheduling Strategies Comparison", fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, num_steps - 1])

    # Plot 2: Cosine variations
    ax2 = axes[1]

    cosine_configs = [
        {"start": 10.0, "end": 0.1, "label": "Strong → Very Weak (10→0.1)", "color": "darkred"},
        {"start": 5.0, "end": 0.5, "label": "Strong → Weak (5→0.5)", "color": "red"},
        {"start": 3.0, "end": 1.0, "label": "Medium → Medium (3→1)", "color": "orange"},
        {"start": 2.0, "end": 1.5, "label": "Weak → Weak (2→1.5)", "color": "gold"},
    ]

    for config in cosine_configs:
        scheduler = WeightScheduler(
            strategy="cosine_anneal",
            start_step=0,
            end_step=num_steps - 1,
            start_weight=config["start"],
            end_weight=config["end"]
        )

        weights = [scheduler.get_weight(step) for step in steps]

        ax2.plot(
            steps,
            weights,
            label=config["label"],
            color=config["color"],
            linewidth=2.5,
            marker='s',
            markersize=3,
            markevery=5
        )

    ax2.set_xlabel("Denoising Step", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Weight Multiplier", fontsize=13, fontweight='bold')
    ax2.set_title("Cosine Annealing - Different Weight Ranges", fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, num_steps - 1])

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")

    plt.show()


def visualize_temperature_effect(
    output_path: str = "temperature_effect.png"
):
    """
    Soft mask temperature의 효과를 시각화합니다.
    """
    # Heatmap value range
    x = np.linspace(0, 1, 200)
    threshold = 0.5

    # Different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors = ['darkred', 'red', 'orange', 'gold', 'yellow']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Sigmoid curves
    ax1 = axes[0]
    for temp, color in zip(temperatures, colors):
        # Sigmoid function: σ((x - threshold) / temperature)
        sigmoid = 1 / (1 + np.exp(-(x - threshold) / temp))

        ax1.plot(
            x,
            sigmoid,
            label=f'Temp = {temp}',
            color=color,
            linewidth=2.5
        )

    # Add reference lines
    ax1.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    ax1.set_xlabel("Heatmap Value", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Mask Value", fontsize=13, fontweight='bold')
    ax1.set_title("Soft Masking: Temperature Effect", fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot 2: Derivative (gradient of mask)
    ax2 = axes[1]
    for temp, color in zip(temperatures, colors):
        # Derivative of sigmoid
        sigmoid = 1 / (1 + np.exp(-(x - threshold) / temp))
        derivative = sigmoid * (1 - sigmoid) / temp

        ax2.plot(
            x,
            derivative,
            label=f'Temp = {temp}',
            color=color,
            linewidth=2.5
        )

    ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')

    ax2.set_xlabel("Heatmap Value", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Gradient Magnitude", fontsize=13, fontweight='bold')
    ax2.set_title("Mask Gradient (Transition Sharpness)", fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved temperature effect visualization to: {output_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize weight scheduling strategies")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")
    print()

    # Weight schedules
    print("[1/2] Visualizing weight schedules...")
    visualize_schedules(
        num_steps=args.num_steps,
        output_path=output_dir / "weight_schedules.png"
    )

    # Temperature effect
    print("\n[2/2] Visualizing temperature effect...")
    visualize_temperature_effect(
        output_path=output_dir / "temperature_effect.png"
    )

    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
