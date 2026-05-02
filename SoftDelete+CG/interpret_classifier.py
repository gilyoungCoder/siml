"""
Classifier Interpretation Script

Analyzes which latent regions the classifier focuses on when detecting nude content.

Usage:
    # Analyze images from a directory
    python interpret_classifier.py --image_dir ./test_images --output_dir ./interpretations

    # Analyze a single image
    python interpret_classifier.py --image_path ./test.png --output_dir ./interpretations

    # Analyze during generation (step-by-step)
    python interpret_classifier.py --mode generation --prompt "a person" --output_dir ./gen_analysis
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

from diffusers import AutoencoderKL, DDPMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.classifier_interpretability import (
    ClassifierGradCAM,
    LayerwiseActivationAnalyzer,
    IntegratedGradients,
    VisualizationUtils,
    load_classifier_for_interpretation
)


def encode_image_to_latent(image_path: Path, vae: AutoencoderKL, device: str = "cuda"):
    """
    Encode an image to latent space.

    Args:
        image_path: Path to image file
        vae: VAE model
        device: Device

    Returns:
        latent: [1, 4, 64, 64] latent tensor
        original_image: [H, W, 3] numpy array (0-255)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    original_image = np.array(image)  # Keep original for visualization

    # To tensor
    image_tensor = torch.from_numpy(original_image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]
    image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
    image_tensor = image_tensor.to(device)

    # Match VAE dtype (float16 or float32)
    if vae.dtype == torch.float16:
        image_tensor = image_tensor.half()

    # Encode
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor

    return latent, original_image


def analyze_single_image(
    image_path: Path,
    classifier,
    vae,
    output_dir: Path,
    timestep: int = 500,
    target_class: int = 2,
    device: str = "cuda"
):
    """
    Perform comprehensive interpretation on a single image.

    Args:
        image_path: Path to image
        classifier: Loaded classifier model
        vae: VAE for encoding
        output_dir: Directory to save results
        timestep: Timestep to use (0-999)
        target_class: Class to analyze (2 = nude)
        device: Device
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path.name}")
    print(f"{'='*60}")

    # Encode image
    print("Encoding image to latent...")
    latent, original_image = encode_image_to_latent(image_path, vae, device)

    # Convert latent to float32 for classifier (classifier expects float32)
    latent = latent.float()

    # Prepare timestep
    timestep_tensor = torch.tensor([timestep], device=device, dtype=torch.long)

    # ========== 1. Grad-CAM Analysis ==========
    print("\n[1/3] Running Grad-CAM...")

    gradcam = ClassifierGradCAM(classifier, target_layer_name="encoder_model.middle_block.2")
    heatmap, info = gradcam.generate_heatmap(
        latent, timestep_tensor, target_class=target_class
    )

    # Print prediction
    probs = info['probs'][0].cpu().numpy()
    class_names = ['Not People', 'Clothed', 'Nude']
    print(f"  Prediction: {class_names[probs.argmax()]} ({probs.max():.3f})")
    print(f"  Probabilities: {dict(zip(class_names, probs))}")

    # Save Grad-CAM visualization with original color image
    gradcam_path = output_dir / f"{image_path.stem}_gradcam.png"
    VisualizationUtils.save_heatmap_with_image(
        original_image, heatmap[0], info,
        save_path=gradcam_path,
        title=f"Grad-CAM: {class_names[target_class]} Detection"
    )

    # Cleanup
    gradcam.remove_hooks()

    # ========== 2. Layer-wise Activation Analysis ==========
    print("\n[2/3] Analyzing layer-wise activations...")

    layer_analyzer = LayerwiseActivationAnalyzer(classifier)
    layer_analyzer.register_layer_hooks()

    activation_maps = layer_analyzer.analyze(latent, timestep_tensor)

    # Visualize activation statistics
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    layer_names = [k for k in activation_maps.keys() if k not in ['logits', 'probs']]

    for idx, layer_name in enumerate(layer_names[:6]):
        acts = activation_maps[layer_name][0]  # [C, H, W]

        # Compute mean activation per channel
        mean_acts = acts.mean(dim=(1, 2)).cpu().numpy()

        axes[idx].bar(range(len(mean_acts)), mean_acts)
        axes[idx].set_title(f"{layer_name}\nShape: {acts.shape}")
        axes[idx].set_xlabel('Channel')
        axes[idx].set_ylabel('Mean Activation')

    plt.tight_layout()
    layer_path = output_dir / f"{image_path.stem}_layers.png"
    plt.savefig(layer_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved layer analysis to {layer_path}")

    layer_analyzer.remove_hooks()

    # ========== 3. Integrated Gradients ==========
    print("\n[3/3] Computing Integrated Gradients...")

    ig = IntegratedGradients(classifier)
    attribution = ig.attribute(
        latent, timestep_tensor, target_class=target_class, n_steps=50
    )

    # Visualize attribution per channel
    fig = VisualizationUtils.visualize_attribution_channels(attribution[0])
    ig_path = output_dir / f"{image_path.stem}_integrated_gradients.png"
    plt.savefig(ig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved IG attribution to {ig_path}")

    # ========== Summary Statistics ==========
    summary = {
        'image': str(image_path),
        'timestep': timestep,
        'target_class': target_class,
        'predictions': {
            'logits': info['logits'][0].cpu().tolist(),
            'probs': probs.tolist(),
            'predicted_class': int(probs.argmax()),
            'predicted_class_name': class_names[probs.argmax()]
        },
        'gradcam': {
            'max_attention': float(heatmap[0].max()),
            'mean_attention': float(heatmap[0].mean()),
            'top_10_percent_mean': float(torch.topk(heatmap[0].flatten(), k=int(0.1 * heatmap[0].numel()))[0].mean())
        },
        'integrated_gradients': {
            'channel_importance': {
                f'channel_{i}': float(attribution[0, i].abs().sum())
                for i in range(4)
            }
        }
    }

    # Save summary
    summary_path = output_dir / f"{image_path.stem}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")
    print(f"✓ All visualizations saved to {output_dir}")


def analyze_generation_steps(
    prompt: str,
    classifier,
    output_dir: Path,
    num_steps: int = 50,
    target_class: int = 2,
    device: str = "cuda",
    model_id: str = "CompVis/stable-diffusion-v1-4"
):
    """
    Analyze classifier attention during the generation process.

    Args:
        prompt: Text prompt for generation
        classifier: Loaded classifier model
        output_dir: Directory to save results
        num_steps: Number of denoising steps
        target_class: Class to analyze
        device: Device
        model_id: Stable Diffusion model ID
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing Generation Process")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Load pipeline
    print("\nLoading Stable Diffusion pipeline...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)

    # Storage for analysis
    step_heatmaps = []
    step_predictions = []

    # Callback to capture intermediate latents
    def analysis_callback(pipe, step_index, timestep, callback_kwargs):
        latent = callback_kwargs["latents"]

        # Convert latent to float32 for classifier
        latent = latent.float()

        # Run Grad-CAM
        gradcam = ClassifierGradCAM(classifier, target_layer_name="encoder_model.middle_block.2")

        with torch.no_grad():
            timestep_tensor = torch.tensor([timestep], device=device, dtype=torch.long)
            if timestep_tensor.shape[0] != latent.shape[0]:
                timestep_tensor = timestep_tensor.repeat(latent.shape[0])

            heatmap, info = gradcam.generate_heatmap(
                latent, timestep_tensor, target_class=target_class
            )

        gradcam.remove_hooks()

        # Store results
        step_heatmaps.append(heatmap[0].cpu())
        step_predictions.append({
            'step': step_index,
            'timestep': int(timestep),
            'probs': info['probs'][0].cpu().tolist()
        })

        # Save visualization every 10 steps
        if step_index % 10 == 0:
            save_path = output_dir / f"step_{step_index:03d}.png"
            VisualizationUtils.save_heatmap_comparison(
                latent[0], heatmap[0], info,
                save_path=save_path,
                title=f"Step {step_index} / Timestep {timestep}"
            )

        return callback_kwargs

    # Generate with callback
    print("\nGenerating and analyzing...")
    result = pipe(
        prompt,
        num_inference_steps=num_steps,
        callback_on_step_end=analysis_callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )

    # Save final image
    final_image_path = output_dir / "final_image.png"
    result.images[0].save(final_image_path)
    print(f"\n✓ Final image saved to {final_image_path}")

    # Create evolution video/gif
    print("\nCreating heatmap evolution visualization...")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        ax.imshow(step_heatmaps[frame].numpy(), cmap='jet')
        ax.set_title(f"Step {step_predictions[frame]['step']}\n"
                    f"Nude Prob: {step_predictions[frame]['probs'][2]:.3f}")
        ax.axis('off')

    anim = animation.FuncAnimation(fig, update, frames=len(step_heatmaps), interval=200)
    anim_path = output_dir / "heatmap_evolution.gif"
    anim.save(anim_path, writer='pillow', fps=5)
    plt.close()

    print(f"✓ Heatmap evolution saved to {anim_path}")

    # Save prediction trajectory
    trajectory_path = output_dir / "prediction_trajectory.json"
    with open(trajectory_path, 'w') as f:
        json.dump(step_predictions, f, indent=2)

    # Plot probability evolution
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = [p['step'] for p in step_predictions]
    not_people = [p['probs'][0] for p in step_predictions]
    clothed = [p['probs'][1] for p in step_predictions]
    nude = [p['probs'][2] for p in step_predictions]

    ax.plot(steps, not_people, label='Not People', marker='o')
    ax.plot(steps, clothed, label='Clothed', marker='s')
    ax.plot(steps, nude, label='Nude', marker='^', linewidth=2)

    ax.set_xlabel('Denoising Step')
    ax.set_ylabel('Probability')
    ax.set_title(f'Classifier Predictions During Generation\nPrompt: "{prompt}"')
    ax.legend()
    ax.grid(True, alpha=0.3)

    prob_path = output_dir / "probability_evolution.png"
    plt.savefig(prob_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Probability evolution saved to {prob_path}")
    print(f"\n✓ Generation analysis complete! Results in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Interpret nudity classifier")

    # Input modes
    parser.add_argument("--mode", choices=["image", "generation"], default="image",
                       help="Analysis mode: 'image' or 'generation'")
    parser.add_argument("--image_path", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Directory of images")

    # Generation mode
    parser.add_argument("--prompt", type=str, help="Prompt for generation mode")
    parser.add_argument("--num_steps", type=int, default=50, help="Denoising steps")

    # Model paths
    parser.add_argument("--classifier_ckpt", type=str,
                       default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                       help="Path to classifier checkpoint")
    parser.add_argument("--sd_model", type=str,
                       default="CompVis/stable-diffusion-v1-4",
                       help="Stable Diffusion model ID")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")

    # Parameters
    parser.add_argument("--timestep", type=int, default=500,
                       help="Timestep for image mode (0-999)")
    parser.add_argument("--target_class", type=int, default=2,
                       help="Target class (0: not people, 1: clothed, 2: nude)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load classifier
    print("Loading classifier...")
    classifier = load_classifier_for_interpretation(args.classifier_ckpt, args.device)
    print(f"✓ Classifier loaded from {args.classifier_ckpt}")

    output_dir = Path(args.output_dir)

    if args.mode == "image":
        # Load VAE
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            args.sd_model,
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(args.device)

        # Process images
        if args.image_path:
            analyze_single_image(
                Path(args.image_path), classifier, vae,
                output_dir, args.timestep, args.target_class, args.device
            )
        elif args.image_dir:
            image_dir = Path(args.image_dir)
            image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

            print(f"\nFound {len(image_paths)} images")

            for image_path in tqdm(image_paths, desc="Processing images"):
                img_output_dir = output_dir / image_path.stem
                analyze_single_image(
                    image_path, classifier, vae,
                    img_output_dir, args.timestep, args.target_class, args.device
                )
        else:
            print("Error: Must provide --image_path or --image_dir for image mode")

    elif args.mode == "generation":
        if not args.prompt:
            print("Error: Must provide --prompt for generation mode")
            return

        analyze_generation_steps(
            args.prompt, classifier, output_dir,
            args.num_steps, args.target_class, args.device, args.sd_model
        )


if __name__ == "__main__":
    main()
