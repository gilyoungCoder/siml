"""
4-Class Classifier Grad-CAM Visualization Script

Classes:
    0: Benign (not people)
    1: Person (clothed)
    2: Nude (harmful)
    3: Harm_color (color artifacts)

Usage:
    # Single image
    python interpret_classifier_4class.py --image_path ./test.png --output_dir ./gradcam_vis_4class

    # Directory of images
    python interpret_classifier_4class.py --image_dir ./test_images --output_dir ./gradcam_vis_4class

    # Generation mode (analyze during diffusion)
    python interpret_classifier_4class.py --mode generation --prompt "a person" --output_dir ./gradcam_vis_4class
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
)
from geo_models.classifier.classifier import load_discriminator

CLASS_NAMES = ['Benign', 'Person', 'Nude', 'Harm_color']


def load_classifier_4class(ckpt_path: str, device: str = "cuda"):
    classifier = load_discriminator(
        ckpt_path=ckpt_path,
        condition=None,
        eval=True,
        channel=4,
        num_classes=4
    )
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def encode_image_to_latent(image_path: Path, vae, device: str = "cuda"):
    """Returns (latent, original_image_array)"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    original_image = np.array(image)  # Keep original for visualization

    image_tensor = torch.from_numpy(original_image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = (image_tensor - 0.5) * 2
    image_tensor = image_tensor.to(device)
    if vae.dtype == torch.float16:
        image_tensor = image_tensor.half()
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
    gradcam_layer: str = "encoder_model.middle_block.2",
    device: str = "cuda"
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path.name}")
    print(f"{'='*60}")

    latent, original_image = encode_image_to_latent(image_path, vae, device)
    latent = latent.float()
    timestep_tensor = torch.tensor([timestep], device=device, dtype=torch.long)

    # === Grad-CAM ===
    print("\n[1/3] Running Grad-CAM...")
    gradcam = ClassifierGradCAM(classifier, target_layer_name=gradcam_layer)
    heatmap, info = gradcam.generate_heatmap(latent, timestep_tensor, target_class=target_class)

    probs = info['probs'][0].cpu().numpy()
    print(f"  Prediction: {CLASS_NAMES[probs.argmax()]} ({probs.max():.3f})")
    print(f"  Probabilities: {dict(zip(CLASS_NAMES, probs))}")

    # Save with original image overlay (color)
    gradcam_path = output_dir / f"{image_path.stem}_gradcam_class{target_class}.png"
    VisualizationUtils.save_heatmap_with_image(
        original_image, heatmap[0], info,
        save_path=gradcam_path,
        title=f"Grad-CAM: {CLASS_NAMES[target_class]} Detection"
    )
    gradcam.remove_hooks()

    # === Layer-wise Activation ===
    print("\n[2/3] Analyzing layer-wise activations...")
    layer_analyzer = LayerwiseActivationAnalyzer(classifier)
    layer_analyzer.register_layer_hooks()
    activation_maps = layer_analyzer.analyze(latent, timestep_tensor)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    layer_names = [k for k in activation_maps.keys() if k not in ['logits', 'probs']]
    for idx, layer_name in enumerate(layer_names[:6]):
        acts = activation_maps[layer_name][0]
        mean_acts = acts.mean(dim=(1, 2)).cpu().numpy()
        axes[idx].bar(range(len(mean_acts)), mean_acts)
        axes[idx].set_title(f"{layer_name}\nShape: {acts.shape}")
        axes[idx].set_xlabel('Channel')
        axes[idx].set_ylabel('Mean Activation')
    plt.tight_layout()
    layer_path = output_dir / f"{image_path.stem}_layers.png"
    plt.savefig(layer_path, dpi=300, bbox_inches='tight')
    plt.close()
    layer_analyzer.remove_hooks()

    # === Integrated Gradients ===
    print("\n[3/3] Computing Integrated Gradients...")
    ig = IntegratedGradients(classifier)
    attribution = ig.attribute(latent, timestep_tensor, target_class=target_class, n_steps=50)
    fig = VisualizationUtils.visualize_attribution_channels(attribution[0])
    ig_path = output_dir / f"{image_path.stem}_integrated_gradients.png"
    plt.savefig(ig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # === Summary ===
    summary = {
        'image': str(image_path),
        'timestep': timestep,
        'target_class': target_class,
        'target_class_name': CLASS_NAMES[target_class],
        'predictions': {
            'logits': info['logits'][0].cpu().tolist(),
            'probs': probs.tolist(),
            'predicted_class': int(probs.argmax()),
            'predicted_class_name': CLASS_NAMES[probs.argmax()]
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
    summary_path = output_dir / f"{image_path.stem}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


def analyze_generation_steps(
    prompt: str,
    classifier,
    output_dir: Path,
    num_steps: int = 50,
    target_class: int = 2,
    gradcam_layer: str = "encoder_model.middle_block.2",
    device: str = "cuda",
    model_id: str = "CompVis/stable-diffusion-v1-4"
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing Generation Process")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    pipe = CustomStableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    step_heatmaps = []
    step_predictions = []

    def analysis_callback(pipe, step_index, timestep, callback_kwargs):
        latent = callback_kwargs["latents"].float()
        gradcam = ClassifierGradCAM(classifier, target_layer_name=gradcam_layer)
        with torch.no_grad():
            timestep_tensor = torch.tensor([timestep], device=device, dtype=torch.long)
            if timestep_tensor.shape[0] != latent.shape[0]:
                timestep_tensor = timestep_tensor.repeat(latent.shape[0])
            heatmap, info = gradcam.generate_heatmap(latent, timestep_tensor, target_class=target_class)
        gradcam.remove_hooks()

        step_heatmaps.append(heatmap[0].cpu())
        step_predictions.append({
            'step': step_index,
            'timestep': int(timestep),
            'probs': info['probs'][0].cpu().tolist()
        })

        if step_index % 10 == 0:
            save_path = output_dir / f"step_{step_index:03d}.png"
            VisualizationUtils.save_heatmap_comparison(
                latent[0], heatmap[0], info,
                save_path=save_path,
                title=f"Step {step_index} / Timestep {timestep}"
            )
        return callback_kwargs

    result = pipe(
        prompt,
        num_inference_steps=num_steps,
        callback_on_step_end=analysis_callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )

    final_image_path = output_dir / "final_image.png"
    result.images[0].save(final_image_path)

    # Heatmap evolution GIF
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        ax.imshow(step_heatmaps[frame].numpy(), cmap='jet')
        probs = step_predictions[frame]['probs']
        ax.set_title(
            f"Step {step_predictions[frame]['step']}\n"
            f"Nude: {probs[2]:.3f} | Harm_color: {probs[3]:.3f}"
        )
        ax.axis('off')

    anim = animation.FuncAnimation(fig, update, frames=len(step_heatmaps), interval=200)
    anim.save(output_dir / "heatmap_evolution.gif", writer='pillow', fps=5)
    plt.close()

    # Probability evolution plot
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = [p['step'] for p in step_predictions]
    for i, name in enumerate(CLASS_NAMES):
        vals = [p['probs'][i] for p in step_predictions]
        ax.plot(steps, vals, label=name, marker='os^d'[i], linewidth=2 if i >= 2 else 1)
    ax.set_xlabel('Denoising Step')
    ax.set_ylabel('Probability')
    ax.set_title(f'4-Class Predictions During Generation\nPrompt: "{prompt}"')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "probability_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()

    with open(output_dir / "prediction_trajectory.json", 'w') as f:
        json.dump(step_predictions, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="4-Class Classifier Grad-CAM Visualization")

    parser.add_argument("--mode", choices=["image", "generation"], default="image")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--num_steps", type=int, default=50)

    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth")
    parser.add_argument("--sd_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--target_class", type=int, default=2,
                        help="0: Benign, 1: Person, 2: Nude, 3: Harm_color")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("Loading 4-class classifier...")
    classifier = load_classifier_4class(args.classifier_ckpt, args.device)
    print(f"Loaded from {args.classifier_ckpt}")

    output_dir = Path(args.output_dir)

    if args.mode == "image":
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            args.sd_model, subfolder="vae", torch_dtype=torch.float16
        ).to(args.device)

        if args.image_path:
            analyze_single_image(
                Path(args.image_path), classifier, vae,
                output_dir, args.timestep, args.target_class,
                args.gradcam_layer, args.device
            )
        elif args.image_dir:
            image_dir = Path(args.image_dir)
            image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            print(f"\nFound {len(image_paths)} images")
            for image_path in tqdm(image_paths, desc="Processing"):
                img_output_dir = output_dir / image_path.stem
                analyze_single_image(
                    image_path, classifier, vae,
                    img_output_dir, args.timestep, args.target_class,
                    args.gradcam_layer, args.device
                )
        else:
            print("Error: Must provide --image_path or --image_dir for image mode")

    elif args.mode == "generation":
        if not args.prompt:
            print("Error: Must provide --prompt for generation mode")
            return
        analyze_generation_steps(
            args.prompt, classifier, output_dir,
            args.num_steps, args.target_class,
            args.gradcam_layer, args.device, args.sd_model
        )


if __name__ == "__main__":
    main()
