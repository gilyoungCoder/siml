"""
Quick Interpretation Example

Minimal example showing how to use the interpretation tools.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from geo_utils.classifier_interpretability import (
    ClassifierGradCAM,
    LayerwiseActivationAnalyzer,
    IntegratedGradients,
    VisualizationUtils,
    load_classifier_for_interpretation
)
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np


def main():
    """
    Quick example: Load an image, encode to latent, and visualize
    what the classifier focuses on.
    """

    # Configuration
    CLASSIFIER_CKPT = "./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
    IMAGE_PATH = "./test_image.png"  # Replace with your image
    OUTPUT_DIR = Path("./interpretation_results")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("Quick Classifier Interpretation")
    print("="*60)

    # Load classifier
    print("\n[1/4] Loading classifier...")
    classifier = load_classifier_for_interpretation(CLASSIFIER_CKPT, DEVICE)
    print(f"✓ Classifier loaded")

    # Load VAE
    print("\n[2/4] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(DEVICE)
    print("✓ VAE loaded")

    # Encode image
    print("\n[3/4] Encoding image to latent...")
    image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = (image_tensor - 0.5) * 2
    image_tensor = image_tensor.to(DEVICE)

    # Match VAE dtype
    if vae.dtype == torch.float16:
        image_tensor = image_tensor.half()

    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor

    # Convert latent to float32 for classifier
    latent = latent.float()

    print(f"✓ Latent shape: {latent.shape}")

    # Setup Grad-CAM
    print("\n[4/4] Running Grad-CAM analysis...")
    gradcam = ClassifierGradCAM(
        classifier,
        target_layer_name="encoder_model.middle_block.2"  # Bottleneck (highest semantic level)
    )

    # Analyze at timestep 500 (mid-diffusion)
    timestep = torch.tensor([500], device=DEVICE, dtype=torch.long)

    # Generate heatmap for nude class (class 2)
    heatmap, info = gradcam.generate_heatmap(
        latent,
        timestep,
        target_class=2  # Nude class
    )

    # Print results
    probs = info['probs'][0].cpu().numpy()
    class_names = ['Not People', 'Clothed', 'Nude']

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {class_names[probs.argmax()]}")
    print(f"Confidence: {probs.max():.3f}\n")
    print("Class Probabilities:")
    for name, prob in zip(class_names, probs):
        print(f"  {name:12s}: {prob:.3f} {'█' * int(prob * 50)}")

    print(f"\nHeatmap Statistics:")
    print(f"  Max attention:  {heatmap[0].max():.3f}")
    print(f"  Mean attention: {heatmap[0].mean():.3f}")

    # Save visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gradcam_result.png"

    VisualizationUtils.save_heatmap_comparison(
        latent[0],
        heatmap[0],
        info,
        save_path=output_path,
        title="Classifier Grad-CAM: Nude Detection"
    )

    print(f"\n✓ Visualization saved to: {output_path}")
    print("\n" + "="*60)
    print("Interpretation:")
    print("="*60)
    print("The heatmap shows which regions of the latent space the classifier")
    print("focuses on when determining if the image contains nude content.")
    print("\nRed/bright areas = High attention (important for nude detection)")
    print("Blue/dark areas = Low attention (less relevant)")
    print("="*60)

    # Cleanup
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
