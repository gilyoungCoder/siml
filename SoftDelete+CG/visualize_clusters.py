"""
Cluster Visualization Script
각 cluster에 속한 latent를 완전히 denoise한 후 VAE decoder로 복원하여 시각화

noisy latent (t>0)를 받아서 t=0까지 denoising 후 깨끗한 이미지 생성
"""

import os
import argparse
import numpy as np
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionPipeline, DPMSolverMultistepScheduler

from geo_utils.mode_aware_gradient_model import ClusterManager


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize cluster latents')

    parser.add_argument('--latents_path', type=str, required=True,
                        help='Path to collected latents (e.g., cluster_centroids/violence_clusters_latents.pt)')
    parser.add_argument('--centroids_path', type=str, required=True,
                        help='Path to cluster centroids (e.g., cluster_centroids/violence_clusters.pt)')
    parser.add_argument('--output_dir', type=str, default='cluster_visualizations',
                        help='Output directory for visualizations')

    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4',
                        help='Path to Stable Diffusion model')
    parser.add_argument('--samples_per_cluster', type=int, default=5,
                        help='Number of samples to visualize per cluster')
    parser.add_argument('--latent_timestep', type=int, default=500,
                        help='Timestep at which latents were captured (for denoising)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def denoise_latents(pipe, latents: torch.Tensor, start_timestep: int, device: str, prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Denoise latents from start_timestep to t=0 using the UNet"""
    scheduler = pipe.scheduler
    unet = pipe.unet

    # Set timesteps starting from start_timestep
    scheduler.set_timesteps(50)  # Use 50 steps for denoising

    # Find the starting index in the timestep schedule
    timesteps = scheduler.timesteps
    start_idx = 0
    for i, t in enumerate(timesteps):
        if t <= start_timestep:
            start_idx = i
            break

    # Convert to fp16 if needed
    latents = latents.to(device=device, dtype=unet.dtype)

    # Denoise from start_timestep to t=0
    with torch.no_grad():
        for t in timesteps[start_idx:]:
            # Predict noise
            encoder_hidden_states = prompt_embeds.expand(latents.shape[0], -1, -1)
            noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to images using VAE"""
    # Scale latents for VAE decoding
    latents = latents / vae.config.scaling_factor

    with torch.no_grad():
        images = vae.decode(latents).sample

    # Convert to [0, 1] range
    images = (images / 2 + 0.5).clamp(0, 1)

    return images


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    # Expect tensor of shape (C, H, W) with values in [0, 1]
    tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Convert to numpy and PIL
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def create_grid(images: list, cols: int = 5) -> Image.Image:
    """Create a grid of images"""
    if len(images) == 0:
        return None

    # Get dimensions from first image
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols

    # Create grid
    grid = Image.new('RGB', (cols * w, rows * h), color='white')

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))

    return grid


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("CLUSTER VISUALIZATION (with full denoising)")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load latents
    print(f"\n[Loading latents from {args.latents_path}]")
    latents_data = torch.load(args.latents_path, map_location='cpu')

    # Handle both single tensor and multi-timestep dict formats
    if isinstance(latents_data, dict):
        # Multi-timestep format - use the smallest timestep for cleaner visualization
        available_timesteps = sorted(latents_data.keys())
        vis_timestep = available_timesteps[0]  # Use smallest (cleanest)
        latents = latents_data[vis_timestep]
        print(f"  Multi-timestep latents detected. Using t={vis_timestep} for visualization.")
        print(f"  Available timesteps: {available_timesteps}")
    else:
        latents = latents_data
        vis_timestep = args.latent_timestep

    print(f"  Loaded {latents.shape[0]} latents of shape {latents.shape[1:]}")
    print(f"  Latent timestep: t={vis_timestep}")

    # Load cluster manager
    print(f"\n[Loading cluster centroids from {args.centroids_path}]")
    cluster_manager = ClusterManager()
    cluster_manager.load(args.centroids_path)

    # Assign latents to clusters
    labels = cluster_manager.predict(latents)
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"\n[Cluster distribution]")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} samples")

    # Load Stable Diffusion pipeline for denoising
    print(f"\n[Loading Stable Diffusion pipeline from {args.model_path}]")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        safety_checker=None,
        torch_dtype=torch.float16 if 'cuda' in args.device else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    vae = pipe.vae
    vae.eval()

    # Pre-compute empty prompt embeddings for unconditional denoising
    print(f"  Computing prompt embeddings...")
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(
            pipe.tokenizer(
                "",
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(args.device)
        )[0]

    # Visualize each cluster
    print(f"\n[Generating visualizations (denoising from t={vis_timestep} to t=0)]")
    all_cluster_images = []

    for cluster_id in tqdm(range(cluster_manager.n_clusters), desc="Clusters"):
        mask = labels == cluster_id
        cluster_latents = latents[mask]

        if len(cluster_latents) == 0:
            print(f"  Cluster {cluster_id}: No samples")
            continue

        # Select samples (closest to centroid)
        distances = cluster_manager.get_distances(cluster_latents)[:, cluster_id]
        sorted_indices = torch.argsort(distances)

        n_samples = min(args.samples_per_cluster, len(cluster_latents))
        selected_indices = sorted_indices[:n_samples]
        selected_latents = cluster_latents[selected_indices]

        # Denoise and decode latents
        cluster_images = []
        for i in range(len(selected_latents)):
            latent = selected_latents[i:i+1]

            # Denoise from vis_timestep to t=0
            denoised_latent = denoise_latents(pipe, latent, vis_timestep, args.device, prompt_embeds)

            # Decode to image
            image_tensor = decode_latents(vae, denoised_latent)
            pil_image = tensor_to_pil(image_tensor)
            cluster_images.append(pil_image)

        # Save individual cluster grid
        if cluster_images:
            grid = create_grid(cluster_images, cols=min(5, len(cluster_images)))
            grid_path = os.path.join(args.output_dir, f"cluster_{cluster_id:02d}.png")
            grid.save(grid_path)

            all_cluster_images.append((cluster_id, cluster_images[0]))  # First image for summary

    # Create summary grid (one image per cluster)
    print(f"\n[Creating summary grid]")
    summary_images = [img for _, img in sorted(all_cluster_images)]
    if summary_images:
        summary_grid = create_grid(summary_images, cols=5)
        summary_path = os.path.join(args.output_dir, "summary_all_clusters.png")
        summary_grid.save(summary_path)
        print(f"  Saved summary to {summary_path}")

    print(f"\n[DONE] Visualizations saved to {args.output_dir}/")
    print(f"  - Individual cluster grids: cluster_XX.png")
    print(f"  - Summary grid: summary_all_clusters.png")


if __name__ == "__main__":
    main()
