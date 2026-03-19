"""
Violence Latent Clustering Script

이 스크립트는 Mode-Aware Guidance를 위한 cluster centroids를 생성합니다.

사용법:
    python cluster_violence_latents.py \
        --ckpt_path /path/to/geodiffusion \
        --prompts_file violence_prompts.txt \
        --output_path cluster_centroids/violence_clusters.pt \
        --n_clusters 10 \
        --num_samples_per_prompt 10

또는 기존 latent 데이터가 있다면:
    python cluster_violence_latents.py \
        --latents_path collected_latents.pt \
        --output_path cluster_centroids/violence_clusters.pt \
        --n_clusters 10
"""

import os
import argparse
import numpy as np
from typing import List, Optional
from tqdm import tqdm

import torch
from diffusers import AutoencoderKL, PNDMScheduler, DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.mode_aware_gradient_model import ClusterManager, get_predicted_x0


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster violence latents for mode-aware guidance')

    # Model paths
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to GeoDiffusion checkpoint for generating samples')

    # Input options (choose one)
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='Text file with violence prompts (one per line)')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Violence prompts directly as arguments')
    parser.add_argument('--latents_path', type=str, default=None,
                        help='Pre-collected latents tensor path (skip generation)')

    # Generation options
    parser.add_argument('--num_samples_per_prompt', type=int, default=10,
                        help='Number of samples to generate per prompt')
    parser.add_argument('--capture_timestep', type=int, default=500,
                        help='Timestep at which to capture latents (0-1000)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--height', type=int, default=512,
                        help='Generated image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Generated image width')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')

    # Clustering options
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'flatten', 'attention'],
                        help='Pooling method for latent features')

    # Output
    parser.add_argument('--output_path', type=str, default='cluster_centroids/violence_clusters.pt',
                        help='Path to save cluster centroids')
    parser.add_argument('--save_latents', action='store_true',
                        help='Also save collected latents for future use')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def load_prompts(args) -> List[str]:
    """Load violence prompts from file or arguments"""
    if args.prompts is not None:
        return args.prompts

    if args.prompts_file is not None:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts

    # Default violence prompts if none provided
    default_prompts = [
        "a violent scene with guns",
        "a person holding a knife threateningly",
        "a fist fight between two people",
        "an aggressive confrontation",
        "a person with a weapon",
        "a violent street fight",
        "an armed robbery scene",
        "people fighting violently",
        "a threatening situation with weapons",
        "an assault scene",
    ]
    print("[Warning] No prompts provided, using default violence prompts")
    return default_prompts


def load_pipeline(ckpt_path: str, device: str) -> CustomStableDiffusionPipeline:
    """Load Stable Diffusion pipeline"""
    print(f"Loading pipeline from {ckpt_path}...")

    tokenizer = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder")

    pipe = CustomStableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=DPMSolverMultistepScheduler.from_config(ckpt_path, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=CLIPFeatureExtractor.from_pretrained(ckpt_path, subfolder="feature_extractor"),
    )

    pipe = pipe.to(device)
    return pipe


def collect_latents(
    pipe: CustomStableDiffusionPipeline,
    prompts: List[str],
    num_samples_per_prompt: int,
    capture_timestep: int,
    num_inference_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    device: str,
    seed: int,
) -> torch.Tensor:
    """
    Generate images and collect latents at specified timestep
    """
    collected_latents = []
    capture_step = int((1 - capture_timestep / 1000) * num_inference_steps)

    print(f"\n[Collecting Latents]")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Samples per prompt: {num_samples_per_prompt}")
    print(f"  Capture timestep: {capture_timestep} (step {capture_step}/{num_inference_steps})")
    print(f"  Total samples: {len(prompts) * num_samples_per_prompt}")

    total_samples = len(prompts) * num_samples_per_prompt
    pbar = tqdm(total=total_samples, desc="Generating samples")

    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in range(num_samples_per_prompt):
            # Set seed for reproducibility
            current_seed = seed + prompt_idx * 1000 + sample_idx
            torch.manual_seed(current_seed)

            # Capture latent at target step
            captured_latent = [None]

            def capture_callback(pipe_obj, step, t, callback_kwargs):
                if step == capture_step:
                    noise_pred = callback_kwargs['noise_pred']
                    prev_latents = callback_kwargs['prev_latents']
                    x0 = get_predicted_x0(pipe_obj, prev_latents, noise_pred, t)
                    captured_latent[0] = x0.detach().cpu()
                return callback_kwargs

            try:
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=capture_callback,
                    callback_on_step_end_tensor_inputs=['noise_pred', 'prev_latents'],
                )

                if captured_latent[0] is not None:
                    collected_latents.append(captured_latent[0])
                else:
                    print(f"\n  [Warning] Failed to capture latent for prompt {prompt_idx}, sample {sample_idx}")

            except Exception as e:
                print(f"\n  [Error] Generation failed: {e}")

            pbar.update(1)

    pbar.close()

    if len(collected_latents) == 0:
        raise RuntimeError("No latents collected!")

    latents_tensor = torch.cat(collected_latents, dim=0)
    print(f"\n[Collected] {latents_tensor.shape[0]} latents of shape {latents_tensor.shape[1:]}")

    return latents_tensor


def analyze_clusters(cluster_manager: ClusterManager, latents: torch.Tensor, labels: np.ndarray):
    """Analyze and print cluster statistics"""
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS")
    print("=" * 60)

    n_clusters = cluster_manager.n_clusters
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\nCluster Distribution (Total: {len(labels)} samples)")
    print("-" * 40)
    for cluster_id in range(n_clusters):
        count = counts[unique == cluster_id][0] if cluster_id in unique else 0
        pct = 100 * count / len(labels)
        bar = "#" * int(pct / 2)
        print(f"  Cluster {cluster_id:2d}: {count:4d} ({pct:5.1f}%) {bar}")

    # Compute within-cluster variance
    print(f"\nWithin-Cluster Statistics")
    print("-" * 40)

    features = cluster_manager.pool_features(latents)
    centroids = cluster_manager.centroids

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.sum() == 0:
            continue

        cluster_features = features[mask]
        centroid = centroids[cluster_id]

        # Distance to centroid
        distances = torch.norm(cluster_features - centroid, dim=1)
        mean_dist = distances.mean().item()
        std_dist = distances.std().item()

        print(f"  Cluster {cluster_id:2d}: mean_dist={mean_dist:.4f}, std={std_dist:.4f}")

    print("\n" + "=" * 60)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 60)
    print("""
1. Visualize samples from each cluster to understand what each mode represents
2. Adjust mode_aware schedules in config based on cluster characteristics
3. For clusters with high variance, consider using stricter thresholds
4. For dominant clusters, consider splitting further or adjusting scale
""")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load or generate latents
    if args.latents_path is not None and os.path.exists(args.latents_path):
        print(f"[Loading latents from {args.latents_path}]")
        latents = torch.load(args.latents_path, map_location='cpu')
        print(f"  Loaded {latents.shape[0]} latents of shape {latents.shape[1:]}")
    else:
        if args.ckpt_path is None:
            raise ValueError("Either --latents_path or --ckpt_path must be provided")

        # Load pipeline
        pipe = load_pipeline(args.ckpt_path, args.device)

        # Load prompts
        prompts = load_prompts(args)

        # Collect latents
        latents = collect_latents(
            pipe=pipe,
            prompts=prompts,
            num_samples_per_prompt=args.num_samples_per_prompt,
            capture_timestep=args.capture_timestep,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            device=args.device,
            seed=args.seed,
        )

        # Optionally save latents
        if args.save_latents:
            latents_save_path = args.output_path.replace('.pt', '_latents.pt')
            os.makedirs(os.path.dirname(latents_save_path), exist_ok=True)
            torch.save(latents, latents_save_path)
            print(f"[Saved] Latents to {latents_save_path}")

    # Perform clustering
    print(f"\n[Clustering with K={args.n_clusters}]")
    cluster_manager = ClusterManager(
        n_clusters=args.n_clusters,
        pooling=args.pooling,
    )

    labels = cluster_manager.fit(latents, save_path=args.output_path)

    # Analyze clusters
    analyze_clusters(cluster_manager, latents, labels)

    print(f"\n[Done] Centroids saved to {args.output_path}")
    print(f"Use this path in your mode_aware_discriminator.yaml config:")
    print(f"  centroids_path: \"{args.output_path}\"")


if __name__ == "__main__":
    main()
