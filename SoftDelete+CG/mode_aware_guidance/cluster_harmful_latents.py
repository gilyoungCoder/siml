"""
Harmful Latent Clustering Script for Mode-Aware Guidance

이 스크립트는 Mode-Aware 3-way Classifier Guidance를 위한 cluster centroids를 생성합니다.

3-way Classifier:
- Class 0: benign (not relevant)
- Class 1: safe (clothed)
- Class 2: harmful (nude)

사용법:
    # 방법 1: Harmful prompt로 이미지 생성하면서 latent 수집
    python cluster_harmful_latents.py \
        --mode generate \
        --ckpt_path runwayml/stable-diffusion-v1-5 \
        --classifier_ckpt ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth \
        --prompts_file harmful_prompts.txt \
        --output_path cluster_centroids/harmful_clusters.pt \
        --n_clusters 10

    # 방법 2: 기존 latent 데이터로부터 clustering
    python cluster_harmful_latents.py \
        --mode fit \
        --latents_path collected_harmful_latents.pt \
        --output_path cluster_centroids/harmful_clusters.pt \
        --n_clusters 10
"""

import os
import argparse
import numpy as np
from typing import List, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler

from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.mode_aware_gradient_model import (
    ClusterManager,
    fit_harmful_clusters
)
from geo_models.classifier.classifier import load_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster harmful latents for mode-aware guidance')

    parser.add_argument('--mode', type=str, choices=['generate', 'fit'], default='generate',
                        help='Mode: generate (collect + fit) or fit (from existing latents)')

    # Model paths
    parser.add_argument('--ckpt_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Path to Stable Diffusion checkpoint')
    parser.add_argument('--classifier_ckpt', type=str,
                        default='./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth',
                        help='Path to 3-way classifier checkpoint')

    # Input options
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='Text file with harmful prompts (one per line)')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Harmful prompts directly as arguments')
    parser.add_argument('--latents_path', type=str, default=None,
                        help='Pre-collected latents tensor path (for fit mode)')

    # Generation options
    parser.add_argument('--num_samples_per_prompt', type=int, default=10,
                        help='Number of samples to generate per prompt')
    parser.add_argument('--capture_timesteps', type=int, nargs='+',
                        default=[900, 800, 700, 600, 500, 400, 300, 200, 100],
                        help='Timesteps at which to capture latents (0-1000), 9 values recommended')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--harmful_class', type=int, default=2,
                        help='Class index for harmful content')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                        help='Minimum harmful probability to include sample')

    # Clustering options
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'flatten', 'attention'],
                        help='Pooling method for latent features')

    # Output
    parser.add_argument('--output_path', type=str, default='cluster_centroids/harmful_clusters.pt',
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
    """Load harmful prompts from file or arguments"""
    if args.prompts is not None:
        return args.prompts

    if args.prompts_file is not None:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts

    # Default harmful prompts if none provided
    # WARNING: These are intentionally vague - replace with your actual prompts
    default_prompts = [
        "a person without clothes",
        "nude figure",
        "unclothed person",
        "naked body",
        "explicit scene",
        "person undressing",
        "revealing outfit",
        "bare skin exposure",
        "intimate scene",
        "provocative pose",
    ]
    print("[Warning] No prompts provided, using default prompts")
    print("  For better results, provide domain-specific prompts via --prompts_file")
    return default_prompts


def load_pipeline(ckpt_path: str, device: str) -> CustomStableDiffusionPipeline:
    """Load Stable Diffusion pipeline"""
    print(f"[INFO] Loading pipeline from {ckpt_path}...")

    pipe = CustomStableDiffusionPipeline.from_pretrained(
        ckpt_path,
        safety_checker=None,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    return pipe


def collect_harmful_latents(
    pipe: CustomStableDiffusionPipeline,
    classifier_model: torch.nn.Module,
    prompts: List[str],
    num_samples_per_prompt: int,
    capture_timesteps: List[int],
    num_inference_steps: int,
    guidance_scale: float,
    harmful_class: int,
    confidence_threshold: float,
    device: str,
    seed: int,
) -> dict:
    """Generate images and collect harmful latents at multiple timesteps

    Returns:
        dict: {timestep: torch.Tensor} mapping each timestep to collected latents
    """
    # Calculate capture steps for each timestep
    capture_steps = {
        t: int((1 - t / 1000) * num_inference_steps)
        for t in capture_timesteps
    }

    # Initialize storage for each timestep
    collected_latents = {t: [] for t in capture_timesteps}

    print(f"\n[Collecting Harmful Latents at Multiple Timesteps]")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Samples per prompt: {num_samples_per_prompt}")
    print(f"  Capture timesteps: {capture_timesteps}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Total attempts: {len(prompts) * num_samples_per_prompt}")

    classifier_model.eval()
    total_collected = {t: 0 for t in capture_timesteps}
    total_attempts = 0

    pbar = tqdm(total=len(prompts) * num_samples_per_prompt, desc="Generating")

    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in range(num_samples_per_prompt):
            current_seed = seed + prompt_idx * 1000 + sample_idx
            torch.manual_seed(current_seed)

            captured = {t: None for t in capture_timesteps}

            def capture_callback(pipe_obj, step, t, callback_kwargs):
                # Check if current step matches any capture step
                for target_t, target_step in capture_steps.items():
                    if step == target_step:
                        latents = callback_kwargs.get('latents')
                        if latents is not None:
                            with torch.no_grad():
                                # Classify current latent
                                ts = torch.tensor([t], device=device).expand(latents.size(0))
                                latents_float = latents.float()
                                logits = classifier_model(latents_float, ts)
                                probs = F.softmax(logits, dim=-1)
                                harmful_prob = probs[:, harmful_class].mean().item()

                                if harmful_prob > confidence_threshold:
                                    captured[target_t] = latents_float.detach().cpu()
                return callback_kwargs

            try:
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=capture_callback,
                    callback_on_step_end_tensor_inputs=['latents'],
                )

                total_attempts += 1
                for t in capture_timesteps:
                    if captured[t] is not None:
                        collected_latents[t].append(captured[t])
                        total_collected[t] += 1

            except Exception as e:
                print(f"\n  [Warning] Generation failed: {e}")

            pbar.update(1)
            avg_collected = sum(total_collected.values()) / len(capture_timesteps)
            pbar.set_postfix({'avg_collected': f'{avg_collected:.0f}', 'attempts': total_attempts})

    pbar.close()

    # Convert to tensors
    result = {}
    print(f"\n[Collection Summary]")
    for t in capture_timesteps:
        if len(collected_latents[t]) > 0:
            result[t] = torch.cat(collected_latents[t], dim=0)
            rate = 100 * len(collected_latents[t]) / max(1, total_attempts)
            print(f"  t={t:4d}: {result[t].shape[0]:4d} latents ({rate:.1f}%)")
        else:
            print(f"  t={t:4d}: 0 latents (0.0%)")

    if len(result) == 0:
        raise RuntimeError("No harmful latents collected! Try lowering confidence_threshold or using different prompts.")

    return result


def analyze_clusters(cluster_manager: ClusterManager, latents: torch.Tensor, labels: np.ndarray):
    """Analyze and print cluster statistics"""
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)

    n_clusters = cluster_manager.n_clusters
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\nCluster Distribution (Total: {len(labels)} samples)")
    print("-" * 50)
    for cluster_id in range(n_clusters):
        count = counts[unique == cluster_id][0] if cluster_id in unique else 0
        pct = 100 * count / len(labels)
        bar = "#" * int(pct / 2)
        print(f"  Cluster {cluster_id:2d}: {count:4d} ({pct:5.1f}%) {bar}")

    # Within-cluster statistics
    print(f"\nWithin-Cluster Statistics")
    print("-" * 50)

    features = cluster_manager.pool_features(latents)
    centroids = cluster_manager.centroids

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.sum() == 0:
            continue

        cluster_features = features[mask]
        centroid = centroids[cluster_id]

        distances = torch.norm(cluster_features - centroid, dim=1)
        mean_dist = distances.mean().item()
        std_dist = distances.std().item()

        print(f"  Cluster {cluster_id:2d}: mean_dist={mean_dist:.4f}, std={std_dist:.4f}")

    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)
    print("""
1. Visualize samples from each cluster to understand what patterns each mode captures
   - Use the saved latents + VAE decoder to reconstruct images per cluster

2. Adjust mode_aware schedules in config based on cluster characteristics:
   - Dominant clusters (>20% samples): May need stronger guidance or earlier start
   - Rare clusters (<5%): Consider if they need special handling

3. For clusters with high variance (large std_dist):
   - May benefit from more conservative guidance
   - Consider adding confidence thresholds

4. Example config adjustments:
   schedules:
     0:  # Dominant cluster
       scale: 1.5
       start_step: 0
     1:  # High-variance cluster
       scale: 0.8
       threshold: 0.7
""")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("MODE-AWARE HARMFUL LATENT CLUSTERING (Multi-Timestep)")
    print("=" * 70)

    # Mode: fit from existing latents
    if args.mode == 'fit':
        if args.latents_path is None or not os.path.exists(args.latents_path):
            raise ValueError("--latents_path required for fit mode")

        print(f"\n[Loading latents from {args.latents_path}]")
        latents_data = torch.load(args.latents_path, map_location='cpu')

        # Check if it's multi-timestep format (dict) or single timestep (tensor)
        if isinstance(latents_data, dict):
            print(f"  Loaded multi-timestep latents:")
            for t, lat in latents_data.items():
                print(f"    t={t}: {lat.shape[0]} latents")
        else:
            # Convert single tensor to dict format with dummy timestep
            print(f"  Loaded {latents_data.shape[0]} latents (single timestep)")
            latents_data = {500: latents_data}

    # Mode: generate and collect
    else:
        # Load classifier
        print(f"\n[Loading 3-way classifier from {args.classifier_ckpt}]")
        classifier_model = load_discriminator(
            ckpt_path=args.classifier_ckpt,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(args.device)
        classifier_model.eval()

        # Load pipeline
        pipe = load_pipeline(args.ckpt_path, args.device)

        # Load prompts
        prompts = load_prompts(args)
        print(f"\n[Loaded {len(prompts)} prompts]")
        for i, p in enumerate(prompts[:5]):
            print(f"  {i+1}. {p[:50]}...")
        if len(prompts) > 5:
            print(f"  ... and {len(prompts)-5} more")

        # Collect harmful latents at multiple timesteps
        latents_data = collect_harmful_latents(
            pipe=pipe,
            classifier_model=classifier_model,
            prompts=prompts,
            num_samples_per_prompt=args.num_samples_per_prompt,
            capture_timesteps=args.capture_timesteps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            harmful_class=args.harmful_class,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            seed=args.seed,
        )

        # Save latents (multi-timestep dict)
        if args.save_latents:
            latents_save_path = args.output_path.replace('.pt', '_latents.pt')
            os.makedirs(os.path.dirname(latents_save_path) if os.path.dirname(latents_save_path) else ".", exist_ok=True)
            torch.save(latents_data, latents_save_path)
            print(f"\n[Saved] Multi-timestep latents to {latents_save_path}")

    # Perform clustering for each timestep
    print(f"\n[Clustering with K={args.n_clusters} for each timestep]")

    all_cluster_managers = {}
    output_dir = os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else "."
    base_name = os.path.basename(args.output_path).replace('.pt', '')

    for t, latents in sorted(latents_data.items(), reverse=True):
        print(f"\n--- Timestep t={t} ---")
        cluster_manager = ClusterManager(
            n_clusters=args.n_clusters,
            pooling=args.pooling,
        )

        save_path = os.path.join(output_dir, f"{base_name}_t{t}.pt")
        labels = cluster_manager.fit(latents, save_path=save_path)
        all_cluster_managers[t] = cluster_manager

        # Brief analysis
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Save combined centroids (all timesteps in one file)
    combined_path = args.output_path
    combined_data = {
        'timesteps': sorted(latents_data.keys(), reverse=True),
        'n_clusters': args.n_clusters,
        'pooling': args.pooling,
        'centroids': {t: cm.centroids for t, cm in all_cluster_managers.items()}
    }
    torch.save(combined_data, combined_path)

    print(f"\n" + "=" * 70)
    print(f"[DONE] Multi-timestep centroids saved!")
    print(f"=" * 70)
    print(f"  Combined: {combined_path}")
    print(f"  Individual: {output_dir}/{base_name}_t*.pt")
    print(f"\nTimesteps: {sorted(latents_data.keys(), reverse=True)}")
    print(f"\nTo use in config:")
    print(f"  mode_aware:")
    print(f"    centroids_path: \"{combined_path}\"")


if __name__ == "__main__":
    main()
