"""
Machine Unlearning with FK Steering: Remove Nudity from Prompts
================================================================

Takes prompts that may contain nude/sexual content and generates images
with nudity removed (steered towards clothed people).

Usage:
    python unlearn_nudity_fk.py --prompt_file prompts/sexual_50.txt --output_dir outputs/unlearned
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add paths
sys.path.insert(0, '../three_classificaiton')

from diffusers import StableDiffusionPipeline
from fk_steering import FKSteering, NudityClassifierReward
from geo_models.classifier.classifier import load_discriminator


def load_prompts(prompt_file):
    """Load prompts from file."""
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove quotes if present
                line = line.strip('"').strip("'")
                prompts.append(line)
    return prompts


def fk_steering_unlearn(
    pipe,
    classifier,
    latents,
    text_embeddings,
    timesteps,
    num_particles,
    potential_type='max',
    lambda_scale=10.0,
    resampling_schedule=None,
    guidance_scale=7.5,
    verbose=False,
):
    """
    Run FK steering for unlearning nudity.

    Args:
        pipe: StableDiffusionPipeline
        classifier: Nudity classifier
        latents: Initial latents [k, C, H, W]
        text_embeddings: Text embeddings [2*k, seq_len, dim] (uncond + cond)
        timesteps: Diffusion timesteps
        num_particles: Number of particles
        potential_type: Type of potential
        lambda_scale: Lambda scaling
        resampling_schedule: When to resample
        guidance_scale: CFG scale
        verbose: Show progress

    Returns:
        Final latents [k, C, H, W]
        Reward history [num_steps, k]
    """
    # Create reward function (target clothed people - class 1)
    reward_fn = NudityClassifierReward(
        classifier=classifier,
        target_class=1,  # Clothed people
        use_softmax=False,
    )

    # Create FK steering
    fk = FKSteering(
        reward_fn=reward_fn,
        num_particles=num_particles,
        potential_type=potential_type,
        lambda_scale=lambda_scale,
        resampling_schedule=resampling_schedule or [],
    )

    # Track reward history
    rewards_history = []

    # Denoising loop
    iterator = tqdm(enumerate(timesteps), total=len(timesteps), disable=not verbose)
    for i, t in iterator:
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Compute rewards
        current_rewards = reward_fn(latents, t.item())
        rewards_history.append(current_rewards)

        # FK steering: resample if needed
        if fk.should_resample(i):
            potentials = fk.compute_potential(
                rewards_history[:-1],
                current_rewards,
                i,
            )
            latents, indices = fk.resample_particles(latents, potentials)

            if verbose:
                iterator.set_postfix({
                    'step': i,
                    'reward_mean': current_rewards.mean().item(),
                    'reward_max': current_rewards.max().item(),
                })

    return latents, torch.stack([r.cpu() for r in rewards_history]).numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Machine Unlearning with FK Steering - Remove Nudity'
    )

    # Input/Output
    parser.add_argument('--prompt_file', type=str, required=True,
                       help='File containing prompts (one per line)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')

    # Model paths
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4',
                       help='Stable Diffusion model ID')
    parser.add_argument('--classifier_ckpt', type=str,
                       default='./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth',
                       help='Path to classifier checkpoint')

    # FK steering parameters
    parser.add_argument('--num_particles', type=int, default=4,
                       help='Number of particles (k)')
    parser.add_argument('--potential_type', type=str, default='max',
                       choices=['max', 'difference', 'sum'],
                       help='Type of potential function')
    parser.add_argument('--lambda_scale', type=float, default=15.0,
                       help='Lambda scaling factor (higher = stronger nudity removal)')
    parser.add_argument('--resampling_interval', type=int, default=10,
                       help='Resample every N steps')

    # Generation parameters
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='CFG scale')
    parser.add_argument('--nsamples', type=int, default=1,
                       help='Number of samples per prompt')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Options
    parser.add_argument('--save_all_particles', action='store_true',
                       help='Save all particles (not just best)')
    parser.add_argument('--generate_baseline', action='store_true',
                       help='Also generate baseline (without FK steering)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show progress bars')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")

    # Load models
    print("\nLoading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"  Loading Stable Diffusion ({args.model_id})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    ).to(device)

    print(f"  Loading classifier ({args.classifier_ckpt})...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3,
    ).to(device)

    # Setup
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    resampling_schedule = list(range(0, args.num_inference_steps, args.resampling_interval))

    # Create reward function for evaluation
    reward_fn = NudityClassifierReward(classifier, target_class=1, use_softmax=False)

    # Results tracking
    results = {
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
        'prompts': [],
    }

    print("\n" + "="*80)
    print("MACHINE UNLEARNING - NUDITY REMOVAL WITH FK STEERING")
    print("="*80)
    print(f"Target: Clothed People (Class 1)")
    print(f"Particles: {args.num_particles}")
    print(f"Lambda: {args.lambda_scale}")
    print(f"Potential: {args.potential_type}")
    print(f"Resampling: Every {args.resampling_interval} steps")
    print("="*80 + "\n")

    # Process each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{len(prompts)}] Processing: {prompt[:80]}...")

        prompt_results = {
            'prompt': prompt,
            'prompt_idx': prompt_idx,
            'samples': [],
        }

        # Generate samples
        for sample_idx in range(args.nsamples):
            sample_seed = args.seed + prompt_idx * 1000 + sample_idx
            generator = torch.Generator(device=device).manual_seed(sample_seed)

            # Encode prompt
            text_inputs = pipe.tokenizer(
                [prompt] * args.num_particles,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

            # Unconditional embeddings
            uncond_tokens = pipe.tokenizer(
                [""] * args.num_particles,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = pipe.text_encoder(uncond_tokens.input_ids.to(device))[0]

            # Combine for CFG
            combined_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Initialize latents
            shape = (
                args.num_particles,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            )
            # Use same dtype as the model
            dtype = text_embeddings.dtype
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * pipe.scheduler.init_noise_sigma

            # Generate baseline if requested
            if args.generate_baseline:
                print(f"  Generating baseline...")
                # Reset scheduler for baseline generation
                pipe.scheduler.set_timesteps(args.num_inference_steps)
                baseline_timesteps = pipe.scheduler.timesteps

                # Create fresh baseline latent
                baseline_shape = (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
                baseline_generator = torch.Generator(device=device).manual_seed(sample_seed)
                baseline_latent = torch.randn(baseline_shape, generator=baseline_generator, device=device, dtype=dtype)
                baseline_latent = baseline_latent * pipe.scheduler.init_noise_sigma

                baseline_emb = torch.cat([uncond_embeddings[0:1], text_embeddings[0:1]])

                for t in baseline_timesteps:
                    latent_input = torch.cat([baseline_latent] * 2)
                    latent_input = pipe.scheduler.scale_model_input(latent_input, t)

                    with torch.no_grad():
                        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=baseline_emb).sample

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    baseline_latent = pipe.scheduler.step(noise_pred, t, baseline_latent).prev_sample

                # Reset scheduler again for FK steering
                pipe.scheduler.set_timesteps(args.num_inference_steps)

                # Decode baseline
                baseline_latent = 1 / pipe.vae.config.scaling_factor * baseline_latent
                with torch.no_grad():
                    baseline_image = pipe.vae.decode(baseline_latent).sample
                baseline_image = (baseline_image / 2 + 0.5).clamp(0, 1)
                baseline_image = baseline_image.cpu().permute(0, 2, 3, 1).numpy()[0]
                baseline_image = (baseline_image * 255).round().astype("uint8")
                baseline_pil = Image.fromarray(baseline_image)

                baseline_reward = reward_fn(baseline_latent[0], 0).item()

                # Save baseline
                baseline_path = os.path.join(
                    args.output_dir,
                    f"prompt{prompt_idx:03d}_sample{sample_idx}_baseline.png"
                )
                baseline_pil.save(baseline_path)

            # FK Steering
            print(f"  Running FK steering (sample {sample_idx+1}/{args.nsamples})...")
            fk_latents, reward_history = fk_steering_unlearn(
                pipe=pipe,
                classifier=classifier,
                latents=latents,
                text_embeddings=combined_embeddings,
                timesteps=timesteps,
                num_particles=args.num_particles,
                potential_type=args.potential_type,
                lambda_scale=args.lambda_scale,
                resampling_schedule=resampling_schedule,
                guidance_scale=args.guidance_scale,
                verbose=args.verbose,
            )

            # Decode images
            fk_latents = 1 / pipe.vae.config.scaling_factor * fk_latents
            with torch.no_grad():
                fk_images = pipe.vae.decode(fk_latents).sample
            fk_images = (fk_images / 2 + 0.5).clamp(0, 1)
            fk_images = fk_images.cpu().permute(0, 2, 3, 1).numpy()
            fk_images = (fk_images * 255).round().astype("uint8")

            # Get final rewards
            final_rewards = reward_history[-1]
            best_idx = np.argmax(final_rewards)

            # Save only best particle (save disk space)
            best_path = os.path.join(
                args.output_dir,
                f"prompt{prompt_idx:03d}_sample{sample_idx}_fk_best.png"
            )
            Image.fromarray(fk_images[best_idx]).save(best_path)

            # Optionally save all particles if flag is set
            if args.save_all_particles:
                for particle_idx in range(args.num_particles):
                    img_path = os.path.join(
                        args.output_dir,
                        f"prompt{prompt_idx:03d}_sample{sample_idx}_particle{particle_idx}_reward{final_rewards[particle_idx]:.3f}.png"
                    )
                    Image.fromarray(fk_images[particle_idx]).save(img_path)

            # Track results
            sample_result = {
                'sample_idx': sample_idx,
                'seed': sample_seed,
                'best_reward': float(final_rewards[best_idx]),
                'mean_reward': float(np.mean(final_rewards)),
                'all_rewards': final_rewards.tolist(),
                'best_particle_idx': int(best_idx),
            }

            if args.generate_baseline:
                sample_result['baseline_reward'] = baseline_reward
                sample_result['improvement'] = float(final_rewards[best_idx] - baseline_reward)

            prompt_results['samples'].append(sample_result)

            # Print results
            print(f"    Best reward: {final_rewards[best_idx]:.4f}")
            print(f"    Mean reward: {np.mean(final_rewards):.4f}")
            if args.generate_baseline:
                print(f"    Baseline reward: {baseline_reward:.4f}")
                print(f"    Improvement: {final_rewards[best_idx] - baseline_reward:.4f}")

        results['prompts'].append(prompt_results)

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_best_rewards = []
    all_mean_rewards = []
    all_improvements = []

    for prompt_result in results['prompts']:
        for sample_result in prompt_result['samples']:
            all_best_rewards.append(sample_result['best_reward'])
            all_mean_rewards.append(sample_result['mean_reward'])
            if 'improvement' in sample_result:
                all_improvements.append(sample_result['improvement'])

    print(f"Total prompts: {len(prompts)}")
    print(f"Total samples: {len(all_best_rewards)}")
    print(f"\nBest Particle Rewards:")
    print(f"  Mean: {np.mean(all_best_rewards):.4f}")
    print(f"  Std:  {np.std(all_best_rewards):.4f}")
    print(f"  Min:  {np.min(all_best_rewards):.4f}")
    print(f"  Max:  {np.max(all_best_rewards):.4f}")

    print(f"\nMean Particle Rewards:")
    print(f"  Mean: {np.mean(all_mean_rewards):.4f}")
    print(f"  Std:  {np.std(all_mean_rewards):.4f}")

    if all_improvements:
        print(f"\nImprovement over Baseline:")
        print(f"  Mean: {np.mean(all_improvements):.4f}")
        print(f"  Std:  {np.std(all_improvements):.4f}")
        print(f"  % Positive: {100 * np.mean(np.array(all_improvements) > 0):.1f}%")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Results JSON: {results_path}")
    print("="*80)


if __name__ == '__main__':
    main()
