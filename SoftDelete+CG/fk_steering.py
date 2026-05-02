"""
Feynman-Kac Steering for Diffusion Models
==========================================

Implementation of FK steering from "A General Framework for Inference-time Scaling and
Steering of Diffusion Models" (Singhal et al., 2025)

Steers diffusion models using particle-based sampling with intermediate rewards.
"""

import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class FKSteering:
    """
    Feynman-Kac Steering for diffusion models.

    Args:
        reward_fn: Function that computes reward r(x_t, t) -> float
        num_particles: Number of particles (k)
        potential_type: Type of potential ('max', 'difference', 'sum')
        lambda_scale: Scaling factor λ for reward
        resampling_schedule: List of timesteps to resample at (e.g., [0, 20, 40, 60, 80])
    """

    def __init__(
        self,
        reward_fn: Callable[[torch.Tensor, int], torch.Tensor],
        num_particles: int = 4,
        potential_type: str = 'max',
        lambda_scale: float = 10.0,
        resampling_schedule: Optional[List[int]] = None,
    ):
        self.reward_fn = reward_fn
        self.num_particles = num_particles
        self.potential_type = potential_type
        self.lambda_scale = lambda_scale
        self.resampling_schedule = resampling_schedule or []

        # Track particle history for debugging
        self.particle_rewards = []
        self.particle_history = []

    def compute_potential(
        self,
        rewards_history: List[torch.Tensor],
        current_reward: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Compute potential G_t based on reward history.

        Args:
            rewards_history: List of rewards [r_T, r_{T-1}, ..., r_t]
            current_reward: Current reward r_t
            t: Current timestep

        Returns:
            Potential scores for each particle [k]
        """
        # Move history to same device as current_reward
        device = current_reward.device
        rewards_history_device = [r.to(device) if r.device != device else r for r in rewards_history]

        if self.potential_type == 'max':
            # G_t = exp(λ * max_{s>=t} r_φ(x_s))
            all_rewards = torch.stack(rewards_history_device + [current_reward], dim=1)  # [k, history_len]
            max_rewards = torch.max(all_rewards, dim=1)[0]  # [k]
            potential = torch.exp(self.lambda_scale * max_rewards)

        elif self.potential_type == 'difference':
            # G_t = exp(λ * (r_φ(x_t) - r_φ(x_{t+1})))
            if len(rewards_history_device) > 0:
                prev_reward = rewards_history_device[-1]
                potential = torch.exp(self.lambda_scale * (current_reward - prev_reward))
            else:
                potential = torch.ones_like(current_reward)

        elif self.potential_type == 'sum':
            # G_t = exp(λ * Σ_{s>=t} r_φ(x_s))
            all_rewards = torch.stack(rewards_history_device + [current_reward], dim=1)  # [k, history_len]
            sum_rewards = torch.sum(all_rewards, dim=1)  # [k]
            potential = torch.exp(self.lambda_scale * sum_rewards)

        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")

        return potential

    def resample_particles(
        self,
        particles: torch.Tensor,
        potentials: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resample particles based on potential weights using multinomial sampling.

        Args:
            particles: Current particles [k, C, H, W]
            potentials: Potential weights [k]

        Returns:
            Resampled particles [k, C, H, W]
            Indices of selected particles [k]
        """
        k = particles.shape[0]

        # Normalize potentials to get probabilities
        weights = potentials / (potentials.sum() + 1e-10)

        # Sample k indices with replacement
        indices = torch.multinomial(weights, num_samples=k, replacement=True)

        # Resample particles
        resampled_particles = particles[indices]

        return resampled_particles, indices

    def should_resample(self, t: int) -> bool:
        """Check if we should resample at timestep t."""
        return t in self.resampling_schedule

    def get_best_particle(
        self,
        particles: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Get the particle with highest reward.

        Args:
            particles: All particles [k, C, H, W]
            rewards: Reward for each particle [k]

        Returns:
            Best particle [C, H, W]
            Index of best particle
        """
        best_idx = torch.argmax(rewards)
        return particles[best_idx], best_idx.item()


class NudityClassifierReward:
    """
    Reward function using the nudity classifier.

    Reward design:
        - Target class 1 (clothed people): r = logit[1] - logit[2]
        - Target class 0 (not people): r = logit[0] - max(logit[1], logit[2])
        - Target class 2 (nude people): r = logit[2] - logit[1]

    Args:
        classifier: Nudity classifier model
        target_class: Target class (0, 1, or 2)
        use_softmax: If True, use probabilities instead of logits
    """

    def __init__(
        self,
        classifier: torch.nn.Module,
        target_class: int = 1,
        use_softmax: bool = False,
    ):
        self.classifier = classifier
        self.target_class = target_class
        self.use_softmax = use_softmax
        self.classifier.eval()

    def __call__(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute reward for latent x_t at timestep t.

        Args:
            x_t: Latent tensor [k, C, H, W] or [C, H, W]
            t: Timestep (scalar)

        Returns:
            Rewards [k] or scalar
        """
        single_input = x_t.ndim == 3
        if single_input:
            x_t = x_t.unsqueeze(0)  # [1, C, H, W]

        k = x_t.shape[0]
        device = x_t.device
        original_dtype = x_t.dtype

        # Create timestep tensor
        t_tensor = torch.full((k,), t, dtype=torch.long, device=device)

        # Get classifier logits
        # Convert to float32 for classifier (which is trained in float32)
        with torch.no_grad():
            x_t_float32 = x_t.float()
            logits = self.classifier(x_t_float32, t_tensor)  # [k, 3]

        if self.use_softmax:
            logits = F.softmax(logits, dim=-1)

        # Compute reward based on target class
        if self.target_class == 0:
            # Not people: maximize class 0, minimize others
            reward = logits[:, 0] - torch.max(logits[:, 1], logits[:, 2])
        elif self.target_class == 1:
            # Clothed people: maximize class 1, minimize nude (class 2)
            reward = logits[:, 1] - logits[:, 2]
        elif self.target_class == 2:
            # Nude people: maximize class 2, minimize clothed (class 1)
            reward = logits[:, 2] - logits[:, 1]
        else:
            raise ValueError(f"Invalid target class: {self.target_class}")

        if single_input:
            reward = reward[0]

        return reward


def fk_steering_pipeline(
    pipe,
    prompt: str,
    classifier: torch.nn.Module,
    target_class: int = 1,
    num_particles: int = 4,
    potential_type: str = 'max',
    lambda_scale: float = 10.0,
    resampling_schedule: Optional[List[int]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    return_all_particles: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run FK steering on a Stable Diffusion pipeline.

    Args:
        pipe: StableDiffusionPipeline
        prompt: Text prompt
        classifier: Nudity classifier model
        target_class: Target class for steering (0, 1, or 2)
        num_particles: Number of particles k
        potential_type: Type of potential ('max', 'difference', 'sum')
        lambda_scale: Scaling factor λ
        resampling_schedule: Timesteps to resample at
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        seed: Random seed
        return_all_particles: If True, return all particles, else best one
        verbose: Show progress bar

    Returns:
        dict with keys:
            - 'images': Generated images (all particles or best)
            - 'rewards': Final rewards for each particle
            - 'particle_history': History of particle rewards
            - 'best_idx': Index of best particle
    """
    # Set seed
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    # Setup resampling schedule
    if resampling_schedule is None:
        # Default: resample every 10 steps
        resampling_schedule = list(range(0, num_inference_steps, 10))

    # Create reward function
    reward_fn = NudityClassifierReward(
        classifier=classifier,
        target_class=target_class,
        use_softmax=False,
    )

    # Create FK steering
    fk = FKSteering(
        reward_fn=reward_fn,
        num_particles=num_particles,
        potential_type=potential_type,
        lambda_scale=lambda_scale,
        resampling_schedule=resampling_schedule,
    )

    # Encode prompt
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps

    # Get text embeddings
    text_inputs = pipe.tokenizer(
        [prompt] * num_particles,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]

    # Unconditional embeddings for CFG
    uncond_tokens = pipe.tokenizer(
        [""] * num_particles,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_tokens.input_ids.to(pipe.device))[0]

    # Combine for CFG
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initialize particles (latents)
    shape = (
        num_particles,
        pipe.unet.config.in_channels,
        pipe.unet.config.sample_size,
        pipe.unet.config.sample_size,
    )
    # Use same dtype as model
    dtype = text_embeddings.dtype
    latents = torch.randn(shape, generator=generator, device=pipe.device, dtype=dtype)
    latents = latents * pipe.scheduler.init_noise_sigma

    # Track reward history for each particle
    rewards_history = []

    # Denoising loop with FK steering
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

        # Compute rewards for current particles
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

    # Decode latents to images
    latents = 1 / pipe.vae.config.scaling_factor * latents
    with torch.no_grad():
        images = pipe.vae.decode(latents).sample

    # Convert to PIL images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    images = [pipe.numpy_to_pil(img)[0] for img in images]

    # Final rewards
    final_rewards = rewards_history[-1]

    # Get best particle
    best_idx = torch.argmax(final_rewards).item()

    results = {
        'images': images if return_all_particles else images[best_idx],
        'all_images': images,
        'rewards': final_rewards.cpu().numpy(),
        'particle_history': torch.stack([r.cpu() for r in rewards_history]).numpy(),  # [num_steps, k]
        'best_idx': best_idx,
    }

    return results


def compare_fk_vs_baseline(
    pipe,
    prompt: str,
    classifier: torch.nn.Module,
    target_class: int = 1,
    num_particles: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
):
    """
    Compare FK steering vs baseline (best-of-n) vs single sample.

    Returns:
        dict with baseline, best-of-n, and FK steering results
    """
    results = {}

    # 1. Single baseline sample
    print("Generating baseline (single sample)...")
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    baseline_output = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    results['baseline'] = {
        'image': baseline_output.images[0],
        'type': 'baseline'
    }

    # 2. Best-of-n sampling
    print(f"Generating best-of-n (n={num_particles})...")
    best_of_n_images = []
    best_of_n_rewards = []

    reward_fn = NudityClassifierReward(classifier, target_class)

    for i in range(num_particles):
        gen = torch.Generator(device=pipe.device).manual_seed(seed + i) if seed else None
        output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )

        # Convert to latent and compute reward
        with torch.no_grad():
            image_tensor = torch.from_numpy(np.array(output.images[0])).to(pipe.device)
            image_tensor = image_tensor.float() / 127.5 - 1.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            latent = pipe.vae.encode(image_tensor).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor

            reward = reward_fn(latent[0], 0)

        best_of_n_images.append(output.images[0])
        best_of_n_rewards.append(reward.item())

    best_idx = np.argmax(best_of_n_rewards)
    results['best_of_n'] = {
        'image': best_of_n_images[best_idx],
        'all_images': best_of_n_images,
        'rewards': best_of_n_rewards,
        'best_idx': best_idx,
        'type': 'best_of_n'
    }

    # 3. FK steering
    print(f"Generating with FK steering (k={num_particles})...")
    fk_results = fk_steering_pipeline(
        pipe=pipe,
        prompt=prompt,
        classifier=classifier,
        target_class=target_class,
        num_particles=num_particles,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        return_all_particles=True,
    )
    fk_results['type'] = 'fk_steering'
    results['fk_steering'] = fk_results

    return results
