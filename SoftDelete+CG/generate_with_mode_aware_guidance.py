"""
Mode-Aware Classifier Guidance Image Generation

각 cluster(mode)별로 다른 guidance scale을 적용하여 이미지 생성
"""

import os
import argparse
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from geo_utils.mode_aware_gradient_model import ClusterManager
from geo_models.classifier.classifier import load_discriminator


# =========================
# Argument Parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with mode-aware classifier guidance')

    # Model paths
    parser.add_argument('--ckpt_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--classifier_ckpt', type=str, required=True)
    parser.add_argument('--centroids_path', type=str, required=True)

    # Input
    parser.add_argument('--prompts_file', type=str, default=None)
    parser.add_argument('--prompts', type=str, nargs='+', default=None)

    # Generation
    parser.add_argument('--num_samples_per_prompt', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--cfg_scale', type=float, default=7.5)

    # Mode-aware guidance
    parser.add_argument('--base_guidance_scale', type=float, default=100.0)
    parser.add_argument('--harmful_class', type=int, default=2)
    parser.add_argument('--guidance_start_step', type=int, default=0)
    parser.add_argument('--guidance_end_step', type=int, default=50)
    parser.add_argument('--mode_scales_file', type=str, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, default='mode_aware_outputs')
    parser.add_argument('--save_metadata', action='store_true')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# =========================
# Utilities
# =========================
def load_prompts(args) -> List[str]:
    if args.prompts is not None:
        return args.prompts
    if args.prompts_file is not None:
        with open(args.prompts_file, 'r') as f:
            return [l.strip() for l in f if l.strip()]
    return ["a person walking in the park"]


def load_mode_scales(path: Optional[str], n_clusters: int) -> Dict[int, float]:
    if path is not None and os.path.exists(path):
        with open(path, 'r') as f:
            scales = json.load(f)
        return {int(k): float(v) for k, v in scales.items()}
    return {i: 1.0 for i in range(n_clusters)}


# =========================
# Mode-Aware Guidance
# =========================
class ModeAwareGuidance:
    def __init__(
        self,
        classifier: torch.nn.Module,
        cluster_manager: ClusterManager,
        harmful_class: int,
        base_scale: float,
        mode_scales: Dict[int, float],
        guidance_start_step: int,
        guidance_end_step: int,
        device: str,
    ):
        self.classifier = classifier
        self.cluster_manager = cluster_manager
        self.harmful_class = harmful_class
        self.base_scale = base_scale
        self.mode_scales = mode_scales
        self.guidance_start_step = guidance_start_step
        self.guidance_end_step = guidance_end_step
        self.device = device

        self.classifier.eval()

    def get_guidance_scale(self, latents: torch.Tensor, step: int) -> float:
        if step < self.guidance_start_step or step >= self.guidance_end_step:
            return 0.0

        with torch.no_grad():
            cluster_ids, _ = self.cluster_manager.get_nearest_cluster(
                latents.float().cpu()
            )
            cluster_id = cluster_ids[0].item()

        return self.base_scale * self.mode_scales.get(cluster_id, 1.0)

    def compute_guidance(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        step: int,
    ) -> torch.Tensor:

        guidance_scale = self.get_guidance_scale(latents, step)
        if guidance_scale == 0.0:
            return torch.zeros_like(latents)

        # ---- FP32 path for classifier + gradient ----
        latents_fp32 = latents.detach().float().requires_grad_(True)

        if timestep.dtype != torch.long:
            timestep = timestep.long()

        with torch.cuda.amp.autocast(enabled=False):
            logits = self.classifier(latents_fp32, timestep)
            log_probs = F.log_softmax(logits, dim=-1)
            harmful_log_prob = log_probs[:, self.harmful_class].sum()
            grad = torch.autograd.grad(harmful_log_prob, latents_fp32)[0]
            guidance_fp32 = -guidance_scale * grad

        return guidance_fp32.to(dtype=latents.dtype)


# =========================
# Generation
# =========================
def generate_with_mode_aware_guidance(
    pipe: StableDiffusionPipeline,
    guidance_module: ModeAwareGuidance,
    prompt: str,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
) -> Tuple[Image.Image, Dict]:

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    text_embeddings = pipe._encode_prompt(
        prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        generator=generator,
        device=pipe.device,
        dtype=text_embeddings.dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps

    metadata = {"prompt": prompt, "seed": seed, "guidance_applied": []}

    for i, t in enumerate(tqdm(timesteps, leave=False)):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_text - noise_uncond)

        timestep_tensor = torch.tensor(
            [int(t.item())], device=pipe.device, dtype=torch.long
        ).expand(latents.size(0))

        classifier_guidance = guidance_module.compute_guidance(
            latents, timestep_tensor, i
        )

        if classifier_guidance.abs().sum() > 0:
            metadata["guidance_applied"].append({
                "step": i,
                "scale": guidance_module.get_guidance_scale(latents, i),
                "grad_norm": classifier_guidance.norm().item(),
            })

        noise_pred = noise_pred - classifier_guidance
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        latents = latents / 0.18215
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype("uint8"))

    return image, metadata


# =========================
# Main
# =========================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3,
    ).to(args.device).float().eval()

    cluster_manager = ClusterManager()
    cluster_manager.load(args.centroids_path)

    mode_scales = load_mode_scales(args.mode_scales_file, cluster_manager.n_clusters)

    guidance_module = ModeAwareGuidance(
        classifier=classifier,
        cluster_manager=cluster_manager,
        harmful_class=args.harmful_class,
        base_scale=args.base_guidance_scale,
        mode_scales=mode_scales,
        guidance_start_step=args.guidance_start_step,
        guidance_end_step=args.guidance_end_step,
        device=args.device,
    )

    prompts = load_prompts(args)
    all_metadata = []

    for p_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        for s_idx in range(args.num_samples_per_prompt):
            seed = args.seed + p_idx * 1000 + s_idx
            image, metadata = generate_with_mode_aware_guidance(
                pipe,
                guidance_module,
                prompt,
                args.num_inference_steps,
                args.cfg_scale,
                seed,
            )
            name = f"prompt_{p_idx:04d}_sample_{s_idx:02d}.png"
            image.save(os.path.join(args.output_dir, name))
            metadata["image_path"] = name
            all_metadata.append(metadata)

    if args.save_metadata:
        with open(os.path.join(args.output_dir, "generation_metadata.json"), "w") as f:
            json.dump(all_metadata, f, indent=2)

    print(f"[DONE] Generated {len(all_metadata)} images → {args.output_dir}")


if __name__ == "__main__":
    main()
