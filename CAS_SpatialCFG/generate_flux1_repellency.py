#!/usr/bin/env python3
"""FLUX.1-dev SAFREE + SafeDenoiser/SGF MVP for I2P Q16 runs.

This script reuses the working FLUX.1 denoising loop and adds a
concept-specific negative-reference repellency hook. References are FLUX VAE
latents under exemplars/i2p_v1_flux1.
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

CAS = Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG')
sys.path.insert(0, str(CAS))
from generate_flux1_safree import encode_prompt_flux1, load_prompts, set_seed  # noqa: E402


def pack_ref(pipe, ref4d: torch.Tensor, h: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ref4d = ref4d.to(device=device, dtype=dtype)
    if ref4d.ndim != 4:
        raise ValueError(f'ref_latents must be 4D, got {tuple(ref4d.shape)}')
    m, c, rh, rw = ref4d.shape
    if rh != h or rw != w:
        ref4d = torch.nn.functional.interpolate(ref4d.float(), size=(h, w), mode='bilinear', align_corners=False).to(dtype)
        rh, rw = h, w
    return pipe._pack_latents(ref4d, m, c, rh, rw)


class PackedRepellency:
    def __init__(self, ref_packed: torch.Tensor, scale: float, sigma: float, mode: str,
                 t_low: float = 0.0, t_high: float = 1.0, normalize: bool = True):
        self.ref = ref_packed.detach().float()
        self.scale = float(scale)
        self.sigma = float(max(sigma, 1e-6))
        self.mode = mode
        self.t_low = float(t_low)
        self.t_high = float(t_high)
        self.normalize = normalize
        self.m = self.ref.shape[0]
        self.ref_flat = self.ref.reshape(self.m, -1)
        self.ref_unit = F.normalize(self.ref_flat, dim=-1)

    def in_window(self, sigma_t: float) -> bool:
        return self.t_low <= sigma_t <= self.t_high

    @torch.no_grad()
    def neg_direction(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float().reshape(x.shape[0], -1)
        xu = F.normalize(xf, dim=-1)
        logits = (xu @ self.ref_unit.T) / self.sigma
        weights = torch.softmax(logits, dim=-1)
        neg = (weights @ self.ref_flat).reshape_as(x.float())
        if self.normalize:
            neg_norm = neg.flatten(1).norm(dim=-1).clamp_min(1e-6).view(-1, 1, 1)
            x_norm = x.float().flatten(1).norm(dim=-1).clamp_min(1e-6).view(-1, 1, 1)
            neg = neg / neg_norm * x_norm
        return neg.to(dtype)

    @torch.no_grad()
    def apply(self, latents: torch.Tensor, velocity: torch.Tensor, sigma_t: float) -> torch.Tensor:
        if not self.in_window(float(sigma_t)) or self.scale <= 0:
            return velocity
        if self.mode == 'sgf':
            return velocity - self.scale * self.neg_direction(velocity)
        st = max(float(sigma_t), 1e-6)
        z0 = latents - st * velocity
        z0_corr = z0 - self.scale * self.neg_direction(z0)
        return (latents - z0_corr) / st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='black-forest-labs/FLUX.1-dev')
    p.add_argument('--prompts', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--ref_latents', required=True)
    p.add_argument('--concept', default='sexual')
    p.add_argument('--mode', choices=['safedenoiser','sgf'], default='safedenoiser')
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--guidance_scale', type=float, default=3.5)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--max_sequence_length', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--nsamples', type=int, default=1)
    p.add_argument('--start_idx', type=int, default=0)
    p.add_argument('--end_idx', type=int, default=-1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16','float16'], default='bfloat16')
    p.add_argument('--repellency_scale', type=float, default=0.08)
    p.add_argument('--repellency_sigma', type=float, default=0.08)
    p.add_argument('--repellency_t_low', type=float, default=0.0)
    p.add_argument('--repellency_t_high', type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device = torch.device(args.device)
    gpu_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
    set_seed(args.seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f'[FLUX1 {args.mode}] concept={args.concept} out={outdir}')
    print(f'  ref={args.ref_latents} scale={args.repellency_scale} sigma={args.repellency_sigma} window=[{args.repellency_t_low},{args.repellency_t_high}]')

    pipe = FluxPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    pipe.set_progress_bar_config(disable=True)
    transformer, vae, scheduler = pipe.transformer, pipe.vae, pipe.scheduler
    guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32) if transformer.config.guidance_embeds else None

    num_channels = transformer.config.in_channels // 4
    vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
    lat_h = 2 * (args.height // (vae_scale_factor * 2))
    lat_w = 2 * (args.width // (vae_scale_factor * 2))
    latent_ids = pipe._prepare_latent_image_ids(1, lat_h // 2, lat_w // 2, device, dtype)

    ref4d = torch.load(args.ref_latents, map_location='cpu')
    ref_packed = pack_ref(pipe, ref4d, lat_h, lat_w, dtype, device)
    rep = PackedRepellency(ref_packed, args.repellency_scale, args.repellency_sigma, args.mode,
                           args.repellency_t_low, args.repellency_t_high)
    print(f'  ref_packed={tuple(ref_packed.shape)} latents=[{num_channels},{lat_h},{lat_w}]')

    prompts, seeds = load_prompts(args.prompts)
    end = len(prompts) if args.end_idx < 0 else min(args.end_idx, len(prompts))
    items = list(enumerate(prompts))[args.start_idx:end]
    json.dump(vars(args), open(outdir/'args.json','w'), indent=2)
    stats = []

    for pi, prompt in tqdm(items, desc=f'flux1-{args.mode}'):
        seed_base = seeds[pi] if pi < len(seeds) and seeds[pi] is not None else args.seed + pi
        for si in range(args.nsamples):
            seed = int(seed_base) + si
            set_seed(seed)
            with torch.no_grad():
                pe, pooled, text_ids = encode_prompt_flux1(pipe, prompt, device, args.max_sequence_length)
            latents = torch.randn(1, num_channels, lat_h, lat_w, device=device, dtype=dtype)
            latents = pipe._pack_latents(latents, 1, num_channels, lat_h, lat_w)
            sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
            mu = calculate_shift(latents.shape[1], scheduler.config.get('base_image_seq_len', 256),
                                 scheduler.config.get('max_image_seq_len', 4096),
                                 scheduler.config.get('base_shift', 0.5), scheduler.config.get('max_shift', 1.15))
            timesteps, _ = retrieve_timesteps(scheduler, args.steps, device, sigmas=sigmas, mu=mu)
            for step_idx, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with torch.no_grad():
                    v = transformer(hidden_states=latents.to(transformer.dtype), timestep=timestep/1000,
                                    guidance=guidance, pooled_projections=pooled.to(dtype),
                                    encoder_hidden_states=pe.to(dtype), txt_ids=text_ids,
                                    img_ids=latent_ids, return_dict=False)[0]
                sigma_t = float(scheduler.sigmas[step_idx].item()) if hasattr(scheduler, 'sigmas') else float(sigmas[step_idx])
                v = rep.apply(latents, v, sigma_t)
                latents = scheduler.step(v, t, latents, return_dict=False)[0]
            with torch.no_grad():
                unpack = pipe._unpack_latents(latents, args.height, args.width, vae_scale_factor)
                unpack = (unpack / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(unpack.to(vae.dtype), return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                arr = (image[0].cpu().permute(1,2,0).float().numpy()*255).round().astype(np.uint8)
            fn = f'{pi:04d}.png' if args.nsamples == 1 else f'{pi:04d}_{si:02d}.png'
            Image.fromarray(arr).save(outdir/fn)
            stats.append({'idx': pi, 'seed': seed, 'prompt': prompt[:160]})
            print(f'  saved {fn}')
    json.dump(stats, open(outdir/'stats.json','w'), indent=2)
    print(f'Done: {len(stats)} images -> {outdir}')

if __name__ == '__main__':
    main()
