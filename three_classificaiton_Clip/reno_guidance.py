#!/usr/bin/env python
# reno_guidance.py

import os
import argparse
from functools import partial

import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='ReNO-style CLIPScore Guidance')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_reno')
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--clip_scale', type=float, default=1.0,
                        help="How strongly to push away from neg prompt")
    parser.add_argument('--reno_steps', type=int, default=10,
                        help="How many gradient steps per diffusion step")
    parser.add_argument('--negative_prompt', type=str,
                        default="an image showing nudity")
    return parser.parse_args()

def save_image(pil_img: Image.Image, path: str):
    pil_img.save(path)

def reno_callback(
    pipe, step, timestep, callback_kwargs,
    clip_model, neg_embeds, scale, reno_steps
):
    # 1) latents 复制并开启 grad
    latents = callback_kwargs["latents"].detach().clone().requires_grad_(True)
    device = latents.device

    # 保证 VAE/CLIP 不会被修改
    for p in pipe.vae.parameters(): p.requires_grad_(False)
    for p in clip_model.parameters(): p.requires_grad_(False)

    # 每个 diffusion step 里做多次 gradient ascent
    for _ in range(reno_steps):
        # **重新打开梯度计算**
        with torch.enable_grad():
            # 2) 解码到像素空间
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)[0]
            decoded = (decoded.clamp(-1,1) + 1) / 2  # → [0..1]

            # 3) resize 到 CLIP 要的 224×224
            img = torch.nn.functional.interpolate(
                decoded, size=(224,224), mode="bilinear", align_corners=False
            )

            # 4) normalize
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                device=device).view(1,3,1,1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                device=device).view(1,3,1,1)
            pixel_values = (img - mean) / std

            # 5) CLIP image features
            img_embeds = clip_model.get_image_features(pixel_values=pixel_values)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            # 6) 计算“nudity”相似度
            sim = (img_embeds @ neg_embeds.T).sum()

            # 7) 对 -sim 做梯度上升
            grad = torch.autograd.grad(sim, latents)[0]
            latents = (latents - scale * grad).detach().clone().requires_grad_(True)

    # 最终把修改后的 latents 返回给 diffusion
    return {"latents": latents.detach()}

def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        safety_checker=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # negative prompt embedding
    neg_inputs = clip_processor(
        text=[args.negative_prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)
    neg_embeds = clip_model.get_text_features(
        input_ids=neg_inputs.input_ids,
        attention_mask=neg_inputs.attention_mask
    )
    neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]

    for idx, prompt in enumerate(prompts):
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        pipe.scheduler._step_index = 0

        cb = partial(
            reno_callback,
            clip_model=clip_model,
            neg_embeds=neg_embeds,
            scale=args.clip_scale,
            reno_steps=args.reno_steps
        )

        out = pipe(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.nsamples,
            callback_on_step_end=cb,
            callback_steps=1,
            height=512, width=512,
        )
        for i, img in enumerate(out.images):
            path = os.path.join(args.output_dir, f"{idx+1:03d}_{i+1:02d}.png")
            save_image(img, path)
            print(f"[{idx+1}/{len(prompts)}] Saved → {path}")

if __name__ == "__main__":
    main()
