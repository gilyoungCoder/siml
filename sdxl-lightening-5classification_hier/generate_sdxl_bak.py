#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_sdxl_lightning_guidance.py

import os
from argparse import ArgumentParser
from functools import partial
from PIL import Image

import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    DDIMScheduler,
)

from geo_utils.guidance_utils import GuidanceModel


def parse_args():
    p = ArgumentParser("SDXL-Lightning + Classifier Guidance")
    p.add_argument('lightning_repo',
                   help="HuggingFace ID of Lightning UNet (e.g. ByteDance/SDXL-Lightning)")
    p.add_argument('--revision',       default=None, help="(Optional) Lightning repo revision")
    p.add_argument('--sdxl_base',      default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="Base SDXL for VAE, tokenizer, text_encoder")
    p.add_argument('--prompt_file',    required=True, help="줄단위 .txt 프롬프트 파일")
    p.add_argument('--output_dir',     required=True, help="이미지 저장 디렉터리")
    p.add_argument('--nsamples',       type=int, default=4)
    p.add_argument('--num_inference_steps', type=int, default=4,
                   help="Lightning distilled steps (4)")
    p.add_argument('--cfg_scale',      type=float, default=0.0,
                   help="Text guidance scale (Lightning에서는 0)")
    p.add_argument('--freedom',        action='store_true',
                   help="Enable classifier guidance")
    p.add_argument('--freedom_model_args',  default="configs/models/classifier.yaml",
                   help="Guidance model config (.yaml)")
    p.add_argument('--freedom_model_ckpt',  help="Guidance classifier .pth 체크포인트")
    p.add_argument('--freedom_scale',  type=float, default=10.0,
                   help="Classifier guidance strength")
    p.add_argument('--guide_start',    type=int, default=1,
                   help="몇 번째 스텝부터 classifier guidance 적용")
    p.add_argument('--device',         default='cuda:0')
    p.add_argument('--use_fp16',       action='store_true')
    p.add_argument('--seed',           type=int, default=None)
    return p.parse_args()


def load_prompts(path: str):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def save_image(img: Image.Image, out_dir: str, idx: int):
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"{idx:06d}.png"))


def main():
    args = parse_args()
    accel = Accelerator(mixed_precision='fp16' if args.use_fp16 else 'no')
    device = torch.device(args.device)
    dtype = torch.float16 if args.use_fp16 else torch.float32

    # ── 1) Load SDXL-base components
    vae = AutoencoderKL.from_pretrained(
        args.sdxl_base, subfolder="vae"
    ).to(device, dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.sdxl_base, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.sdxl_base, subfolder="text_encoder"
    ).to(device)

    # ── 2) Load Lightning UNet weights
    ckpt_name = f"sdxl_lightning_{args.num_inference_steps}step_unet.safetensors"
    ckpt_path = hf_hub_download(
        repo_id=args.lightning_repo, filename=ckpt_name, revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.sdxl_base, subfolder="unet", torch_dtype=dtype
    ).to(device)
    state = load_file(ckpt_path, "cpu")
    unet.load_state_dict(state, strict=True)

    # ── 3) Load official XL-pipeline & overwrite components
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl_base, torch_dtype=dtype, safety_checker=None
    ).to(device)
    pipe.vae = vae
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.unet = unet

    # ── ➞ 여기서 커스텀 콜백 텐서 입력을 추가 허용
    pipe._callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "noise_pred",
        "prev_latents",
    ]

    # ── 4) 4-step DDIM 스케줄러로 교체
    pipe.scheduler = DDIMScheduler(
        num_train_timesteps=4,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

    # 완전 NSFW 비활성화
    pipe.safety_checker = None
    pipe.feature_extractor = None

    # ── 5) Optional: classifier guidance callback 설정
    if args.freedom:
        guide = GuidanceModel(
            pipe,
            args.freedom_model_args,
            args.freedom_model_ckpt,
            target_class=1,
            device=device
        )
        def cg_callback(pipeline, step, timestep, cb_kwargs):
            # step >= guide_start 에서만 guidance 적용
            if step >= args.guide_start:
                return guide.guidance(
                    pipeline,
                    cb_kwargs,         # 이 안에 noise_pred, prev_latents가 포함되어 있음
                    step,
                    timestep,
                    scale=args.freedom_scale,
                    target_class=1
                )
            return cb_kwargs

        cb_fn = partial(cg_callback)
        cb_tensors = ["noise_pred", "prev_latents"]
    else:
        cb_fn, cb_tensors = None, None

    # ── 6) Generation Loop
    prompts = load_prompts(args.prompt_file)
    gen = torch.Generator(device=device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    idx = 0
    for prompt in prompts:
        # timesteps 리셋
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        pipe.scheduler._step_index = 0

        out = pipe(
            prompt=[prompt],
            height=1024,
            width=1024,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.cfg_scale,
            num_images_per_prompt=args.nsamples,
            generator=gen,
            callback_on_step_end=cb_fn,
            callback_on_step_end_tensor_inputs=cb_tensors,
            callback_steps=1,
        )
        for img in out.images:
            save_image(img, args.output_dir, idx)
            idx += 1

    # prompts.txt도 같이 저장
    with open(os.path.join(args.output_dir, "prompts.txt"), "w", encoding='utf-8') as f:
        for p in prompts:
            f.write(p + "\n")


if __name__ == "__main__":
    main()
