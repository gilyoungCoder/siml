#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_sdxl_guidance.py

import os
import json
import torch
from argparse import ArgumentParser
from functools import partial
from PIL import Image
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from geo_utils.custom_sdxl import CustomStableDiffusionXLPipeline
from geo_utils.guidance_utils import GuidanceModel


def parse_args():
    p = ArgumentParser("SDXL + Classifier Guidance")
    p.add_argument("--sdxl_base",        required=True,
                   help="SDXL 체크포인트 (예: stabilityai/stable-diffusion-xl-base-1.0)")
    p.add_argument("--prompt_file",      required=True,
                   help=".txt/.json/.csv 형식의 프롬프트 파일 경로")
    p.add_argument("--output_dir",       required=True,
                   help="이미지 저장 디렉터리")
    p.add_argument("--nsamples",         type=int, default=4,
                   help="프롬프트 당 생성 이미지 수")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM 스텝 수 (기본 50)")
    p.add_argument("--cfg_scale",        type=float, default=7.5,
                   help="Text guidance scale")
    p.add_argument("--freedom",          action="store_true",
                   help="Classifier Guidance 활성화")
    p.add_argument("--freedom_model_args", default="configs/models/classifier.yaml",
                   help="분류기 설정 파일 (.yaml)")
    p.add_argument("--freedom_model_ckpt", required=False,
                   help="분류기 체크포인트 (.pth)")
    p.add_argument("--freedom_scale",    type=float, default=10.0,
                   help="Classifier Guidance scale")
    p.add_argument("--guide_start",      type=int, default=1,
                   help="몇 번째 스텝부터 guidance 적용")
    p.add_argument("--target_class",     type=int, default=1,
                   help="GuidanceModel 타깃 클래스 인덱스")
    p.add_argument("--device",           default="cuda:0",
                   help="연산 디바이스")
    p.add_argument("--use_fp16",         action="store_true",
                   help="VAE/UNet は FP16、その他は FP32")
    p.add_argument("--seed",             type=int, default=None,
                   help="랜덤 시드")
    return p.parse_args()


def load_prompts(path: str):
    if path.endswith(".txt"):
        return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    if path.endswith(".json"):
        js = json.load(open(path, encoding="utf-8"))
        return js.get("prompts", [x["prompt"] for x in js])
    import pandas as pd
    df = pd.read_csv(path)
    return df["prompt"].astype(str).tolist()


def save_image(img: Image.Image, out_dir: str, idx: int):
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"{idx:06d}.png"))


def main():
    args = parse_args()
    device = torch.device(args.device)

    # VAE/UNet は FP16 でも OK、それ以外 (text encoders & classifier) は FP32 固定
    unet_dtype = torch.float16 if args.use_fp16 else torch.float32
    vae_dtype  = unet_dtype

    # 1) SDXL VAE と UNet をロード (FP16 or FP32)
    vae = AutoencoderKL.from_pretrained(
        args.sdxl_base, subfolder="vae", torch_dtype=vae_dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.sdxl_base, subfolder="unet", torch_dtype=unet_dtype
    ).to(device)

    # 2) 토크나이저 & 텍스트 인코더 (항상 FP32)
    tokenizer     = CLIPTokenizer.from_pretrained(args.sdxl_base, subfolder="tokenizer")
    tokenizer_2   = CLIPTokenizer.from_pretrained(args.sdxl_base, subfolder="tokenizer_2")
    text_encoder  = CLIPTextModel.from_pretrained(
        args.sdxl_base, subfolder="text_encoder"
    ).to(device).float()
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        args.sdxl_base, subfolder="text_encoder_2"
    ).to(device).float()

    # 3) 커스텀 파이프라인 로드 & 컴포넌트 교체
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(
        args.sdxl_base,
        torch_dtype=unet_dtype,  # VAE/UNet 用
    ).to(device)
    pipe.vae            = vae
    pipe.unet           = unet
    pipe.tokenizer      = tokenizer
    pipe.tokenizer_2    = tokenizer_2
    pipe.text_encoder   = text_encoder
    pipe.text_encoder_2 = text_encoder2
    pipe.safety_checker = None
    pipe.feature_extractor = None

    # callback に渡すテンソル名を固定
    pipe._callback_tensor_inputs = ["latents", "noise_pred", "prev_latents"]

    # 4) DDIM 스케줄러 설정
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

    # 5) Optional: Classifier Guidance
    if args.freedom:
        guide = GuidanceModel(
            pipe,
            args.freedom_model_args,
            args.freedom_model_ckpt,
            target_class=args.target_class,
            device=device
        )
        # 분류器는 FP32
        guide.gradient_model.model = guide.gradient_model.model.to(device).float()

        def cb_fn(pipeline, step, timestep, cb_kwargs):
            if step >= args.guide_start:
                # Half→Float 変換
                for k in ("latents", "prev_latents"):
                    if k in cb_kwargs:
                        cb_kwargs[k] = cb_kwargs[k].float()
                # guidance
                out = guide.guidance(
                    pipeline, cb_kwargs, step, timestep,
                    scale=args.freedom_scale,
                    target_class=args.target_class
                )
                # Float→Half へ戻す
                if args.use_fp16:
                    for k in ("latents", "prev_latents", "noise_pred"):
                        if k in out:
                            out[k] = out[k].half()
                return out
            return cb_kwargs

        cb_tensors = pipe._callback_tensor_inputs
    else:
        cb_fn = None
        cb_tensors = None

    # 6) 생성 루프
    prompts = load_prompts(args.prompt_file)
    gen = torch.Generator(device=device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    idx = 0
    for prompt in prompts:
        # 毎回 timesteps リセット
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        pipe.scheduler._step_index = 0

        out = pipe(
            prompt=[prompt],
            height=1024, width=1024,
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

    # 最後に prompts.txt も書き出し
    with open(os.path.join(args.output_dir, "prompts.txt"), "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")


if __name__ == "__main__":
    main()
