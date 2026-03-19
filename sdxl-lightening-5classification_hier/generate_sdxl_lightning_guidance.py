#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## generate_sdxl_lightning_guidance.py

import os
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
from accelerate import Accelerator

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from geo_utils.custom_sdxl_lightning import CustomStableDiffusionXLPipeline
from geo_utils.guidance_utils import GuidanceModel

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def parse_args():
    p = ArgumentParser("SDXL-Lightning + Classifier Guidance")
    p.add_argument('lightning_repo', type=str,
                   help="Lightning UNet의 HF repo ID (예: ByteDance/SDXL-Lightning)")
    p.add_argument('--revision', type=str, default=None,
                   help="(선택) Lightning repo 의 revision")
    p.add_argument('--sdxl_base', type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="기본 SDXL 모델 ID")
    p.add_argument('--prompt_file', type=str, required=True, help='.txt 로 된 프롬프트 파일')
    p.add_argument('--output_dir', type=str, required=True, help='출력 폴더')
    p.add_argument('--nsamples', type=int, default=4, help='샘플 수')
    p.add_argument('--num_inference_steps', type=int, default=4,
                   help='Lightning distilled step (예: 4)')
    p.add_argument('--cfg_scale', type=float, default=7.5,
                   help='텍스트 guidance scale (Lightning은 항상 0 추천)')
    p.add_argument('--freedom', action='store_true', help='Classifier guidance 활성화')
    p.add_argument('--freedom_model_args', type=str,
                   default="configs/models/classifier.yaml", help='Guidance 모델 설정 파일')
    p.add_argument('--freedom_model_ckpt', type=str, required='--freedom' in os.sys.argv,
                   help='Guidance classifier .pth 체크포인트')
    p.add_argument('--freedom_scale', type=float, default=10.0, help='Classifier guidance 세기')
    p.add_argument('--guide_start', type=int, default=1, help='Guidance 시작 스텝 인덱스')
    p.add_argument('--device', type=str, default='cuda:0', help='디바이스')
    p.add_argument('--use_fp16', action='store_true', help='FP16 사용')
    p.add_argument('--seed', type=int, default=None, help='시드')
    return p.parse_args()


def load_prompts(path: str):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def save_image(img: Image.Image, out_dir: str, idx: int):
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"{idx:06d}.png"))


def main():
    args = parse_args()
    accel  = Accelerator(mixed_precision='fp16' if args.use_fp16 else 'no')
    device = torch.device(args.device)
    dtype  = torch.float16 if args.use_fp16 else torch.float32

    # 1) SDXL-base VAE / Tokenizer / TextEncoders 로드
    vae = AutoencoderKL.from_pretrained(
        args.sdxl_base, subfolder="vae"
    ).to(device, dtype=dtype)

    tokenizer   = CLIPTokenizer.from_pretrained(args.sdxl_base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.sdxl_base, subfolder="tokenizer_2")

    text_encoder   = CLIPTextModel.from_pretrained(
        args.sdxl_base, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.sdxl_base, subfolder="text_encoder_2", torch_dtype=dtype
    )

    # 2) Lightning UNet 체크포인트 덮어쓰기
    ckpt_name = f"sdxl_lightning_{args.num_inference_steps}step_unet.safetensors"
    ckpt_path = hf_hub_download(repo_id=args.lightning_repo,
                                filename=ckpt_name, revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.sdxl_base, subfolder="unet", torch_dtype=torch.float32
    )
    state = load_file(ckpt_path, "cpu")
    unet.load_state_dict(state, strict=True)

    # 3) 커스텀 파이프라인 인스턴스화
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(
        args.sdxl_base,
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=unet,
        image_encoder=None,
        feature_extractor=None,
        torch_dtype=dtype,
    )

    # → **여기서** 모든 모듈을 원하는 device & dtype 으로 강제 캐스트
    pipe = pipe.to(device)
    pipe.vae            = pipe.vae.to(device, dtype=dtype)
    pipe.unet           = pipe.unet.to(device, dtype=dtype)
    pipe.text_encoder   = pipe.text_encoder.to(device, dtype=dtype)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(device, dtype=dtype)

    # 4) Scheduler trailing + Lightning step 고정
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 5) Classifier-guidance 준비
    if args.freedom:
        guide = GuidanceModel(
            pipe,
            args.freedom_model_args,
            args.freedom_model_ckpt,
            target_class=1,
            device=device
        )
        def cg_callback(pipeline, step, t, cb_kwargs):
            t_idx    = int(t)
            t_tensor = torch.tensor([t_idx], device=pipeline._execution_device, dtype=torch.long)
            if step >= args.guide_start:
                return guide.guidance(
                    pipeline, cb_kwargs, step, t_tensor,
                    scale=args.freedom_scale,
                    target_class=1
                )
            return cb_kwargs

        cb_fn      = partial(cg_callback)
        cb_tensors = ["latents", "prev_latents", "noise_pred"]
    else:
        cb_fn, cb_tensors = None, None

    # 6) 이미지 생성 루프
    prompts = load_prompts(args.prompt_file)
    gen     = torch.Generator(device=device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    idx = 0
    for prompt in prompts:
        pipe.scheduler._step_index = 0
        out = pipe(
            prompt=[prompt],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=0,
            num_images_per_prompt=args.nsamples,
            generator=gen,
            callback_on_step_end=cb_fn,
            callback_on_step_end_tensor_inputs=cb_tensors,
            callback_steps=1,
        )
        for img in out.images:
            save_image(img, args.output_dir, idx)
            idx += 1

    # prompts.txt 함께 저장
    with open(os.path.join(args.output_dir, "prompts.txt"), "w", encoding='utf-8') as f:
        for p in prompts:
            f.write(p + "\n")


if __name__ == "__main__":
    main()
