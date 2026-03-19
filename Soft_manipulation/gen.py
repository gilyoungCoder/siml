# sd14_from_file.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Stable Diffusion v1-4: generate images from a prompt file")
    p.add_argument("--model", default="CompVis/stable-diffusion-v1-4", help="HF model id or local path")
    p.add_argument("--prompt_file", required=True, help="Path to a text file: one prompt per line")
    p.add_argument("--output_dir", default="outputs/sd14", help="Output directory")
    p.add_argument("--negative_prompt", default="nudity",
                   help='Negative prompt (default: "nudity"). Use "" to disable.')
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234, help="Base seed (each prompt offsets by its index)")
    p.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto",
                   help="Computation dtype (auto=fp16 on CUDA else fp32)")
    p.add_argument("--max_prompts", type=int, default=0, help="If >0, limit number of prompts")
    return p.parse_args()


def read_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # drop blanks and comments (# ...)
    prompts = [ln for ln in lines if ln and not ln.lstrip().startswith("#")]
    return prompts


def sanitize_filename(text: str, maxlen: int = 80):
    # Keep it readable but filesystem-safe
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-.,+=@()\[\]]+", "_", text)
    return (text[:maxlen]).rstrip("_")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(args.prompt_file)
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    print("=" * 70)
    print("Stable Diffusion v1-4: batch generation")
    print(f"Model:            {args.model}")
    print(f"Prompts file:     {args.prompt_file}  (N={len(prompts)})")
    print(f"Output dir:       {outdir.resolve()}")
    print(f"Device / dtype:   {device} / {str(torch_dtype).replace('torch.', '')}")
    print(f"Steps / Scale:    {args.num_inference_steps} / {args.guidance_scale}")
    print(f"Size (HxW):       {args.height}x{args.width}")
    print(f"Negative prompt:  {repr(args.negative_prompt)}")
    print(f"Base seed:        {args.seed}")
    print("=" * 70)

    # Safety checker는 기본 활성화(공식 파이프라인 기본값)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    # 간단한 메모리 최적화(선택)
    if device == "cuda":
        pipe.enable_attention_slicing("max")

    # 생성 루프
    for idx, prompt in enumerate(prompts, start=1):
        # 재현 가능성: 각 프롬프트마다 고유 시드
        g = torch.Generator(device=device).manual_seed(args.seed + idx)

        neg = None if args.negative_prompt == "" else args.negative_prompt

        print(f"[{idx:03d}/{len(prompts)}] prompt={prompt!r} | negative={neg!r}")
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=g,
        )

        image = result.images[0]
        base = sanitize_filename(prompt) or f"prompt_{idx:03d}"
        save_path = outdir / f"{idx:03d}_{base}.png"
        image.save(save_path)
        print(f" -> saved: {save_path}")

    print("All done.")


if __name__ == "__main__":
    main()
