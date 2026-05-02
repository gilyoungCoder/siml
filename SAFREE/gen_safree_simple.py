#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, argparse, hashlib, re
from typing import List
import torch
from PIL import Image
import pandas as pd
from diffusers import DPMSolverMultistepScheduler

# 커스텀 SAFREE 파이프라인 (이미 리포에 존재한다고 가정)
from models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline
from models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline

# -------------------------------
# Logger (modified_* 파이프라인이 logger.log(...)를 호출함)
# -------------------------------
class SimpleLogger:
    def __init__(self, path=None):
        self.fp = open(path, "a") if path else None
    def log(self, msg):
        print(str(msg))
        if self.fp:
            self.fp.write(str(msg) + "\n")
            self.fp.flush()
    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None

# -------------------------------
# 유틸
# -------------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def slugify(txt: str, maxlen: int = 60) -> str:
    txt = re.sub(r"\s+", "_", txt.strip())
    txt = re.sub(r"[^a-zA-Z0-9_\-]+", "", txt)
    return txt[:maxlen] if maxlen else txt

def read_prompts(args) -> List[str]:
    """
    우선순위:
      1) --prompt (단일 프롬프트)
      2) --txt (한 줄 = 한 프롬프트)
      3) --csv (--csv-col 컬럼)
    """
    if args.prompt:
        return [args.prompt]

    if args.txt:
        if not os.path.isfile(args.txt):
            raise FileNotFoundError(f"TXT not found: {args.txt}")
        with open(args.txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError(f"TXT is empty: {args.txt}")
        return lines

    if args.csv:
        if not os.path.isfile(args.csv):
            raise FileNotFoundError(f"CSV not found: {args.csv}")
        df = pd.read_csv(args.csv)
        if args.csv_col not in df.columns:
            raise KeyError(f"CSV column '{args.csv_col}' not found. Columns: {list(df.columns)}")
        prompts = [str(x) for x in df[args.csv_col].tolist() if isinstance(x, str) and x.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in column '{args.csv_col}' of CSV: {args.csv}")
        return prompts

    raise ValueError("입력 필요: --prompt 또는 --txt 또는 --csv")

def default_negative_prompt_for_nudity() -> str:
    terms = [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
    ]
    # terms = [
    #     "violence", "assault", "attack", "fight", "beating", "punching",
    #     "kicking", "hitting", "stabbing", "knife attack", "shooting", "gun violence",
    #     "aiming a gun", "strangling", "choking", "execution", "blood", "bloody scene",
    #     "open wound", "corpse", "explosion", "grenade blast", "bomb blast", "molotov cocktail",
    #     "arson",
    # ]
    return ", ".join(terms)

def seed_to_generator(seed: int, device: str):
    g = torch.Generator(device=device)
    if seed is not None and seed >= 0:
        g.manual_seed(seed)
    else:
        g.seed()
    return g

def build_pipeline(model_id: str, device: str):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    if "xl" in model_id.lower():
        pipe = ModifiedStableDiffusionXLPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=torch.float16
        )
    else:
        pipe = ModifiedStableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=torch.float16, revision="fp16"
        )
        # SD v1-4 등 NSFW 필터 비활성화(가능 시)
        for attr in ("safety_checker", "feature_extractor", "image_encoder"):
            if hasattr(pipe, attr):
                try:
                    setattr(pipe, attr, None)
                except Exception:
                    pass
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe

# -------------------------------
# 메인 실행
# -------------------------------
def run_generation(args):
    ensure_dir(args.outdir)
    prompts = read_prompts(args)

    log_path = os.path.join(args.outdir, "logs.txt")
    logger = SimpleLogger(log_path)

    pipe = build_pipeline(args.model_id, args.device)

    # Latent Re-Attention + FreeU 훅 (선택)
    if args.safree and args.lra:
        from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
        b1, b2, s1, s2 = [float(x) for x in args.freeu_hyp.split("-")]
        register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
        register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)

    neg_prompt = args.negative_prompt or (default_negative_prompt_for_nudity() if args.use_default_negative else None)
    neg_space = [t.strip() for t in default_negative_prompt_for_nudity().split(",")] if args.use_default_negative else None

    base_gen = seed_to_generator(args.seed, args.device)

    print(f"[INFO] count={len(prompts)}  images/each={args.num_images}  steps={args.steps}  guidance={args.guidance}")
    print(f"[INFO] model={args.model_id}  device={args.device}  outdir={args.outdir}")
    if args.safree:
        print(f"[INFO] SAFREE: lra={args.lra}, svf={args.svf}, sf_alpha={args.sf_alpha}, re_attn_t={args.re_attn_t}, up_t={args.up_t}, freeu_hyp={args.freeu_hyp}")

    for i, p in enumerate(prompts):
        gen = base_gen
        # --- 강제 토큰 컷 (77 제한) ---
        tok = pipe.tokenizer(
            [p],
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )
        p = pipe.tokenizer.decode(tok.input_ids[0], skip_special_tokens=True)
        # --------------------------------
        if args.per_prompt_seed:
            h = int(hashlib.sha256(p.encode("utf-8")).hexdigest(), 16) % (2**31)
            gen = seed_to_generator(h, args.device)
        elif args.linear_per_prompt_seed:
            gen = seed_to_generator(args.seed + i, args.device)

        if "xl" in args.model_id.lower():
            result = pipe(
                p,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                negative_prompt=neg_prompt,
                negative_prompt_space=neg_space,
                height=args.height,
                width=args.width,
                generator=gen,
                safree=args.safree,
                safree_dict={
                    "re_attn_t": [int(t) for t in args.re_attn_t.split(",")],
                    "alpha": args.sf_alpha,
                    "svf": args.svf,
                    "logger": logger,   # 중요: None 금지
                    "up_t": args.up_t,
                    "category": "nudity"
                },
            )
            images = result.images
        else:
            result = pipe(
                p,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                negative_prompt=neg_prompt,
                negative_prompt_space=neg_space,
                height=args.height,
                width=args.width,
                generator=gen,
                safree_dict={
                    "re_attn_t": [int(t) for t in args.re_attn_t.split(",")],
                    "alpha": args.sf_alpha,
                    "logger": logger,   # 중요: None 금지
                    "safree": args.safree,
                    "svf": args.svf,
                    "lra": args.lra,
                    "up_t": args.up_t,
                    "category": "nudity",
                },
            )
            images = result if isinstance(result, list) else getattr(result, "images", result)

        for k, img in enumerate(images):
            if not isinstance(img, Image.Image):
                try:
                    img = img.convert("RGB")
                except Exception:
                    img = Image.fromarray(img)
            name = f"{i:05d}_{k:02d}_{slugify(p)}.png"
            save_path = os.path.join(args.outdir, name)
            img.save(save_path)
            print(f"[SAVE] {save_path}")

    logger.close()
    print("[DONE]")

# -------------------------------
# CLI
# -------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="SAFREE generation-only (TXT/CSV/one-liner)")
    # 입력
    ap.add_argument("--prompt", type=str, default=None, help="단일 프롬프트")
    ap.add_argument("--txt", type=str, default=None, help="TXT(한 줄=한 프롬프트)")
    ap.add_argument("--csv", type=str, default=None, help="CSV 경로")
    ap.add_argument("--csv-col", type=str, default="prompt", help="CSV 컬럼명")

    # 모델/출력
    ap.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--outdir", type=str, default="./results/safree_out")

    # 생성 하이퍼파라미터
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_prompt_seed", action="store_true")
    ap.add_argument("--linear_per_prompt_seed", action="store_true", help="seed = args.seed + i (matches baseline/ours)")

    # 해상도
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)

    # SAFREE
    ap.add_argument("--safree", action="store_true")
    ap.add_argument("--svf", action="store_true")
    ap.add_argument("--lra", action="store_true")
    ap.add_argument("--sf_alpha", type=float, default=0.01)
    ap.add_argument("--re_attn_t", type=str, default="-1,4")  # 예: "-1,4"
    ap.add_argument("--up_t", type=int, default=10)
    ap.add_argument("--freeu_hyp", type=str, default="1.0-1.0-0.9-0.2")

    # 네거티브 프롬프트
    ap.add_argument("--negative_prompt", type=str, default=None)
    ap.add_argument("--use_default_negative", action="store_true")

    # 디바이스
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()
    run_generation(args)

if __name__ == "__main__":
    main()
