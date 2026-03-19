#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, hashlib, re
from typing import List
import torch
from PIL import Image
import pandas as pd
from diffusers import DPMSolverMultistepScheduler

# 커스텀 SAFREE 파이프라인
from models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline
from models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline

# =========================================================
# 수치 일관성: TF32 OFF (결과 수렴성 향상)
# =========================================================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# -------------------------------
# Logger
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

# -------------------------------
# 기본 네거티브 리스트
# -------------------------------
def simple_default_negative_terms():
    return [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
    ]

def simple_default_negative_prompt_str():
    return ", ".join(simple_default_negative_terms())

# -------------------------------
# 다중 개념
# -------------------------------
def builtin_toxic_terms():
    return {
        "nudity": simple_default_negative_terms(),
        "violence": [
            "violence","gore","blood","mutilation","torture","assault","murder","beheading",
            "explosion","gunfight","execution","brutality","massacre"
        ],
        "terrorism": [
            "terrorism","terrorist attack","bomb","suicide bombing","hostage","IED","extremist"
        ],
        "animal_abuse": [
            "animal abuse","animal cruelty","poaching","trophy hunting","vivisection"
        ],
        "van_gogh": ["VanGogh"]
    }

def load_terms_from_file(path: str):
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.strip()
            if t:
                terms.append(t)
    return terms

def build_toxic_terms(args):
    merged, seen = [], set()
    if args.categories:
        cats = [c.strip().lower() for c in args.categories.split(",") if c.strip()]
        base = builtin_toxic_terms()
        for c in cats:
            if c in base:
                for t in base[c]:
                    tt = t.strip()
                    key = tt.lower()
                    if tt and key not in seen:
                        merged.append(tt); seen.add(key)
            else:
                print(f"[WARN] Unknown category: {c}")
    if args.toxic_list:
        for t in args.toxic_list.split(","):
            tt = t.strip()
            key = tt.lower()
            if tt and key not in seen:
                merged.append(tt); seen.add(key)
    if args.toxic_file:
        for tt in load_terms_from_file(args.toxic_file):
            key = tt.lower()
            if tt and key not in seen:
                merged.append(tt); seen.add(key)
    return merged

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
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        try:
            if hasattr(pipe, "config") and hasattr(pipe.config, "requires_safety_checker"):
                pipe.config.requires_safety_checker = False
        except Exception:
            pass
        for attr in ("feature_extractor", "image_encoder"):
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

    try:
        pipe = build_pipeline(args.model_id, args.device)

        if args.safree and args.lra:
            try:
                from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
                b1, b2, s1, s2 = [float(x) for x in args.freeu_hyp.split("-")]
                register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
                register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
            except Exception as e:
                logger.log(f"[FreeU] skipped: {e}")

        if args.force_simple_neg:
            toxic_terms = simple_default_negative_terms()
            negative_prompt_space = toxic_terms
            negative_prompt = simple_default_negative_prompt_str() if args.use_default_negative and not args.negative_prompt else args.negative_prompt
            category_meta = ["nudity"]
        else:
            toxic_terms = build_toxic_terms(args)
            negative_prompt_space = toxic_terms if toxic_terms else None
            if args.negative_prompt is not None:
                negative_prompt = args.negative_prompt
            elif args.use_default_negative:
                negative_prompt = ", ".join(toxic_terms) if toxic_terms else None
            else:
                negative_prompt = None
            category_meta = [c.strip() for c in args.categories.split(",")] if args.categories else []

        base_gen = seed_to_generator(args.seed, args.device)

        print(f"[INFO] count={len(prompts)}  images/each={args.num_images}  steps={args.steps}  guidance={args.guidance}")
        print(f"[INFO] model={args.model_id}  device={args.device}  outdir={args.outdir}")
        if args.safree:
            print(f"[INFO] SAFREE: lra={args.lra}, svf={args.svf}, sf_alpha={args.sf_alpha}, re_attn_t={args.re_attn_t}, up_t={args.up_t}, freeu_hyp={args.freeu_hyp}")
            print(f"[INFO] categories={category_meta}  toxic_terms={len(toxic_terms)}")

        re_attn_ts = [int(t) for t in args.re_attn_t.split(",")] if args.re_attn_t else []

        for i, p in enumerate(prompts):
            # --- 강제 토큰 컷 (77 제한) ---
            tok = pipe.tokenizer(
                [p],
                truncation=True,
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt"
            )
            p = pipe.tokenizer.decode(tok.input_ids[0], skip_special_tokens=True)
            # --------------------------------

            gen = base_gen
            if args.per_prompt_seed:
                h = int(hashlib.sha256(p.encode("utf-8")).hexdigest(), 16) % (2**31)
                gen = seed_to_generator(h, args.device)

            common_kwargs = dict(
                prompt=p,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                negative_prompt=negative_prompt,
                negative_prompt_space=negative_prompt_space,
                height=args.height,
                width=args.width,
                generator=gen,
                safree_dict={
                    "enabled": bool(args.safree),
                    "safree": bool(args.safree),
                    "re_attn_t": re_attn_ts,
                    "alpha": args.sf_alpha,
                    "svf": args.svf,
                    "lra": args.lra,
                    "up_t": args.up_t,
                    "logger": logger,
                    "category": category_meta,
                    "multi_concept": True if (category_meta and len(category_meta) > 1) else False
                },
            )

            result = pipe(**common_kwargs)
            images = getattr(result, "images", result)
            if not isinstance(images, list):
                images = [images]

            for k, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    try:
                        img = img.convert("RGB")
                    except Exception:
                        img = Image.fromarray(img)
                p_hash = hashlib.md5(p.encode("utf-8")).hexdigest()[:8]
                name = f"{i:05d}_{k:02d}_{slugify(p)}_{p_hash}.png"
                save_path = os.path.join(args.outdir, name)
                img.save(save_path)
                print(f"[SAVE] {save_path}")

        print("[DONE]")
    finally:
        logger.close()

# -------------------------------
# CLI
# -------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="SAFREE generation-only (TXT/CSV/one-liner)")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--txt", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--csv-col", type=str, default="prompt")
    ap.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--outdir", type=str, default="./results/safree_out")
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_prompt_seed", action="store_true")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--safree", action="store_true")
    ap.add_argument("--svf", action="store_true")
    ap.add_argument("--lra", action="store_true")
    ap.add_argument("--sf_alpha", type=float, default=0.01)
    ap.add_argument("--re_attn_t", type=str, default="-1,4")
    ap.add_argument("--up_t", type=int, default=10)
    ap.add_argument("--freeu_hyp", type=str, default="1.0-1.0-0.9-0.2")
    ap.add_argument("--categories", type=str, default="nudity")
    ap.add_argument("--toxic_list", type=str, default=None)
    ap.add_argument("--toxic_file", type=str, default=None)
    ap.add_argument("--negative_prompt", type=str, default=None)
    ap.add_argument("--use_default_negative", action="store_true")
    ap.add_argument("--force_simple_neg", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()
    run_generation(args)

if __name__ == "__main__":
    main()
