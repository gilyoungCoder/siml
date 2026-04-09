"""
RECE (Reliable and Efficient Concept Erasure) — Image Generation
Standalone script that loads RECE fine-tuned UNet and generates images.
Based on: unlearning-baselines/RECE/execs/generate_images.py
"""
import torch, os, argparse, pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def load_prompts(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        col = 'prompt' if 'prompt' in df.columns else df.columns[0]
        return df[col].tolist()
    else:
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

def main(args):
    # Load base SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Load RECE fine-tuned UNet
    # RECE checkpoint is a full pipeline state dict
    print(f"Loading RECE checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if isinstance(ckpt, dict) and "unet" in ckpt:
        pipe.unet.load_state_dict(ckpt["unet"])
    elif isinstance(ckpt, dict) and any("model" in k for k in ckpt.keys()):
        # Try loading as state dict directly
        pipe.unet.load_state_dict(ckpt, strict=False)
    else:
        # RECE saves the full pipeline - try loading UNet from it
        try:
            pipe.unet.load_state_dict(ckpt)
        except:
            print("Trying to load as full pipeline checkpoint...")
            # The checkpoint might be a diffusers-style save
            unet = UNet2DConditionModel.from_pretrained(
                args.ckpt, torch_dtype=torch.float16
            )
            pipe.unet = unet

    pipe = pipe.to(args.device)

    prompts = load_prompts(args.prompts)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Generating {len(prompts)} images...")
    for i, prompt in enumerate(prompts):
        outpath = os.path.join(args.outdir, f"{i:04d}_00.png")
        if os.path.exists(outpath):
            continue
        try:
            gen = torch.Generator(args.device).manual_seed(args.seed)
            img = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=gen,
            ).images[0]
            img.save(outpath)
        except Exception as e:
            print(f"  Error {i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)} done")

    print(f"Done! {len([f for f in os.listdir(args.outdir) if f.endswith('.png')])} images in {args.outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="RECE checkpoint path (.pt)")
    p.add_argument("--prompts", required=True, help="Prompt file (.txt or .csv)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    main(p.parse_args())
