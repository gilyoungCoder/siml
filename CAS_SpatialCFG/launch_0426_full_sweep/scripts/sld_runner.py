"""SLD generator — minimal CLI wrapping SLDPipeline.

Usage:
  python sld_runner.py --prompts <txt> --outdir <dir> --variant {Max,Medium,Strong,Weak} \
                       --steps <N> --seed 42 --cfg_scale 7.5

Reads prompt list (one per line, blank lines skipped), generates 1 image per prompt
using the SLD variant's hyper-params, saves as <outdir>/{idx:04d}.png.

The package is loaded via sys.path injection from the local safe-latent-diffusion
checkout, so no global pip install is required.
"""
import argparse, sys
from pathlib import Path

SLD_SRC = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/safe-latent-diffusion/src"
if SLD_SRC not in sys.path:
    sys.path.insert(0, SLD_SRC)

import torch
from sld import SLDPipeline

VARIANTS = {
    "Max": dict(sld_guidance_scale=5000, sld_warmup_steps=0,
                sld_threshold=1.0, sld_momentum_scale=0.5, sld_mom_beta=0.7),
    "Medium": dict(sld_guidance_scale=1000, sld_warmup_steps=10,
                   sld_threshold=0.01, sld_momentum_scale=0.3, sld_mom_beta=0.4),
    "Strong": dict(sld_guidance_scale=2000, sld_warmup_steps=7,
                   sld_threshold=0.025, sld_momentum_scale=0.5, sld_mom_beta=0.7),
    "Weak": dict(sld_guidance_scale=200, sld_warmup_steps=15,
                 sld_threshold=0.005, sld_momentum_scale=0.0, sld_mom_beta=0.0),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    # Optional overrides for scale-robustness sweep:
    p.add_argument("--sld_guidance_scale", type=float, default=None)
    p.add_argument("--sld_warmup_steps", type=int, default=None)
    p.add_argument("--sld_threshold", type=float, default=None)
    p.add_argument("--sld_momentum_scale", type=float, default=None)
    p.add_argument("--sld_mom_beta", type=float, default=None)
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    prompts = [ln.strip() for ln in open(args.prompts) if ln.strip()]
    print(f"[sld] variant={args.variant} steps={args.steps} prompts={len(prompts)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = SLDPipeline.from_pretrained(args.model, safety_checker=None,
                                       feature_extractor=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    vp = dict(VARIANTS[args.variant])
    # Apply individual overrides (used by scale-robustness sweep)
    for k, default_key in [("sld_guidance_scale", "sld_guidance_scale"),
                            ("sld_warmup_steps", "sld_warmup_steps"),
                            ("sld_threshold", "sld_threshold"),
                            ("sld_momentum_scale", "sld_momentum_scale"),
                            ("sld_mom_beta", "sld_mom_beta")]:
        v = getattr(args, k)
        if v is not None:
            vp[default_key] = v
    print(f"[sld] effective params: {vp}")
    for i, prompt in enumerate(prompts):
        out_path = Path(args.outdir) / f"{i:04d}.png"
        if out_path.exists():
            continue
        gen = torch.Generator(device=device).manual_seed(args.seed)
        out = pipe(prompt=prompt, generator=gen,
                   guidance_scale=args.cfg_scale,
                   num_inference_steps=args.steps,
                   height=args.height, width=args.width,
                   num_images_per_prompt=1, **vp)
        out.images[0].save(out_path)

    print(f"[sld] done {len(list(Path(args.outdir).glob('*.png')))} imgs -> {args.outdir}")


if __name__ == "__main__":
    main()
