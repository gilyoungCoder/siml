#!/usr/bin/env python
"""
MJA Exemplar Image Generation (image-only).

Generates family-grouped target/anchor exemplar images for one MJA concept on
one backbone (SD1.4 / SD3 medium / FLUX.1-dev). CLIP feature extraction is a
separate downstream step so this script stays backbone-agnostic beyond the
diffusion pipeline choice.

Layout produced:

    {output_root}/{concept}/{backbone}/
        {family_name}/
            target_00.png ... target_NN.png
            anchor_00.png ... anchor_NN.png
        generation_log.json

Usage:
    python -m safegen.gen_mja_exemplar_images \
        --concept sexual \
        --backbone sd14 \
        --concept_pack SafeGen/configs/concept_packs/mja_sexual \
        --output_root CAS_SpatialCFG/exemplars/mja_v1
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm


BACKBONE_CONFIG = {
    "sd14": {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "resolution": 512,
        "steps": 50,
        "cfg_scale": 7.5,
        "cpu_offload": False,
    },
    "sd3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "resolution": 1024,
        "steps": 28,
        "cfg_scale": 7.0,
        "cpu_offload": True,
    },
    "flux1": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "resolution": 1024,
        "steps": 28,
        "cfg_scale": 3.5,
        "cpu_offload": True,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_families(pack_dir: Path) -> List[dict]:
    with (pack_dir / "families.json").open() as f:
        data = json.load(f)
    fams = data.get("families", [])
    pilot = [f for f in fams if f.get("pilot", False)]
    return pilot if pilot else fams


def load_prompts(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def partition_prompts(all_prompts: List[str], n_families: int) -> List[List[str]]:
    """Distribute prompts evenly across families in order."""
    per = len(all_prompts) // n_families
    if per == 0:
        raise ValueError(
            f"Only {len(all_prompts)} prompts for {n_families} families — need at least {n_families}."
        )
    chunks = []
    for i in range(n_families):
        start = i * per
        end = start + per if i < n_families - 1 else len(all_prompts)
        chunks.append(all_prompts[start:end])
    return chunks


def load_pipeline(backbone: str, device: torch.device, dtype: torch.dtype):
    cfg = BACKBONE_CONFIG[backbone]
    model_id = cfg["model_id"]
    print(f"[{backbone}] loading {model_id} ...")
    if backbone == "sd14":
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None,
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
    elif backbone == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.enable_model_cpu_offload() if cfg["cpu_offload"] else pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
    elif backbone == "flux1":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.enable_model_cpu_offload() if cfg["cpu_offload"] else pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return pipe


def run_one(pipe, backbone: str, prompt: str, seed: int,
            device: torch.device, resolution: int, steps: int, cfg_scale: float) -> Image.Image:
    cfg = BACKBONE_CONFIG[backbone]
    gen_device = "cpu" if cfg["cpu_offload"] else device
    gen = torch.Generator(device=gen_device).manual_seed(seed)
    if backbone == "sd14":
        img = pipe(
            prompt, num_inference_steps=steps, guidance_scale=cfg_scale,
            height=resolution, width=resolution, generator=gen,
        ).images[0]
    elif backbone == "sd3":
        img = pipe(
            prompt, negative_prompt="", num_inference_steps=steps,
            guidance_scale=cfg_scale, height=resolution, width=resolution, generator=gen,
        ).images[0]
    elif backbone == "flux1":
        # FLUX.1-dev: embedded guidance, no negative prompt
        img = pipe(
            prompt, num_inference_steps=steps, guidance_scale=cfg_scale,
            height=resolution, width=resolution, generator=gen,
            max_sequence_length=512,
        ).images[0]
    else:
        raise ValueError(backbone)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True,
                   choices=["sexual", "violent", "disturbing", "illegal"])
    p.add_argument("--backbone", required=True, choices=list(BACKBONE_CONFIG.keys()))
    p.add_argument("--concept_pack", required=True,
                   help="Path to concept pack dir, e.g. SafeGen/configs/concept_packs/mja_sexual")
    p.add_argument("--output_root", required=True,
                   help="Root output dir, e.g. CAS_SpatialCFG/exemplars/mja_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=None,
                   help="Override resolution (default: backbone native)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip prompts whose image file already exists")
    args = p.parse_args()

    bb_cfg = BACKBONE_CONFIG[args.backbone]
    resolution = args.resolution or bb_cfg["resolution"]
    steps = args.steps or bb_cfg["steps"]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else bb_cfg["cfg_scale"]

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack_dir = Path(args.concept_pack)
    families = load_families(pack_dir)
    target_prompts = load_prompts(pack_dir / "target_prompts.txt")
    anchor_prompts = load_prompts(pack_dir / "anchor_prompts.txt")
    if len(target_prompts) != len(anchor_prompts):
        raise ValueError(
            f"target/anchor count mismatch: {len(target_prompts)} vs {len(anchor_prompts)}"
        )

    t_chunks = partition_prompts(target_prompts, len(families))
    a_chunks = partition_prompts(anchor_prompts, len(families))

    out_dir = Path(args.output_root) / args.concept / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MJA Exemplar Generation")
    print(f"  concept     = {args.concept}")
    print(f"  backbone    = {args.backbone}  ({bb_cfg['model_id']})")
    print(f"  resolution  = {resolution}, steps={steps}, cfg={cfg_scale}")
    print(f"  families    = {[f['name'] for f in families]}")
    print(f"  prompts/fam = target:{[len(c) for c in t_chunks]} anchor:{[len(c) for c in a_chunks]}")
    print(f"  output      = {out_dir}")
    print("=" * 70)

    pipe = load_pipeline(args.backbone, device, dtype)
    print("[pipe] ready\n")

    log: Dict = {
        "concept": args.concept,
        "backbone": args.backbone,
        "model_id": bb_cfg["model_id"],
        "resolution": resolution,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed_base": args.seed,
        "dtype": args.dtype,
        "families": {},
    }

    t0 = time.time()
    for fi, family in enumerate(families):
        fname = family["name"]
        fam_dir = out_dir / fname
        fam_dir.mkdir(exist_ok=True)
        tps, aps = t_chunks[fi], a_chunks[fi]
        print(f"[{fi + 1}/{len(families)}] {fname}  target={len(tps)} anchor={len(aps)}")

        fam_log = {"target": [], "anchor": []}
        for pi, prompt in enumerate(tqdm(tps, desc=f"  target/{fname}")):
            out_path = fam_dir / f"target_{pi:02d}.png"
            seed = args.seed + fi * 1000 + pi
            if args.skip_existing and out_path.exists():
                fam_log["target"].append({"idx": pi, "prompt": prompt, "seed": seed, "skipped": True})
                continue
            img = run_one(pipe, args.backbone, prompt, seed, device, resolution, steps, cfg_scale)
            img.save(out_path)
            fam_log["target"].append({"idx": pi, "prompt": prompt, "seed": seed})

        for pi, prompt in enumerate(tqdm(aps, desc=f"  anchor/{fname}")):
            out_path = fam_dir / f"anchor_{pi:02d}.png"
            seed = args.seed + fi * 1000 + pi + 500
            if args.skip_existing and out_path.exists():
                fam_log["anchor"].append({"idx": pi, "prompt": prompt, "seed": seed, "skipped": True})
                continue
            img = run_one(pipe, args.backbone, prompt, seed, device, resolution, steps, cfg_scale)
            img.save(out_path)
            fam_log["anchor"].append({"idx": pi, "prompt": prompt, "seed": seed})

        log["families"][fname] = fam_log

    elapsed = time.time() - t0
    log["elapsed_sec"] = round(elapsed, 1)
    (out_dir / "generation_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))

    n_imgs = sum(len(t_chunks[i]) + len(a_chunks[i]) for i in range(len(families)))
    print(f"\n[done] {args.concept}/{args.backbone}: {n_imgs} images in {elapsed:.1f}s")
    print(f"  log: {out_dir / 'generation_log.json'}")


if __name__ == "__main__":
    main()
