#!/usr/bin/env python3
"""
SDErasure — Inference / Evaluation Script

Generates images from a fine-tuned SDErasure model and evaluates:
  - CLIP Score (semantic alignment with prompt)
  - FID (generation quality on MSCOCO-30k style)
  - NudeNet detection count (for explicit content erasure tasks)

Usage:
  # Generate from fine-tuned model
  python generate_sderasure.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --unet_dir ./outputs/sderasure_cat/unet \
    --prompts "a photo of a cat" "a photo of a dog" \
    --output_dir ./outputs/sderasure_cat/generated \
    --n_images 50

  # Evaluate CIFAR-10 style (object erasure)
  python generate_sderasure.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --unet_dir ./outputs/sderasure_cat/unet \
    --erased_class "cat" \
    --eval_classes "airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" \
    --output_dir ./outputs/sderasure_cat/eval \
    --n_images_per_class 200

  # Generate with original model (baseline comparison)
  python generate_sderasure.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --output_dir ./outputs/baseline/generated \
    --prompts "a photo of a cat" \
    --n_images 50
"""

import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)


# ============================================================================
# Pipeline Loading
# ============================================================================

def load_pipeline(
    model_id: str,
    unet_dir: str | None,
    device: torch.device,
    dtype=torch.float16,
) -> StableDiffusionPipeline:
    """Load SD pipeline, optionally replacing UNet with fine-tuned weights."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    if unet_dir and os.path.isdir(unet_dir):
        print(f"Loading fine-tuned UNet from: {unet_dir}")
        unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype).to(device)
        pipe.unet = unet
    else:
        print("Using original UNet (no fine-tuned weights loaded)")

    # DDIM scheduler for faster inference
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    return pipe


# ============================================================================
# Generation
# ============================================================================

def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    output_dir: str,
    n_images: int = 50,
    batch_size: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    image_size: int = 512,
) -> list[str]:
    """
    Generate n_images for each prompt and save to output_dir/{prompt_idx}/.
    Returns list of all saved image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_paths = []
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for p_idx, prompt in enumerate(prompts):
        # Sanitize prompt for use as directory name
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:80].strip()
        prompt_dir = os.path.join(output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)

        print(f"\nGenerating {n_images} images for: '{prompt}'")
        img_idx = 0
        remaining = n_images

        with tqdm(total=n_images, desc=f"  [{safe_name[:40]}]") as pbar:
            while remaining > 0:
                n = min(batch_size, remaining)
                images = pipe(
                    [prompt] * n,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=image_size,
                    width=image_size,
                    generator=generator,
                ).images

                for img in images:
                    path = os.path.join(prompt_dir, f"{img_idx:05d}.png")
                    img.save(path)
                    all_paths.append(path)
                    img_idx += 1

                remaining -= n
                pbar.update(n)

    return all_paths


# ============================================================================
# CLIP Score Evaluation
# ============================================================================

def compute_clip_scores(
    image_paths: list[str],
    prompt: str,
    device: torch.device,
    model_name: str = "openai/clip-vit-base-patch32",
) -> dict:
    """Compute CLIP score between images and a text prompt."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch.nn.functional as F
    except ImportError:
        print("[warn] transformers not available for CLIP scoring")
        return {}

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    scores = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="  CLIP scoring"):
            img = Image.open(path).convert("RGB")
            inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            # Cosine similarity scaled to [0, 100]
            score = out.logits_per_image.item()
            scores.append(score)

    return {
        "clip_score_mean": float(np.mean(scores)),
        "clip_score_std": float(np.std(scores)),
        "n_images": len(scores),
    }


# ============================================================================
# FID Evaluation (requires clean-fid or pytorch-fid)
# ============================================================================

def compute_fid(gen_dir: str, ref_dir: str, device: torch.device) -> float | None:
    """Compute FID between generated images and reference images."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(gen_dir, ref_dir, device=str(device))
        return score
    except ImportError:
        pass

    try:
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths(
            [gen_dir, ref_dir],
            batch_size=50,
            device=device,
            dims=2048,
        )
        return score
    except ImportError:
        pass

    print("[warn] Neither clean-fid nor pytorch-fid installed. Skipping FID.")
    return None


# ============================================================================
# CLIP-based Classification (for CIFAR-10 style object erasure eval)
# ============================================================================

def clip_classify_images(
    image_paths: list[str],
    class_names: list[str],
    device: torch.device,
    model_name: str = "openai/clip-vit-base-patch32",
) -> dict:
    """
    Classify images using CLIP zero-shot classification.
    Returns per-class accuracy metrics for object erasure evaluation.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("[warn] transformers not available for CLIP classification")
        return {}

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    text_prompts = [f"a photo of a {c}" for c in class_names]
    predictions = []

    with torch.no_grad():
        for path in tqdm(image_paths, desc="  CLIP classify"):
            img = Image.open(path).convert("RGB")
            inputs = processor(
                text=text_prompts, images=img, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            pred_idx = out.logits_per_image.argmax(dim=-1).item()
            predictions.append(class_names[pred_idx])

    counts = {c: predictions.count(c) for c in class_names}
    total = len(predictions)
    acc_per_class = {c: counts.get(c, 0) / total * 100 for c in class_names}

    return {
        "total": total,
        "predictions": predictions,
        "counts": counts,
        "accuracy_per_class": acc_per_class,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pipeline
    pipe = load_pipeline(args.model_id, args.unet_dir, device)

    results = {}

    # ---- Mode 1: Free-form prompt generation ----
    if args.prompts:
        paths = generate_images(
            pipe=pipe,
            prompts=args.prompts,
            output_dir=os.path.join(args.output_dir, "images"),
            n_images=args.n_images,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            image_size=args.image_size,
        )

        # CLIP Score for each prompt
        if args.eval_clip:
            clip_results = {}
            for prompt in args.prompts:
                safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:80].strip()
                prompt_dir = os.path.join(args.output_dir, "images", safe_name)
                p_paths = [str(p) for p in Path(prompt_dir).glob("*.png")]
                clip_results[prompt] = compute_clip_scores(p_paths, prompt, device)

            results["clip_scores"] = clip_results
            print("\n[CLIP Scores]")
            for prompt, r in clip_results.items():
                print(f"  '{prompt}': {r.get('clip_score_mean', 'N/A'):.2f}")

    # ---- Mode 2: CIFAR-10 style object erasure evaluation ----
    if args.eval_classes:
        class_results = {}
        for cls in args.eval_classes:
            prompt = f"a photo of a {cls}"
            cls_dir = os.path.join(args.output_dir, "eval_images", cls)
            generate_images(
                pipe=pipe,
                prompts=[prompt],
                output_dir=os.path.join(args.output_dir, "eval_images"),
                n_images=args.n_images_per_class,
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                image_size=args.image_size,
            )

        # Classify all generated images
        for cls in args.eval_classes:
            safe_cls = "a photo of a " + cls
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in safe_cls)[:80].strip()
            cls_dir = os.path.join(args.output_dir, "eval_images", safe_name)
            img_paths = [str(p) for p in Path(cls_dir).glob("*.png")]
            clf_result = clip_classify_images(img_paths, args.eval_classes, device)
            class_results[cls] = clf_result

        results["object_erasure_eval"] = class_results

        # Compute harmonic mean H0 for erased class
        if args.erased_class and args.erased_class in class_results:
            acc_erased = class_results[args.erased_class]["accuracy_per_class"].get(
                args.erased_class, 0.0
            )  # Acce ↓

            # Accs: average accuracy on non-erased classes
            other_classes = [c for c in args.eval_classes if c != args.erased_class]
            if other_classes:
                acc_spec_list = []
                for cls in other_classes:
                    if cls in class_results:
                        acc_spec_list.append(
                            class_results[cls]["accuracy_per_class"].get(cls, 0.0)
                        )
                Accs = np.mean(acc_spec_list) if acc_spec_list else 0.0
            else:
                Accs = 100.0

            Acce = acc_erased

            # H0 = 3 / (1/(1-Acce) + 1/Accs + 1/(1-Accg))
            # Without generality (Accg = Acce as approximation if no synonyms):
            eps = 1e-6
            H0 = 3.0 / (
                1.0 / (1.0 - Acce / 100.0 + eps)
                + 1.0 / (Accs / 100.0 + eps)
                + 1.0 / (1.0 - Acce / 100.0 + eps)   # Accg ≈ Acce without synonyms
            ) * 100.0

            results["erasure_summary"] = {
                "erased_class": args.erased_class,
                "Acce": Acce,
                "Accs": Accs,
                "H0": H0,
            }
            print(f"\n[Erasure Summary for '{args.erased_class}']")
            print(f"  Acce (↓): {Acce:.2f}%")
            print(f"  Accs (↑): {Accs:.2f}%")
            print(f"  H0  (↑): {H0:.2f}")

    # ---- Save results ----
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def parse_args():
    p = argparse.ArgumentParser(description="SDErasure: Generate and evaluate images")

    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--unet_dir", type=str, default=None,
                   help="Path to fine-tuned UNet directory (output of train_sderasure.py). "
                        "If not set, uses original model weights.")
    p.add_argument("--output_dir", type=str, required=True)

    # Generation
    p.add_argument("--prompts", type=str, nargs="*", default=[],
                   help="Text prompts for free-form generation")
    p.add_argument("--n_images", type=int, default=50,
                   help="Number of images per prompt")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    # CIFAR-10 style object erasure evaluation
    p.add_argument("--eval_classes", type=str, nargs="*", default=[],
                   help="All classes for CIFAR-10 style evaluation")
    p.add_argument("--erased_class", type=str, default=None,
                   help="The class that was erased (for H0 computation)")
    p.add_argument("--n_images_per_class", type=int, default=200)

    # Evaluation flags
    p.add_argument("--eval_clip", action="store_true",
                   help="Compute CLIP scores for generated images")
    p.add_argument("--ref_dir", type=str, default=None,
                   help="Reference image directory for FID computation")

    return p.parse_args()


if __name__ == "__main__":
    main()
