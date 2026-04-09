#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare ALL concept packs at once for multi-concept erasing.

For each concept pack directory, this script:
  1. Reads target_prompts.txt and anchor_prompts.txt
  2. Calls prepare_concept_subspace.py logic -> concept_directions.pt
  3. Calls prepare_clip_patch_tokens.py logic -> clip_patch_tokens.pt
  4. Calls prepare_contrastive_direction.py logic -> contrastive_embeddings.pt

Generated .pt files are saved INTO the pack directory alongside the text files.

Usage:
    # Process specific packs:
    CUDA_VISIBLE_DEVICES=0 python prepare_multi_concept.py \
        --pack_dirs concept_packs/violence concept_packs/sexual

    # Process all packs:
    CUDA_VISIBLE_DEVICES=0 python prepare_multi_concept.py --all

    # Only generate concept_directions (skip CLIP-based preps):
    CUDA_VISIBLE_DEVICES=0 python prepare_multi_concept.py --all --skip_clip --skip_contrastive
"""

import sys
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from concept_pack_loader import load_concept_pack


# Default base path for concept packs
DEFAULT_PACK_BASE = str(
    Path(__file__).resolve().parent.parent
    / "docs" / "neurips_plan" / "multi_concept" / "concept_packs"
)


def discover_all_packs(base_dir: str) -> list:
    """Find all concept pack directories under base_dir."""
    base = Path(base_dir)
    if not base.is_dir():
        print(f"ERROR: Pack base directory not found: {base}")
        sys.exit(1)
    packs = sorted([
        str(d) for d in base.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])
    return packs


def run_concept_subspace(pack_dir: str, pack, args):
    """
    Generate concept_directions.pt for a concept pack.

    Calls prepare_concept_subspace.py as a subprocess, passing the pack's
    target and anchor prompts as concept keywords.
    """
    target_prompts = pack.target_prompts
    anchor_prompts = pack.anchor_prompts
    if not target_prompts or not anchor_prompts:
        print(f"  SKIP concept_subspace: no target/anchor prompts in {pack_dir}")
        return False

    output_path = str(Path(pack_dir) / "concept_directions.pt")
    script = str(Path(__file__).resolve().parent / "prepare_concept_subspace.py")

    target_concepts = pack.target_keywords_primary[:5] if pack.target_keywords_primary else [pack.name]
    anchor_concepts = pack.anchor_keywords[:5] if pack.anchor_keywords else [f"safe {pack.name}"]
    exemplar_dir = Path(pack_dir) / "exemplar_images"

    cmd = [
        sys.executable, script,
        "--output", output_path,
        "--steps", str(args.steps),
        "--cfg_scale", str(args.cfg_scale),
        "--seed", str(args.seed),
        "--target_concepts", *target_concepts,
        "--anchor_concepts", *anchor_concepts,
        "--target_prompt_file", str(Path(pack_dir) / "target_prompts.txt"),
        "--anchor_prompt_file", str(Path(pack_dir) / "anchor_prompts.txt"),
        "--target_image_prefix", args.target_prefix,
        "--anchor_image_prefix", args.anchor_prefix,
    ]
    if args.use_glass:
        cmd.append("--use_glass")
    if args.save_images or args.ensure_exemplar_images:
        cmd.append("--save_images")

    print(f"  Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=not args.verbose, text=True)
    if result.returncode != 0:
        print(f"  FAILED concept_subspace for {pack.name}")
        if not args.verbose and result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        return False
    print(f"  OK concept_directions.pt -> {output_path}")
    return True


def run_clip_patch_tokens(pack_dir: str, pack, args):
    """
    Generate clip_patch_tokens.pt for a concept pack.

    Requires exemplar images to already exist in the pack's exemplar_images/
    subdirectory. If no images exist, skips gracefully.
    """
    exemplar_dir = str(Path(pack_dir) / "exemplar_images")
    if not Path(exemplar_dir).is_dir():
        print(f"  SKIP clip_patch_tokens: no exemplar_images/ in {pack_dir}")
        return False

    output_path = str(Path(pack_dir) / "clip_patch_tokens.pt")
    script = str(Path(__file__).resolve().parent / "prepare_clip_patch_tokens.py")

    cmd = [
        sys.executable, script,
        "--exemplar_dir", exemplar_dir,
        "--output", output_path,
        "--num_patches", str(args.num_patches),
        "--target_prefix", args.target_prefix,
        "--anchor_prefix", args.anchor_prefix,
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=not args.verbose, text=True)
    if result.returncode != 0:
        print(f"  FAILED clip_patch_tokens for {pack.name}")
        if not args.verbose and result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        return False
    print(f"  OK clip_patch_tokens.pt -> {output_path}")
    return True


def run_contrastive_direction(pack_dir: str, pack, args):
    """
    Generate contrastive_embeddings.pt for a concept pack.

    Requires exemplar images to already exist in the pack's exemplar_images/
    subdirectory. If no images exist, skips gracefully.
    """
    exemplar_dir = str(Path(pack_dir) / "exemplar_images")
    if not Path(exemplar_dir).is_dir():
        print(f"  SKIP contrastive_embeddings: no exemplar_images/ in {pack_dir}")
        return False

    output_path = str(Path(pack_dir) / "contrastive_embeddings.pt")
    script = str(Path(__file__).resolve().parent / "prepare_contrastive_direction.py")

    cmd = [
        sys.executable, script,
        "--exemplar_dir", exemplar_dir,
        "--output", output_path,
        "--n_tokens", str(args.n_tokens),
        "--top_k", str(args.top_k),
        "--target_prefix", args.target_prefix,
        "--anchor_prefix", args.anchor_prefix,
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=not args.verbose, text=True)
    if result.returncode != 0:
        print(f"  FAILED contrastive_embeddings for {pack.name}")
        if not args.verbose and result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        return False
    print(f"  OK contrastive_embeddings.pt -> {output_path}")
    return True


def parse_args():
    p = ArgumentParser(
        description="Prepare all concept packs for multi-concept erasing"
    )
    # Pack selection
    p.add_argument("--pack_dirs", type=str, nargs="+", default=None,
                   help="Specific concept pack directories to process")
    p.add_argument("--all", action="store_true",
                   help="Process all concept_packs/*/ under --pack_base")
    p.add_argument("--pack_base", type=str, default=DEFAULT_PACK_BASE,
                   help="Base directory containing concept pack folders")

    # Skip flags
    p.add_argument("--skip_subspace", action="store_true",
                   help="Skip concept_directions.pt generation")
    p.add_argument("--skip_clip", action="store_true",
                   help="Skip clip_patch_tokens.pt generation")
    p.add_argument("--skip_contrastive", action="store_true",
                   help="Skip contrastive_embeddings.pt generation")

    # Params passed to prepare_concept_subspace.py
    p.add_argument("--steps", type=int, default=50,
                   help="DDIM steps for exemplar generation")
    p.add_argument("--cfg_scale", type=float, default=7.5,
                   help="CFG scale for exemplar generation")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--use_glass", action="store_true",
                   help="Use GLASS instead of DDIM for subspace prep")
    p.add_argument("--save_images", action="store_true",
                   help="Save exemplar images during subspace prep")
    p.add_argument("--ensure_exemplar_images", action="store_true",
                   help="Force concept_subspace to generate exemplar_images/ so downstream prep can run")
    p.add_argument("--target_prefix", type=str, default="target_",
                   help="Filename prefix for generated target exemplar images")
    p.add_argument("--anchor_prefix", type=str, default="anchor_",
                   help="Filename prefix for generated anchor exemplar images")

    # Params passed to prepare_clip_patch_tokens.py
    p.add_argument("--num_patches", type=int, default=16,
                   help="Number of top-K discriminative patches")

    # Params passed to prepare_contrastive_direction.py
    p.add_argument("--n_tokens", type=int, default=4,
                   help="Number of concept tokens in probe embeddings")
    p.add_argument("--top_k", type=int, default=8,
                   help="Number of most discriminative patches for contrastive")

    # General
    p.add_argument("--verbose", action="store_true",
                   help="Show full subprocess output")

    return p.parse_args()


def main():
    args = parse_args()

    # --- Resolve pack directories ---
    if args.all:
        pack_dirs = discover_all_packs(args.pack_base)
    elif args.pack_dirs:
        pack_dirs = args.pack_dirs
    else:
        print("ERROR: Specify --pack_dirs or --all")
        sys.exit(1)

    if not pack_dirs:
        print("No concept packs found.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Multi-Concept Pack Preparation")
    print(f"{'='*70}")
    print(f"  Packs:           {len(pack_dirs)}")
    print(f"  Skip subspace:   {args.skip_subspace}")
    print(f"  Skip CLIP:       {args.skip_clip}")
    print(f"  Skip contrastive:{args.skip_contrastive}")
    print(f"  Ensure images:   {args.ensure_exemplar_images}")
    print(f"  Image prefixes:  target='{args.target_prefix}' anchor='{args.anchor_prefix}'")
    print(f"{'='*70}\n")

    results = {}

    for pack_dir in pack_dirs:
        pack_dir = str(Path(pack_dir).resolve())
        pack = load_concept_pack(pack_dir, device="cpu")
        concept_name = pack.name

        print(f"\n--- [{concept_name}] {pack_dir} ---")
        print(f"  Families: {len(pack.families)}, "
              f"CAS threshold: {pack.cas_threshold}, "
              f"Probe: {pack.probe_source}, "
              f"Mode: {pack.guide_mode}")

        status = {"subspace": None, "clip": None, "contrastive": None}

        # 1. Concept directions
        if not args.skip_subspace:
            status["subspace"] = run_concept_subspace(pack_dir, pack, args)
        else:
            print("  SKIP concept_subspace (--skip_subspace)")

        # 2. CLIP patch tokens
        if not args.skip_clip:
            status["clip"] = run_clip_patch_tokens(pack_dir, pack, args)
        else:
            print("  SKIP clip_patch_tokens (--skip_clip)")

        # 3. Contrastive embeddings
        if not args.skip_contrastive:
            status["contrastive"] = run_contrastive_direction(pack_dir, pack, args)
        else:
            print("  SKIP contrastive_embeddings (--skip_contrastive)")

        results[concept_name] = status

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    for concept, status in results.items():
        parts = []
        for key, val in status.items():
            if val is None:
                parts.append(f"{key}:skip")
            elif val:
                parts.append(f"{key}:OK")
            else:
                parts.append(f"{key}:FAIL")
        print(f"  {concept:20s}  {' | '.join(parts)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
