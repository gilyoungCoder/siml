from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_NUM_IMAGES_PER_PROMPT,
    DEFAULT_SEED,
    MODEL_FAMILIES,
    MODEL_SPECS,
    REPO_ROOT,
    SLD_SAFETY_CONCEPT,
    SLD_VARIANTS,
)


MANIFEST_FIELDS = (
    "prompt_id",
    "dataset",
    "concept",
    "method",
    "variant",
    "image_path",
    "prompt",
    "seed",
    "steps",
    "guidance_scale",
    "height",
    "width",
    "sample_index",
    "runtime_seconds",
    "batched_unet_calls",
    "condition_forward_equivalents",
    "status",
    "error",
)


def _torch_dtype(dtype_name: str, device: str):
    import torch

    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_name}")


def _detect_model_family(model_id: str) -> str:
    normalized = model_id.lower()
    if "flux" in normalized:
        return "flux"
    if "stable-diffusion-3" in normalized or "sd3" in normalized:
        return "sd3"
    return "sd14"


def _apply_model_defaults(args: argparse.Namespace) -> None:
    model_family = args.model_family
    if model_family == "auto":
        if args.model_id:
            model_family = _detect_model_family(args.model_id)
        else:
            model_family = "sd14"

    spec = MODEL_SPECS[model_family]
    args.model_family = model_family
    if args.model_id is None:
        args.model_id = spec["model_id"]
    if args.steps is None:
        args.steps = int(spec["steps"])
    if args.guidance_scale is None:
        args.guidance_scale = float(spec["guidance_scale"])
    if args.height is None:
        args.height = int(spec["height"])
    if args.width is None:
        args.width = int(spec["width"])
    if args.max_sequence_length is None:
        args.max_sequence_length = spec["max_sequence_length"]


def _sync_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


def _place_pipeline(pipe: Any, args: argparse.Namespace):
    if args.enable_model_cpu_offload and args.enable_sequential_cpu_offload:
        raise ValueError("Use only one of --enable-model-cpu-offload or --enable-sequential-cpu-offload")
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(device=args.device)
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=not args.show_progress)
    return pipe


def _row_value(row: pd.Series, key: str, default: Any) -> Any:
    if key not in row or pd.isna(row[key]):
        return default
    return row[key]


def _prompt_token(prompt_id: Any) -> str:
    try:
        return f"{int(prompt_id):06d}"
    except (TypeError, ValueError):
        token = str(prompt_id).strip().replace("/", "_").replace(" ", "_")
        return token or "prompt"


def _load_prompts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"prompt_id", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if df["prompt_id"].duplicated().any():
        duplicates = df.loc[df["prompt_id"].duplicated(), "prompt_id"].head(5).tolist()
        raise ValueError(f"{path} has duplicate prompt_id values, e.g. {duplicates}")
    return df


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest_has_image_rows(manifest_path: Path, image_paths: list[Path]) -> bool:
    if not manifest_path.exists():
        return False
    expected = {str(path) for path in image_paths}
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            recorded = {
                row.get("image_path", "")
                for row in reader
                if row.get("status") in {"generated", "skipped_existing"}
            }
    except csv.Error:
        return False
    return expected.issubset(recorded)


def _nfe_counts(
    method: str,
    model_family: str,
    steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
) -> tuple[int, int]:
    if method == "sld":
        condition_count = 3 if guidance_scale > 1 else 1
    elif model_family == "flux":
        # FLUX guidance_scale is embedded guidance, not classifier-free guidance.
        condition_count = 2 if true_cfg_scale > 1 else 1
    else:
        condition_count = 2 if guidance_scale > 1 else 1
    return steps, steps * condition_count


def _load_safetensors_into_module(module: Any, weights_path: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    weights = load_file(weights_path)
    result = module.load_state_dict(weights, strict=False)
    return {
        "num_weight_keys": len(weights),
        "missing_keys": len(getattr(result, "missing_keys", [])),
        "unexpected_keys": len(getattr(result, "unexpected_keys", [])),
    }


def _load_sd14_pipeline(args: argparse.Namespace):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

    dtype = _torch_dtype(args.dtype, args.device)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if args.method == "uce":
        if not args.uce_weights:
            raise ValueError("--uce-weights is required for --method uce")
        _load_safetensors_into_module(pipe.unet, args.uce_weights)

    if args.method == "rece":
        if not args.rece_ckpt:
            raise ValueError("--rece-ckpt is required for --method rece")
        ckpt_path = Path(args.rece_ckpt)
        if ckpt_path.suffix == ".pt":
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            pipe.unet.load_state_dict(state_dict, strict=False)
        else:
            pipe.unet = UNet2DConditionModel.from_pretrained(str(ckpt_path))

    return _place_pipeline(pipe, args)


def _load_sd3_pipeline(args: argparse.Namespace):
    from diffusers import StableDiffusion3Pipeline

    dtype = _torch_dtype(args.dtype, args.device)
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.method == "uce":
        if not args.uce_weights:
            raise ValueError("--uce-weights is required for --method uce")
        _load_safetensors_into_module(pipe.transformer, args.uce_weights)
    if args.method == "rece":
        if not args.rece_ckpt:
            raise ValueError("--rece-ckpt is required for --method rece")
        _load_safetensors_into_module(pipe.transformer, args.rece_ckpt)

    return _place_pipeline(pipe, args)


def _load_flux_pipeline(args: argparse.Namespace):
    from diffusers import FluxPipeline

    dtype = _torch_dtype(args.dtype, args.device)
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.method == "uce":
        if not args.uce_weights:
            raise ValueError("--uce-weights is required for --method uce")
        _load_safetensors_into_module(pipe.transformer, args.uce_weights)
    if args.method == "rece":
        if not args.rece_ckpt:
            raise ValueError("--rece-ckpt is required for --method rece")
        _load_safetensors_into_module(pipe.transformer, args.rece_ckpt)

    return _place_pipeline(pipe, args)


def _load_sld_pipeline(args: argparse.Namespace):
    from diffusers import DDIMScheduler

    sld_src = Path(args.sld_repo).resolve() / "src"
    if str(sld_src) not in sys.path:
        sys.path.insert(0, str(sld_src))
    from sld import SLDPipeline

    dtype = _torch_dtype(args.dtype, args.device)
    pipe = SLDPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_concept = args.sld_safety_concept
    return _place_pipeline(pipe, args)


def _load_pipeline(args: argparse.Namespace):
    if args.method == "sld":
        return _load_sld_pipeline(args)
    if args.model_family == "sd14":
        return _load_sd14_pipeline(args)
    if args.model_family == "sd3":
        return _load_sd3_pipeline(args)
    if args.model_family == "flux":
        return _load_flux_pipeline(args)
    raise ValueError(f"Unknown model family: {args.model_family}")


def _generate_with_pipe(
    pipe: Any,
    args: argparse.Namespace,
    df: pd.DataFrame,
    images_dir: Path,
    manifest_path: Path,
) -> None:
    import torch

    variant_params = {}
    variant_name = ""
    if args.method == "sld":
        variant_name = args.sld_variant
        variant_params = SLD_VARIANTS[args.sld_variant]
    elif args.model_family == "flux":
        variant_params["true_cfg_scale"] = args.true_cfg_scale

    if args.max_sequence_length is not None and args.model_family in {"sd3", "flux"}:
        variant_params["max_sequence_length"] = args.max_sequence_length

    for _, row in df.iterrows():
        prompt_id = _row_value(row, "prompt_id", row.name)
        prompt = str(row["prompt"])
        seed = int(_row_value(row, "evaluation_seed", args.seed))
        steps = int(_row_value(row, "steps", args.steps))
        guidance_scale = float(_row_value(row, "guidance_scale", args.guidance_scale))
        height = int(_row_value(row, "height", args.height))
        width = int(_row_value(row, "width", args.width))
        dataset = str(_row_value(row, "dataset", ""))
        concept = str(_row_value(row, "concept", ""))
        token = _prompt_token(prompt_id)

        expected_paths = [
            images_dir / f"{token}_{sample_idx}.png"
            for sample_idx in range(args.num_images_per_prompt)
        ]
        if not args.overwrite and all(path.exists() for path in expected_paths):
            if _manifest_has_image_rows(manifest_path, expected_paths):
                continue
            batched_calls, condition_equiv = _nfe_counts(
                args.method,
                args.model_family,
                steps,
                guidance_scale,
                args.true_cfg_scale,
            )
            _append_manifest(
                manifest_path,
                [
                    {
                        "prompt_id": prompt_id,
                        "dataset": dataset,
                        "concept": concept,
                        "method": args.method,
                        "variant": variant_name,
                        "image_path": str(path),
                        "prompt": prompt,
                        "seed": seed,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width,
                        "sample_index": sample_idx,
                        "runtime_seconds": "",
                        "batched_unet_calls": batched_calls,
                        "condition_forward_equivalents": condition_equiv,
                        "status": "skipped_existing",
                        "error": "",
                    }
                    for sample_idx, path in enumerate(expected_paths)
                ],
            )
            continue

        generator_device = args.device if args.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        try:
            _sync_cuda(args.device)
            started = time.perf_counter()
            output = pipe(
                prompt=prompt,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                height=height,
                width=width,
                num_images_per_prompt=args.num_images_per_prompt,
                **variant_params,
            )
            _sync_cuda(args.device)
            runtime = time.perf_counter() - started
            batched_calls, condition_equiv = _nfe_counts(
                args.method,
                args.model_family,
                steps,
                guidance_scale,
                args.true_cfg_scale,
            )
            if not output.images:
                raise RuntimeError("Pipeline returned no images")

            manifest_rows = []
            for sample_idx, image in enumerate(output.images):
                image_path = images_dir / f"{token}_{sample_idx}.png"
                image.save(image_path)
                manifest_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "dataset": dataset,
                        "concept": concept,
                        "method": args.method,
                        "variant": variant_name,
                        "image_path": str(image_path),
                        "prompt": prompt,
                        "seed": seed,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width,
                        "sample_index": sample_idx,
                        "runtime_seconds": f"{runtime / len(output.images):.6f}",
                        "batched_unet_calls": batched_calls,
                        "condition_forward_equivalents": condition_equiv,
                        "status": "generated",
                        "error": "",
                    }
                )
            _append_manifest(manifest_path, manifest_rows)
        except Exception as exc:  # Keep long jobs resumable.
            batched_calls, condition_equiv = _nfe_counts(
                args.method,
                args.model_family,
                steps,
                guidance_scale,
                args.true_cfg_scale,
            )
            _append_manifest(
                manifest_path,
                [
                    {
                        "prompt_id": prompt_id,
                        "dataset": dataset,
                        "concept": concept,
                        "method": args.method,
                        "variant": variant_name,
                        "image_path": "",
                        "prompt": prompt,
                        "seed": seed,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width,
                        "sample_index": "",
                        "runtime_seconds": "",
                        "batched_unet_calls": batched_calls,
                        "condition_forward_equivalents": condition_equiv,
                        "status": "error",
                        "error": repr(exc),
                    }
                ],
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SD v1.4 / SD3 / FLUX generation from standardized prompts."
    )
    parser.add_argument("--method", choices=("sd", "uce", "rece", "sld"), required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="Standardized prompt CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=("auto",) + MODEL_FAMILIES, default="auto")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-images-per-prompt", type=int, default=DEFAULT_NUM_IMAGES_PER_PROMPT)
    parser.add_argument("--uce-weights", type=str, default=None, help="UCE safetensors path")
    parser.add_argument(
        "--rece-ckpt",
        type=str,
        default=None,
        help=(
            "RECE checkpoint. SD v1.4 accepts UNet .pt or diffusers UNet dir; "
            "SD3/FLUX accept transformer .safetensors only."
        ),
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=1.0,
        help="FLUX true CFG scale. Keep at 1.0 for standard FLUX guidance_scale runs.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="SD3/FLUX prompt encoder max sequence length; defaults by model family.",
    )
    parser.add_argument("--sld-variant", choices=tuple(SLD_VARIANTS), default="SLD-Medium")
    parser.add_argument("--sld-repo", type=Path, default=REPO_ROOT / "safe-latent-diffusion")
    parser.add_argument("--sld-safety-concept", type=str, default=SLD_SAFETY_CONCEPT)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing images")
    parser.add_argument("--show-progress", action="store_true", help="Enable diffusers progress bars")
    parser.add_argument(
        "--enable-model-cpu-offload",
        action="store_true",
        help="Use diffusers model CPU offload instead of moving the full pipeline to GPU.",
    )
    parser.add_argument(
        "--enable-sequential-cpu-offload",
        action="store_true",
        help="Use diffusers sequential CPU offload for lower VRAM use and slower inference.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading models")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.num_images_per_prompt <= 0:
        raise ValueError("--num-images-per-prompt must be positive")
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("--height and --width must be divisible by 8")
    if args.method == "uce" and not args.uce_weights:
        raise ValueError("--uce-weights is required for --method uce")
    if args.method == "rece" and not args.rece_ckpt:
        raise ValueError("--rece-ckpt is required for --method rece")
    if args.method == "rece" and args.model_family in {"sd3", "flux"}:
        rece_path = Path(args.rece_ckpt)
        if rece_path.suffix != ".safetensors":
            raise ValueError(
                "SD3/FLUX RECE runner expects transformer .safetensors weights. "
                "SD v1.4 UNet .pt RECE checkpoints cannot be applied to SD3/FLUX."
            )
    if args.method == "sld" and args.model_family != "sd14":
        raise ValueError("SLD is implemented only for the SD v1.4 StableDiffusionPipeline.")
    if args.true_cfg_scale <= 0:
        raise ValueError("--true-cfg-scale must be positive")
    if args.max_sequence_length is not None and args.max_sequence_length <= 0:
        raise ValueError("--max-sequence-length must be positive")
    if args.enable_model_cpu_offload and args.enable_sequential_cpu_offload:
        raise ValueError("Use only one of --enable-model-cpu-offload or --enable-sequential-cpu-offload")
    if (args.enable_model_cpu_offload or args.enable_sequential_cpu_offload) and not args.device.startswith("cuda"):
        raise ValueError("CPU offload requires a CUDA device target")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _apply_model_defaults(args)
    _validate_args(args)

    df = _load_prompts(args.prompts)
    images_dir = args.output_dir / "images"
    manifest_path = args.output_dir / "manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": args.method,
        "variant": args.sld_variant if args.method == "sld" else "",
        "prompts": str(args.prompts),
        "output_dir": str(args.output_dir),
        "model_family": args.model_family,
        "model_id": args.model_id,
        "scheduler": MODEL_SPECS[args.model_family]["scheduler"],
        "default_seed": args.seed,
        "default_steps": args.steps,
        "default_guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale if args.model_family == "flux" else "",
        "default_height": args.height,
        "default_width": args.width,
        "max_sequence_length": args.max_sequence_length or "",
        "num_images_per_prompt": args.num_images_per_prompt,
        "uce_weights": args.uce_weights or "",
        "rece_ckpt": args.rece_ckpt or "",
        "sld_repo": str(args.sld_repo) if args.method == "sld" else "",
        "sld_safety_concept": args.sld_safety_concept if args.method == "sld" else "",
        "sld_variant_params": SLD_VARIANTS[args.sld_variant] if args.method == "sld" else {},
        "enable_model_cpu_offload": args.enable_model_cpu_offload,
        "enable_sequential_cpu_offload": args.enable_sequential_cpu_offload,
    }
    _write_json(args.output_dir / "run_config.json", config)

    print(f"Loaded {len(df)} prompts from {args.prompts}")
    print(f"Output: {args.output_dir}")
    if args.dry_run:
        print("Dry run only; model was not loaded.")
        return

    pipe = _load_pipeline(args)

    _generate_with_pipe(pipe, args, df, images_dir, manifest_path)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
