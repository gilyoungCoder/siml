#!/usr/bin/env python3
"""Run Qwen3-VL v5 evaluation for one EBSG config/output directory.

This wrapper is intentionally thin: it resolves a JSON config's output directory,
normalizes the paper concept name to the evaluator's rubric name, and invokes the
external Qwen3-VL v5 evaluator script used for the paper.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CONCEPT_ALIASES = {
    "sexual": "nudity",
    "nudity": "nudity",
    "violence": "violence",
    "self-harm": "self_harm",
    "self_harm": "self_harm",
    "shocking": "shocking",
    "illegal_activity": "illegal",
    "illegal": "illegal",
    "harassment": "harassment",
    "hate": "hate",
    "disturbing": "disturbing",
}


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(x) for x in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def infer_concept(config_path: Path, cfg: dict) -> str:
    for key in ("eval_concept", "concept"):
        if cfg.get(key):
            return str(cfg[key])
    name = config_path.stem
    parent = config_path.parent.name
    # Nudity benchmark configs live under configs/ours_best/nudity/*.json.
    if parent == "nudity":
        return "nudity"
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Qwen3-VL v5 evaluation for one config.")
    ap.add_argument("--config", required=True, help="Path to config JSON containing outdir.")
    ap.add_argument("--concept", default=None, help="Override evaluator concept/rubric name.")
    ap.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    ap.add_argument("--force", action="store_true", help="Re-run even if result file already exists.")
    ap.add_argument("--model", default="qwen", help="Evaluator model argument; default qwen.")
    args = ap.parse_args()

    repro = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parents[1])).resolve()
    os.environ.setdefault("REPRO_ROOT", str(repro))
    os.environ.setdefault("OUT_ROOT", str(repro))

    py = os.environ.get("PY_VLM")
    vlm = os.environ.get("VLM_SCRIPT")
    if not py or not vlm:
        raise SystemExit("Set PY_VLM=/path/to/vlm/python and VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py")

    config_path = Path(args.config)
    cfg = expand(json.loads(config_path.read_text()))
    outdir = Path(cfg["outdir"])
    concept_raw = args.concept or infer_concept(config_path, cfg)
    concept = CONCEPT_ALIASES.get(concept_raw, concept_raw)

    if not outdir.is_dir():
        raise SystemExit(f"Output directory does not exist: {outdir}")
    pngs = sorted(outdir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNG files found in: {outdir}")

    result = outdir / f"results_qwen3_vl_{concept}_v5.txt"
    if result.exists() and not args.force:
        print(f"[SKIP] {result} already exists; use --force to re-run")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cmd = [py, vlm, str(outdir), concept, args.model]
    print("[EVAL]", " ".join(cmd), flush=True)
    print(f"[INFO] concept={concept} png_count={len(pngs)} outdir={outdir}", flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
