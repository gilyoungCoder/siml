#!/usr/bin/env python3
"""Qwen3-VL v5 evaluation wrapper used by the EBSG paper bundle.

This file intentionally delegates to the canonical v5 evaluator script instead
of carrying a second copy of the rubric prompts.  The canonical script is passed
via VLM_SCRIPT and is the same script used for the verified paper numbers:

    opensource_vlm_i2p_all_v5.py <img_dir> <concept> [model]

Required environment:
    PY_VLM=/path/to/python-with-qwen-vl-deps
    VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py

Examples:
    PY_VLM=/envs/vlm/bin/python \
    VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py \
    python -m evaluation.eval_vlm outputs/i2p/violence violence qwen

    python code/SafeGen/evaluation/eval_vlm.py outputs/i2p/self-harm self-harm qwen

Concept aliases are normalized to the v5 evaluator names:
    sexual -> nudity
    self-harm -> self_harm
    illegal_activity -> illegal
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CONCEPT_ALIASES = {
    "sexual": "nudity",
    "nudity": "nudity",
    "violence": "violence",
    "harassment": "harassment",
    "hate": "hate",
    "shocking": "shocking",
    "disturbing": "disturbing",
    "illegal_activity": "illegal",
    "illegal-activity": "illegal",
    "illegal": "illegal",
    "self-harm": "self_harm",
    "self_harm": "self_harm",
    "selfharm": "self_harm",
    "all": "all",
}


def normalize_concept(concept: str) -> str:
    key = concept.strip().lower()
    if key not in CONCEPT_ALIASES:
        known = ", ".join(sorted(CONCEPT_ALIASES))
        raise SystemExit(f"Unknown concept '{concept}'. Known aliases: {known}")
    return CONCEPT_ALIASES[key]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical Qwen3-VL v5 safety evaluation.")
    parser.add_argument("img_dir", help="Directory containing generated .png/.jpg images")
    parser.add_argument("concept", help="Concept/rubric name, e.g. nudity, violence, self-harm")
    parser.add_argument("model", nargs="?", default="qwen", help="Evaluator model argument; default: qwen")
    parser.add_argument("--gpu", default=os.environ.get("GPU"), help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    py = os.environ.get("PY_VLM", sys.executable)
    vlm_script = os.environ.get("VLM_SCRIPT")
    if not vlm_script:
        raise SystemExit(
            "VLM_SCRIPT is required. Set it to the canonical "
            "opensource_vlm_i2p_all_v5.py used for the paper."
        )

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")

    concept = normalize_concept(args.concept)
    cmd = [py, vlm_script, str(img_dir), concept, args.model]
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("[QWEN-VL-V5]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
