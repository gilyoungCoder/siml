from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    CONCEPT_SETS,
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    REPO_ROOT,
)


def _parse_concepts(raw: str | None, concept_set: str | None) -> list[str]:
    if concept_set:
        return list(CONCEPT_SETS[concept_set])
    if not raw:
        raise ValueError("Pass --concepts or --concept-set")
    parts = raw.replace(",", ";").split(";")
    concepts = [part.strip() for part in parts if part.strip()]
    if not concepts:
        raise ValueError("No concepts parsed from --concepts")
    return concepts


def _concept_mode(concepts: list[str]) -> str:
    return "single" if len(concepts) == 1 else "multi"


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_plan(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "erasure_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    commands = payload.get("commands", [])
    (output_dir / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(_shell_join(command) for command in commands)
        + "\n",
        encoding="utf-8",
    )


def _run_commands(commands: list[list[str]], cwd: Path) -> None:
    for command in commands:
        print(f"[run] {cwd}: {_shell_join(command)}")
        subprocess.run(command, cwd=cwd, check=True)


def _uce_command(args: argparse.Namespace, concepts: list[str]) -> tuple[list[str], Path]:
    uce_repo = args.uce_repo.resolve()
    save_dir = args.output_dir.resolve()
    exp_name = args.exp_name or f"{_concept_mode(concepts)}_{args.tag}"
    output_path = save_dir / f"{exp_name}.safetensors"

    guide_concepts = args.guide_concepts
    if guide_concepts is None:
        guide_concepts = "art" if args.uce_concept_type == "art" else ""

    command = [
        "python",
        "trainscripts/uce_sd_erase.py",
        "--model_id",
        args.model_id,
        "--edit_concepts",
        "; ".join(concepts),
        "--guide_concepts",
        guide_concepts,
        "--concept_type",
        args.uce_concept_type,
        "--device",
        args.device,
        "--erase_scale",
        str(args.erase_scale),
        "--preserve_scale",
        str(args.preserve_scale),
        "--lamb",
        str(args.lamb),
        "--expand_prompts",
        args.expand_prompts,
        "--save_dir",
        str(save_dir),
        "--exp_name",
        exp_name,
    ]
    if args.preserve_concepts:
        command.extend(["--preserve_concepts", args.preserve_concepts])
    return command, output_path


def _rece_commands(args: argparse.Namespace, concepts: list[str]) -> tuple[list[list[str]], Path | None]:
    rece_repo = args.rece_repo.resolve()
    commands: list[list[str]] = []

    target_ckpt = args.target_ckpt
    if args.uce_weights:
        converted = args.output_dir.resolve() / "uce_full_unet.pt"
        commands.append(
            [
                "python",
                "scripts/convert_uce_safetensors.py",
                "--safetensors",
                str(Path(args.uce_weights).resolve()),
                "--model_id",
                args.model_id,
                "--output",
                str(converted),
            ]
        )
        target_ckpt = str(converted)

    if not target_ckpt:
        raise ValueError("RECE requires --target-ckpt or --uce-weights")

    if args.rece_concept_type == "auto":
        rece_concept_type = "nudity" if concepts == ["nudity"] else "unsafe"
    else:
        rece_concept_type = args.rece_concept_type

    notes = []
    if rece_concept_type == "unsafe":
        notes.append(
            "Current RECE/train.py has Q16 import commented out; unsafe RECE training may need that restored."
        )

    train_command = [
        "python",
        "train.py",
        "--concepts",
        ",".join(concepts),
        "--concept_type",
        rece_concept_type,
        "--emb_computing",
        args.emb_computing,
        "--regular_scale",
        str(args.regular_scale),
        "--epochs",
        str(args.epochs),
        "--target_ckpt",
        target_ckpt,
        "--preserve_scale",
        str(args.rece_preserve_scale),
        "--erase_scale",
        str(args.rece_erase_scale),
        "--lamb",
        str(args.rece_lamb),
        "--save_path",
        str(args.output_dir.resolve()),
        "--seed",
        str(args.seed),
        "--ddim_steps",
        str(args.steps),
    ]
    if args.test_csv_path:
        train_command.extend(["--test_csv_path", str(Path(args.test_csv_path).resolve())])
    if args.guide_concepts:
        train_command.extend(["--guided_concepts", args.guide_concepts])
    if args.preserve_concepts:
        train_command.extend(["--preserve_concepts", args.preserve_concepts])

    commands.append(train_command)
    args._rece_notes = notes
    return commands, rece_repo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create reproducible UCE/RECE erasure plans for single or multi-concept baselines."
    )
    parser.add_argument("--method", choices=("uce", "rece"), required=True)
    parser.add_argument("--concepts", type=str, default=None, help="Concepts separated by ';' or ','")
    parser.add_argument("--concept-set", choices=tuple(CONCEPT_SETS), default=None)
    parser.add_argument("--tag", type=str, default="erasure")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--run", action="store_true", help="Actually run the generated commands")

    parser.add_argument("--uce-repo", type=Path, default=REPO_ROOT / "unified-concept-editing")
    parser.add_argument("--uce-concept-type", choices=("object", "art", "unsafe"), default="unsafe")
    parser.add_argument("--guide-concepts", type=str, default=None)
    parser.add_argument("--preserve-concepts", type=str, default=None)
    parser.add_argument("--expand-prompts", choices=("true", "false"), default="false")
    parser.add_argument("--erase-scale", type=float, default=1.0)
    parser.add_argument("--preserve-scale", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--exp-name", type=str, default=None)

    parser.add_argument("--rece-repo", type=Path, default=REPO_ROOT / "RECE")
    parser.add_argument("--uce-weights", type=str, default=None)
    parser.add_argument("--target-ckpt", type=str, default=None)
    parser.add_argument("--rece-concept-type", choices=("auto", "nudity", "unsafe"), default="auto")
    parser.add_argument("--emb-computing", choices=("close_standardreg", "close_surrogatereg", "close_regzero"), default="close_regzero")
    parser.add_argument("--regular-scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rece-preserve-scale", type=float, default=0.1)
    parser.add_argument("--rece-erase-scale", type=float, default=1.0)
    parser.add_argument("--rece-lamb", type=float, default=0.1)
    parser.add_argument("--test-csv-path", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    concepts = _parse_concepts(args.concepts, args.concept_set)
    mode = _concept_mode(concepts)

    if args.method == "uce":
        command, output_path = _uce_command(args, concepts)
        commands = [command]
        cwd = args.uce_repo.resolve()
        notes = [
            "UCE trains single and multi-concept erasures in one closed-form edit.",
            "For safety/nudity concepts, this wrapper uses UCE concept_type='unsafe' and guides to unconditional text by default.",
        ]
        artifacts = {"uce_weights": str(output_path)}
    else:
        commands, cwd = _rece_commands(args, concepts)
        notes = [
            "RECE starts from a UCE/full-UNet target checkpoint and then performs iterative erasure.",
            *getattr(args, "_rece_notes", []),
        ]
        artifacts = {"rece_search_root": str(args.output_dir.resolve())}

    payload = {
        "method": args.method,
        "concept_mode": mode,
        "concepts": concepts,
        "concept_count": len(concepts),
        "model_id": args.model_id,
        "seed": args.seed,
        "steps": args.steps,
        "output_dir": str(args.output_dir.resolve()),
        "working_directory": str(cwd),
        "commands": commands,
        "artifacts": artifacts,
        "notes": notes,
    }
    _write_plan(args.output_dir, payload)

    print(f"Wrote erasure plan -> {args.output_dir / 'erasure_plan.json'}")
    print(f"Wrote commands -> {args.output_dir / 'commands.sh'}")
    for command in commands:
        print(_shell_join(command))

    if args.run:
        _run_commands(commands, cwd)


if __name__ == "__main__":
    main()
