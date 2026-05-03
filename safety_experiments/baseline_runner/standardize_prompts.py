from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    MODEL_FAMILIES,
    MODEL_SPECS,
    PROMPT_COLUMN_CANDIDATES,
)


def _detect_prompt_column(df: pd.DataFrame, prompt_col: str | None) -> str:
    if prompt_col:
        if prompt_col not in df.columns:
            raise ValueError(
                f"Prompt column '{prompt_col}' not found. Available columns: {list(df.columns)}"
            )
        return prompt_col

    for candidate in PROMPT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not infer prompt column. Pass --prompt-col explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def _read_source(input_path: Path, prompt_col: str | None) -> tuple[pd.DataFrame, str]:
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(input_path, sep=sep)
        detected_col = _detect_prompt_column(df, prompt_col)
        return df, detected_col

    if prompt_col:
        raise ValueError("--prompt-col is only valid for tabular input files.")

    prompts = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame({"prompt": prompts}), "prompt"


def standardize_prompts(
    input_path: Path,
    output_path: Path,
    dataset: str | None,
    concept: str | None,
    prompt_col: str | None,
    seed: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    start: int,
    limit: int | None,
) -> pd.DataFrame:
    df, detected_prompt_col = _read_source(input_path, prompt_col)

    if start:
        df = df.iloc[start:]
    if limit is not None:
        df = df.iloc[:limit]
    df = df.reset_index(drop=False).rename(columns={"index": "source_row"})

    raw_prompts = df[detected_prompt_col]
    keep = raw_prompts.notna() & raw_prompts.astype(str).str.strip().ne("")
    df = df[keep].reset_index(drop=True)
    prompts = df[detected_prompt_col].astype(str).str.strip().reset_index(drop=True)

    out = pd.DataFrame(
        {
            "prompt_id": range(len(df)),
            "prompt": prompts,
            "evaluation_seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "dataset": dataset or input_path.stem,
            "concept": concept or "",
            "source_path": str(input_path),
            "source_row": df["source_row"],
            "source_prompt_col": detected_prompt_col,
        }
    )

    if "case_number" in df.columns:
        out["source_case_number"] = df["case_number"]
    else:
        out["source_case_number"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CSV/TXT prompt files to the shared baseline-runner schema."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw CSV/TXT prompt file")
    parser.add_argument("--output", type=Path, required=True, help="Standardized CSV path")
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        default=None,
        help="Use model-family defaults for steps/guidance/size unless explicitly set.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset label to record")
    parser.add_argument("--concept", type=str, default=None, help="Concept label to record")
    parser.add_argument("--prompt-col", type=str, default=None, help="Prompt column for CSV/TSV")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="Row offset before standardization")
    parser.add_argument("--limit", type=int, default=None, help="Optional max prompt count")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    defaults = MODEL_SPECS.get(args.model_family or "sd14", {})
    steps = args.steps if args.steps is not None else int(defaults.get("steps", DEFAULT_STEPS))
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(defaults.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    )
    height = args.height if args.height is not None else int(defaults.get("height", DEFAULT_HEIGHT))
    width = args.width if args.width is not None else int(defaults.get("width", DEFAULT_WIDTH))
    out = standardize_prompts(
        input_path=args.input,
        output_path=args.output,
        dataset=args.dataset,
        concept=args.concept,
        prompt_col=args.prompt_col,
        seed=args.seed,
        steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        start=args.start,
        limit=args.limit,
    )
    print(f"Wrote {len(out)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
