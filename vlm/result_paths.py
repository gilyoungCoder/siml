from __future__ import annotations

from pathlib import Path
from typing import Iterable


def canonical_model_stem(model_name: str) -> str:
    if model_name in {"qwen", "qwen3_vl"}:
        return "qwen3_vl"
    if model_name == "qwen2_vl":
        return "qwen2_vl"
    return model_name


def categories_json_name(model_name: str, concept: str) -> str:
    return f"categories_{canonical_model_stem(model_name)}_{concept}.json"


def results_txt_name(model_name: str, concept: str) -> str:
    return f"results_{canonical_model_stem(model_name)}_{concept}.txt"


def categories_json_candidates(model_name: str, concept: str) -> list[str]:
    canonical = categories_json_name(model_name, concept)
    if canonical_model_stem(model_name) == "qwen":
        return [
            canonical,
            f"categories_qwen3_vl_{concept}.json",
            "categories_qwen2_vl.json",
        ]
    return [canonical]


def results_txt_candidates(model_name: str, concept: str) -> list[str]:
    canonical = results_txt_name(model_name, concept)
    if canonical_model_stem(model_name) == "qwen":
        return [
            canonical,
            f"results_qwen3_vl_{concept}.txt",
            "results.txt",
        ]
    return [canonical]


def find_existing_result_file(directory: str | Path, candidates: Iterable[str]) -> Path | None:
    directory = Path(directory)
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path
    return None
