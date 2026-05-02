from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_prompt_lines_reads_non_empty_lines(tmp_path):
    module = load_module("prepare_concept_subspace", "CAS_SpatialCFG/prepare_concept_subspace.py")

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first\n\nsecond\n", encoding="utf-8")

    assert module.read_prompt_lines(str(prompt_file)) == ["first", "second"]
    assert module.read_prompt_lines(None) == []
