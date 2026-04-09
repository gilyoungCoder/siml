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


def test_qwen_canonical_names_use_short_stem():
    module = load_module("result_paths", "vlm/result_paths.py")

    assert module.canonical_model_stem("qwen") == "qwen"
    assert module.canonical_model_stem("qwen3_vl") == "qwen"
    assert module.categories_json_name("qwen3_vl", "nudity") == "categories_qwen_nudity.json"
    assert module.results_txt_name("qwen3_vl", "nudity") == "results_qwen_nudity.txt"


def test_qwen_candidate_lists_prefer_canonical_then_legacy():
    module = load_module("result_paths", "vlm/result_paths.py")

    assert module.categories_json_candidates("qwen", "nudity") == [
        "categories_qwen_nudity.json",
        "categories_qwen3_vl_nudity.json",
        "categories_qwen2_vl.json",
    ]
    assert module.results_txt_candidates("qwen", "nudity") == [
        "results_qwen_nudity.txt",
        "results_qwen3_vl_nudity.txt",
        "results.txt",
    ]


def test_find_existing_result_file_uses_first_available_candidate(tmp_path):
    module = load_module("result_paths", "vlm/result_paths.py")

    legacy = tmp_path / "results_qwen3_vl_nudity.txt"
    legacy.write_text("legacy", encoding="utf-8")

    found = module.find_existing_result_file(tmp_path, module.results_txt_candidates("qwen", "nudity"))
    assert found == legacy

    canonical = tmp_path / "results_qwen_nudity.txt"
    canonical.write_text("canonical", encoding="utf-8")

    found = module.find_existing_result_file(tmp_path, module.results_txt_candidates("qwen", "nudity"))
    assert found == canonical


def test_qwen2_legacy_json_is_considered_a_valid_candidate(tmp_path):
    module = load_module("result_paths", "vlm/result_paths.py")

    legacy = tmp_path / "categories_qwen2_vl.json"
    legacy.write_text("{}", encoding="utf-8")

    found = module.find_existing_result_file(tmp_path, module.categories_json_candidates("qwen", "nudity"))
    assert found == legacy
