from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_results_writes_canonical_and_legacy_qwen2_outputs(tmp_path):
    module = load_module("opensource_vlm_nudity", "vlm/opensource_vlm_nudity.py")

    results = {
        "a.png": {"category": "NotPeople"},
        "b.png": {"category": "Safe"},
        "c.png": {"category": "Partial"},
        "d.png": {"category": "Full"},
    }
    counts = {"NotPeople": 1, "Safe": 1, "Partial": 1, "Full": 1}

    module.save_results(str(tmp_path), results, counts, "qwen2_vl", concept="nudity")

    canonical_json = tmp_path / "categories_qwen_nudity.json"
    canonical_txt = tmp_path / "results_qwen_nudity.txt"
    legacy_json = tmp_path / "categories_qwen2_vl.json"
    legacy_txt = tmp_path / "results.txt"

    assert canonical_json.exists()
    assert canonical_txt.exists()
    assert legacy_json.exists()
    assert legacy_txt.exists()

    saved = json.loads(canonical_json.read_text(encoding="utf-8"))
    assert saved == results

    summary = canonical_txt.read_text(encoding="utf-8")
    assert "Total images: 4" in summary
    assert "NotRel: 1" in summary
    assert "SR (Safe+Partial): 2/4 (50.0%)" in summary
