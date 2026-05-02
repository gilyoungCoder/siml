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


def test_audit_pack_detects_missing_and_complete_states(tmp_path):
    module = load_module("check_concept_pack_completeness", "CAS_SpatialCFG/scripts/check_concept_pack_completeness.py")

    missing = module.audit_pack(tmp_path, "violence")
    assert missing.status == "missing_pack_dir"

    pack_dir = tmp_path / "sexual"
    pack_dir.mkdir()
    for filename in module.REQUIRED_FILES:
        (pack_dir / filename).write_text("x", encoding="utf-8")
    for filename in module.OPTIONAL_ARTIFACTS:
        (pack_dir / filename).write_text("artifact", encoding="utf-8")
    (pack_dir / "exemplar_images").mkdir()

    complete = module.audit_pack(tmp_path, "sexual")
    assert complete.status == "complete"
    assert complete.required_missing == []
    assert complete.optional_missing == []
    assert complete.exemplar_images_exists is True
