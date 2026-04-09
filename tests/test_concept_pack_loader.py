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


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_concept_pack_supports_legacy_dict_style_access(tmp_path):
    module = load_module("concept_pack_loader", "CAS_SpatialCFG/concept_pack_loader.py")

    pack_dir = tmp_path / "violence"
    pack_dir.mkdir()
    write_json(pack_dir / "metadata.json", {
        "concept": "violence",
        "cas_threshold": 0.55,
        "recommended_probe_source": "both",
        "recommended_guide_mode": "dag_adaptive",
    })
    write_json(pack_dir / "families.json", {"families": []})
    (pack_dir / "target_keywords_primary.txt").write_text("blood\nfight\n", encoding="utf-8")
    (pack_dir / "target_keywords_secondary.txt").write_text("weapon\n", encoding="utf-8")
    (pack_dir / "anchor_keywords.txt").write_text("peaceful\nsafe\n", encoding="utf-8")

    pack = module.load_concept_pack(str(pack_dir), device="cpu")

    assert pack.get("concept") == "violence"
    assert pack["cas_threshold"] == 0.55
    assert pack.get("target_concepts") == ["blood", "fight"]
    assert pack.get("anchor_concepts") == ["peaceful", "safe"]
    assert pack.get("target_words") == ["blood", "fight", "weapon"]
    assert "concept_directions" not in pack
    assert pack.get("concept_directions") is None
