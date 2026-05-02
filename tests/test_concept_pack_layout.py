from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_BASE = REPO_ROOT / "docs" / "neurips_plan" / "multi_concept" / "concept_packs"
EXPECTED_PACKS = [
    "sexual",
    "violence",
    "shocking",
    "self-harm",
    "illegal_activity",
    "harassment",
    "hate",
]
REQUIRED_FILES = [
    "metadata.json",
    "families.json",
    "target_prompts.txt",
    "anchor_prompts.txt",
    "target_keywords_primary.txt",
    "target_keywords_secondary.txt",
    "anchor_keywords.txt",
]


def test_expected_concept_pack_directories_exist():
    for pack in EXPECTED_PACKS:
        assert (PACK_BASE / pack).is_dir(), pack


def test_expected_concept_pack_required_files_exist():
    for pack in EXPECTED_PACKS:
        for filename in REQUIRED_FILES:
            assert (PACK_BASE / pack / filename).exists(), f"{pack}/{filename}"
