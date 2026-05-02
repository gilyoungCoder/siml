from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hygiene_docs_exist():
    assert (REPO_ROOT / "docs/output-storage.md").exists()
    assert (REPO_ROOT / "docs/archive/legacy-experiments.md").exists()


def test_gitignore_contains_core_state_and_artifact_rules():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in [
        ".pytest_cache/",
        ".omx/",
        ".omc/",
        "vlm/mjbench_results/",
        "CAS_SpatialCFG/meeting_pack/",
    ]:
        assert pattern in gitignore


def test_legacy_experiments_doc_mentions_key_legacy_families():
    content = (REPO_ROOT / "docs/archive/legacy-experiments.md").read_text(encoding="utf-8")
    for token in [
        "`3_classification_sd1.4/`",
        "`3_classification_sd1.4TC/`",
        "`sdxl-lightening-3classification/`",
        "`three_classificaiton/`",
        "`z0_clf_guidance/`",
        ]:
        assert token in content


def test_selected_legacy_readmes_have_archive_banner():
    banner = "**Archive notice:** This folder is treated as a legacy experiment branch."
    for relative_path in [
        "3_classification_sd1.4/README.md",
        "3_classification_sd1.4TC/README.md",
        "3_classification_sd1.4_csv/README.md",
        "4_classification_sd1.4/README.md",
        "5_classificaiton/README.md",
        "three_classificaiton/README.md",
        "three_classificaiton_new/README.md",
        "three_classificaiton_scale/README.md",
        "three_classificaiton_Clip/README.md",
        "sdxl-lightening-11classification/README.md",
        "sdxl-lightening-11classificationNegativeLearning/README.md",
        "sdxl-lightening-31classification/README.md",
        "sdxl-lightening-3classification/README.md",
        "sdxl-lightening-3classificationNegtiveLearning/README.md",
        "sdxl-lightening-4classification/README.md",
        "sdxl-lightening-5classification/README.md",
        "sdxl-lightening-5classification_hier/README.md",
        "sdxl-nag-3classification/README.md",
        "10_classificaiton/README.md",
    ]:
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert banner in content
