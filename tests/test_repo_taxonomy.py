from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_docs_exist():
    assert (REPO_ROOT / "docs/repo-taxonomy.md").exists()
    assert (REPO_ROOT / "docs/repo-layout.md").exists()
    assert (REPO_ROOT / "docs/active-workflow.md").exists()


def test_readme_points_to_taxonomy_docs():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/repo-taxonomy.md" in readme
    assert "docs/repo-layout.md" in readme
    assert "docs/active-workflow.md" in readme
    assert "Active Core" in readme


def test_claude_points_to_canonical_guides():
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for required in [
        "docs/repo-taxonomy.md",
        "docs/repo-layout.md",
        "docs/active-workflow.md",
        "docs/runtime-config.md",
        "docs/metrics-contract.md",
    ]:
        assert required in claude


def test_active_workflow_doc_declares_core_flow():
    workflow = (REPO_ROOT / "docs/active-workflow.md").read_text(encoding="utf-8")
    for required in [
        "## Canonical active workflow",
        "`CAS_SpatialCFG/`",
        "`vlm/`",
        "`scripts/`",
        "Generation",
        "Evaluation",
        "Aggregation",
    ]:
        assert required in workflow


def test_taxonomy_doc_declares_active_core_and_legacy_sections():
    taxonomy = (REPO_ROOT / "docs/repo-taxonomy.md").read_text(encoding="utf-8")

    for required in [
        "## Active Core",
        "## Secondary Baselines",
        "## Legacy / Archive",
        "`CAS_SpatialCFG/`",
        "`SAFREE/`",
        "`vlm/`",
        "`scripts/`",
    ]:
        assert required in taxonomy


def test_declared_active_core_directories_exist():
    for directory in ["CAS_SpatialCFG", "SAFREE", "vlm", "scripts"]:
        assert (REPO_ROOT / directory).is_dir()
