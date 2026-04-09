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


def write_stats(directory: Path, trigger_rate: float = 0.1, mask_area: float = 0.2):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stats.json").write_text(
        json.dumps(
            {
                "trigger_rate": trigger_rate,
                "avg_mask_area_fused": mask_area,
                "total_images": 316,
            }
        ),
        encoding="utf-8",
    )


def test_scan_version_reads_legacy_qwen_and_nudenet_outputs(tmp_path):
    module = load_module("report_gridsearch_best_configs", "CAS_SpatialCFG/scripts/report_gridsearch_best_configs.py")

    outputs = tmp_path / "outputs" / "v14" / "ringabell_test_cfg"
    write_stats(outputs)
    (outputs / "results_nudenet.txt").write_text(
        "Total Images: 10\nUnsafe Count: 1\nUnsafe Rate: 0.10\n",
        encoding="utf-8",
    )
    (outputs / "results.txt").write_text(
        "📊 분포:\n"
        " - Safe: 4개 (40.0%)\n"
        " - Partial: 2개 (20.0%)\n"
        " - Full: 3개 (30.0%)\n"
        " - NotPeople: 1개 (10.0%)\n",
        encoding="utf-8",
    )

    rows = module.scan_version(tmp_path / "outputs", "v14")
    assert len(rows) == 1
    row = rows[0]
    assert row.version == "v14"
    assert row.config == "test_cfg"
    assert row.nn == 10.0
    assert row.sr == 60.0
    assert round(row.relevant_sr, 4) == round((6 / 9) * 100, 4)


def test_compute_pareto_frontier_filters_dominated_points(tmp_path):
    module = load_module("report_gridsearch_best_configs", "CAS_SpatialCFG/scripts/report_gridsearch_best_configs.py")

    outputs_root = tmp_path / "outputs"
    for version, config, unsafe_rate, results_txt in [
        (
            "v14",
            "ringabell_cfg_a",
            "0.10",
            "📊 분포:\n - Safe: 4개 (40.0%)\n - Partial: 2개 (20.0%)\n - Full: 3개 (30.0%)\n - NotPeople: 1개 (10.0%)\n",
        ),
        (
            "v14",
            "ringabell_cfg_b",
            "0.20",
            "📊 분포:\n - Safe: 5개 (50.0%)\n - Partial: 2개 (20.0%)\n - Full: 2개 (20.0%)\n - NotPeople: 1개 (10.0%)\n",
        ),
        (
            "v14",
            "ringabell_cfg_c",
            "0.25",
            "📊 분포:\n - Safe: 3개 (30.0%)\n - Partial: 1개 (10.0%)\n - Full: 5개 (50.0%)\n - NotPeople: 1개 (10.0%)\n",
        ),
    ]:
        directory = outputs_root / version / config
        write_stats(directory)
        (directory / "results_nudenet.txt").write_text(
            f"Total Images: 10\nUnsafe Count: 1\nUnsafe Rate: {unsafe_rate}\n",
            encoding="utf-8",
        )
        (directory / "results.txt").write_text(results_txt, encoding="utf-8")

    rows = module.scan_version(outputs_root, "v14")
    frontier = module.compute_pareto_frontier(rows)
    frontier_names = [row.config for row in frontier]

    assert "cfg_c" not in frontier_names
    assert "cfg_a" in frontier_names
    assert "cfg_b" in frontier_names
