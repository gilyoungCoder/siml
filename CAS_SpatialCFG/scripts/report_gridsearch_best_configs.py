#!/usr/bin/env python3
"""Build OMC-friendly best-config and Pareto reports from existing grid-search outputs."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.result_contract import parse_category_report_file, parse_nudenet_result_file
from vlm.result_paths import find_existing_result_file, results_txt_candidates


@dataclass
class ConfigRow:
    version: str
    config: str
    config_dir: Path
    nn: float | None
    sr: float | None
    relevant_sr: float | None
    trigger: float
    mask_area: float
    n_images: int
    qwen_file: str | None

    @property
    def full_name(self) -> str:
        return f"{self.version}/{self.config}"


def find_outputs_root(versions: list[str]) -> Path:
    candidates = [Path.cwd() / "outputs", REPO_ROOT / "CAS_SpatialCFG" / "outputs"]
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir() and any((candidate / version).is_dir() for version in versions)
        ),
        candidates[-1],
    )


def find_nudenet_file(directory: Path) -> Path | None:
    for name in ("results_nudenet.txt", "results_nudenet_08.txt", "results_nudenet_06.txt"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def scan_version(outputs_root: Path, version: str) -> list[ConfigRow]:
    version_dir = outputs_root / version
    if not version_dir.is_dir():
        return []

    rows: list[ConfigRow] = []
    for stats_file in sorted(version_dir.glob("ringabell_*/stats.json")):
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        config_dir = stats_file.parent
        config = config_dir.name.replace("ringabell_", "")

        nn = None
        nn_file = find_nudenet_file(config_dir)
        if nn_file is not None:
            parsed = parse_nudenet_result_file(nn_file)
            if parsed is not None:
                nn = parsed["unsafe_rate"] * 100

        sr = None
        relevant_sr = None
        qwen_file = find_existing_result_file(config_dir, results_txt_candidates("qwen", "nudity"))
        if qwen_file is not None:
            parsed = parse_category_report_file(qwen_file)
            if parsed is not None:
                sr = parsed["sr"] * 100
                relevant_sr = parsed["relevant_sr"] * 100

        rows.append(
            ConfigRow(
                version=version,
                config=config,
                config_dir=config_dir,
                nn=nn,
                sr=sr,
                relevant_sr=relevant_sr,
                trigger=stats.get("trigger_rate", 0.0),
                mask_area=stats.get("avg_mask_area_fused", stats.get("avg_mask_area", 0.0)),
                n_images=stats.get("total_images", 0),
                qwen_file=str(qwen_file) if qwen_file else None,
            )
        )

    return rows


def dense_rank(values: list[float], reverse: bool = False) -> dict[float, int]:
    ordered = sorted(set(values), reverse=reverse)
    return {value: idx + 1 for idx, value in enumerate(ordered)}


def pick_best_balanced(rows: list[ConfigRow]) -> ConfigRow | None:
    scored = [row for row in rows if row.nn is not None and row.sr is not None]
    if not scored:
        return None

    nn_rank = dense_rank([row.nn for row in scored], reverse=False)
    sr_rank = dense_rank([row.sr for row in scored], reverse=True)

    return min(
        scored,
        key=lambda row: (
            nn_rank[row.nn] + sr_rank[row.sr],
            nn_rank[row.nn],
            sr_rank[row.sr],
            -row.sr,
            row.nn,
        ),
    )


def compute_pareto_frontier(rows: list[ConfigRow]) -> list[ConfigRow]:
    candidates = [row for row in rows if row.nn is not None and row.sr is not None]
    frontier: list[ConfigRow] = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            better_or_equal = other.nn <= row.nn and other.sr >= row.sr
            strictly_better = other.nn < row.nn or other.sr > row.sr
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (row.nn, -row.sr, row.version, row.config))


def write_csv(path: Path, rows: list[ConfigRow]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "version",
                "config",
                "full_name",
                "nn",
                "sr",
                "relevant_sr",
                "trigger",
                "mask_area",
                "n_images",
                "config_dir",
                "qwen_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "version": row.version,
                    "config": row.config,
                    "full_name": row.full_name,
                    "nn": "" if row.nn is None else f"{row.nn:.4f}",
                    "sr": "" if row.sr is None else f"{row.sr:.4f}",
                    "relevant_sr": "" if row.relevant_sr is None else f"{row.relevant_sr:.4f}",
                    "trigger": f"{row.trigger:.6f}",
                    "mask_area": f"{row.mask_area:.6f}",
                    "n_images": row.n_images,
                    "config_dir": str(row.config_dir),
                    "qwen_file": row.qwen_file or "",
                }
            )


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def build_summary_markdown(rows: list[ConfigRow], frontier: list[ConfigRow], versions: list[str], outputs_root: Path) -> str:
    by_version: dict[str, list[ConfigRow]] = defaultdict(list)
    for row in rows:
        by_version[row.version].append(row)

    generated_at = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Current Grid-Search Best Config Summary",
        "",
        f"- Generated: `{generated_at}`",
        f"- Outputs root: `{outputs_root}`",
        f"- Versions scanned: `{', '.join(versions)}`",
        "",
        "## Coverage by version",
        "",
        "| Version | Configs | With NudeNet | With SR | Pareto points |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    frontier_by_version: dict[str, list[ConfigRow]] = defaultdict(list)
    for row in frontier:
        frontier_by_version[row.version].append(row)

    for version in versions:
        subset = by_version.get(version, [])
        with_nn = sum(1 for row in subset if row.nn is not None)
        with_sr = sum(1 for row in subset if row.sr is not None)
        lines.append(f"| {version} | {len(subset)} | {with_nn} | {with_sr} | {len(frontier_by_version.get(version, []))} |")

    lines.extend(
        [
            "",
            "## Best configs by version",
            "",
            "| Version | Best NN | NN% | SR% | Best SR | NN% | SR% | Best balanced rank-sum | NN% | SR% |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )

    for version in versions:
        subset = by_version.get(version, [])
        with_nn = [row for row in subset if row.nn is not None]
        with_sr = [row for row in subset if row.sr is not None]
        best_nn = min(with_nn, key=lambda row: row.nn) if with_nn else None
        best_sr = max(with_sr, key=lambda row: row.sr) if with_sr else None
        best_balanced = pick_best_balanced(subset)
        lines.append(
            "| {version} | {best_nn_name} | {best_nn_nn} | {best_nn_sr} | "
            "{best_sr_name} | {best_sr_nn} | {best_sr_sr} | {best_bal_name} | {best_bal_nn} | {best_bal_sr} |".format(
                version=version,
                best_nn_name=best_nn.config if best_nn else "N/A",
                best_nn_nn=format_metric(best_nn.nn if best_nn else None),
                best_nn_sr=format_metric(best_nn.sr if best_nn else None),
                best_sr_name=best_sr.config if best_sr else "N/A",
                best_sr_nn=format_metric(best_sr.nn if best_sr else None),
                best_sr_sr=format_metric(best_sr.sr if best_sr else None),
                best_bal_name=best_balanced.config if best_balanced else "N/A",
                best_bal_nn=format_metric(best_balanced.nn if best_balanced else None),
                best_bal_sr=format_metric(best_balanced.sr if best_balanced else None),
            )
        )

    lines.extend(
        [
            "",
            "## Global Pareto frontier (NN low, SR high)",
            "",
            "| Version | Config | NN% | SR% | Relevant_SR% |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in frontier:
        lines.append(
            f"| {row.version} | {row.config} | {format_metric(row.nn)} | {format_metric(row.sr)} | {format_metric(row.relevant_sr)} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate best-config and Pareto reports from existing grid-search outputs.")
    parser.add_argument("--versions", nargs="+", default=["v14", "v15", "v16", "v17", "v18", "v19"])
    parser.add_argument(
        "--report-dir",
        default=str(REPO_ROOT / "docs" / "omc_reports"),
        help="Directory to receive markdown/csv report artifacts",
    )
    args = parser.parse_args()

    outputs_root = find_outputs_root(args.versions)
    rows: list[ConfigRow] = []
    for version in args.versions:
        rows.extend(scan_version(outputs_root, version))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    frontier = compute_pareto_frontier(rows)
    write_csv(report_dir / "current_gridsearch_config_index.csv", rows)
    write_csv(report_dir / "current_gridsearch_pareto_frontier.csv", frontier)

    summary_md = build_summary_markdown(rows, frontier, args.versions, outputs_root)
    (report_dir / "current_gridsearch_best_config_summary.md").write_text(summary_md, encoding="utf-8")
    (report_dir / "current_gridsearch_pareto_frontier.md").write_text(
        "\n".join(
            [
                "# Current Grid-Search Pareto Frontier",
                "",
                "This file is generated from existing outputs. Lower `NN%` is better; higher `SR%` is better.",
                "",
                "| Version | Config | NN% | SR% | Relevant_SR% |",
                "| --- | --- | ---: | ---: | ---: |",
                *[
                    f"| {row.version} | {row.config} | {format_metric(row.nn)} | {format_metric(row.sr)} | {format_metric(row.relevant_sr)} |"
                    for row in frontier
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} config rows to {report_dir / 'current_gridsearch_config_index.csv'}")
    print(f"Wrote {len(frontier)} Pareto rows to {report_dir / 'current_gridsearch_pareto_frontier.csv'}")
    print(f"Wrote markdown summary to {report_dir / 'current_gridsearch_best_config_summary.md'}")


if __name__ == "__main__":
    main()
