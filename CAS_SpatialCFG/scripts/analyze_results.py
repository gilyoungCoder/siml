#!/usr/bin/env python3
"""Analyze grid-search outputs across v14-v19 with canonical + legacy result support."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.result_contract import parse_category_report_file, parse_nudenet_result_file
from vlm.result_paths import (
    find_existing_result_file,
    results_txt_candidates,
)


def find_nudenet_result_file(directory: str | Path) -> Path | None:
    directory = Path(directory)
    for name in ("results_nudenet.txt", "results_nudenet_08.txt", "results_nudenet_06.txt"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def analyze_version(version_dir: str):
    results = []
    for stats_file in sorted(Path(version_dir).glob("*/stats.json")):
        config_name = stats_file.parent.name.replace("ringabell_", "")
        stats = json.loads(stats_file.read_text(encoding="utf-8"))

        nn_rate = None
        nn_file = find_nudenet_result_file(stats_file.parent)
        if nn_file is not None:
            parsed_nudenet = parse_nudenet_result_file(nn_file)
            if parsed_nudenet is not None:
                nn_rate = parsed_nudenet["unsafe_rate"] * 100

        sr = None
        relevant_sr = None
        qwen_file = find_existing_result_file(
            stats_file.parent,
            results_txt_candidates("qwen", "nudity"),
        )
        if qwen_file is not None:
            summary = parse_category_report_file(qwen_file)
            if summary is not None:
                sr = summary["sr"] * 100
                relevant_sr = summary["relevant_sr"] * 100

        trigger_rate = stats.get("trigger_rate", 0)
        avg_mask = stats.get("avg_mask_area_fused", stats.get("avg_mask_area", 0))

        results.append(
            {
                "config": config_name,
                "nn": nn_rate,
                "sr": sr,
                "relevant_sr": relevant_sr,
                "trigger": trigger_rate,
                "mask_area": avg_mask,
                "qwen_file": str(qwen_file) if qwen_file else None,
            }
        )

    return results


def main():
    versions = sys.argv[1:] if len(sys.argv) > 1 else ["v14", "v15", "v16", "v17", "v18", "v19"]
    candidate_bases = [Path.cwd() / "outputs", REPO_ROOT / "CAS_SpatialCFG" / "outputs"]
    base = next(
        (
            candidate
            for candidate in candidate_bases
            if candidate.is_dir() and any((candidate / ver).is_dir() for ver in versions)
        ),
        candidate_bases[-1],
    )

    for ver in versions:
        ver_dir = base / ver
        if not ver_dir.is_dir():
            continue

        results = analyze_version(str(ver_dir))
        if not results:
            print(f"\n=== {ver}: no results ===")
            continue

        results_with_nn = [r for r in results if r["nn"] is not None]
        results_with_sr = [r for r in results if r["sr"] is not None]
        results_with_nn.sort(key=lambda x: x["nn"])

        print(f"\n{'=' * 110}")
        print(
            f" {ver.upper()}: {len(results)} configs | "
            f"{len(results_with_nn)} with NudeNet | {len(results_with_sr)} with Qwen SR"
        )
        print(f"{'=' * 110}")
        print(f"  {'Config':<55s} {'NN%':>7s} {'SR%':>7s} {'RelSR%':>8s} {'Trig%':>6s} {'Mask':>6s}")
        print(f"  {'-' * 55} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 6}")

        for r in results_with_nn[:15]:
            nn_s = f"{r['nn']:.1f}" if r["nn"] is not None else "N/A"
            sr_s = f"{r['sr']:.1f}" if r["sr"] is not None else "N/A"
            rel_sr_s = f"{r['relevant_sr']:.1f}" if r["relevant_sr"] is not None else "N/A"
            tr_s = f"{r['trigger'] * 100:.0f}" if r["trigger"] else "0"
            ma_s = f"{r['mask_area']:.3f}" if r["mask_area"] else "0"
            print(f"  {r['config']:<55s} {nn_s:>7s} {sr_s:>7s} {rel_sr_s:>8s} {tr_s:>6s} {ma_s:>6s}")

        if len(results_with_nn) > 15:
            print(f"  ... ({len(results_with_nn) - 15} more)")

        if results_with_nn:
            nns = [r["nn"] for r in results_with_nn]
            print(
                f"\n  NN range: {min(nns):.1f}% - {max(nns):.1f}% | "
                f"median: {sorted(nns)[len(nns) // 2]:.1f}%"
            )
            best = results_with_nn[0]
            summary = f"  BEST NN: {best['config']} -> NN={best['nn']:.1f}%"
            if best["sr"] is not None:
                summary += f", SR={best['sr']:.1f}%"
            print(summary)

        if results_with_sr:
            best_sr = max(results_with_sr, key=lambda x: x["sr"])
            print(f"  BEST SR: {best_sr['config']} -> SR={best_sr['sr']:.1f}%")

        for ps in ["text", "image", "both"]:
            subset = [r for r in results_with_nn if ps + "_" in r["config"] or r["config"].startswith(ps + "_")]
            if subset:
                avg_nn = sum(r["nn"] for r in subset) / len(subset)
                best_nn = min(r["nn"] for r in subset)
                print(f"  probe={ps:5s}: avg_NN={avg_nn:.1f}%, best_NN={best_nn:.1f}% ({len(subset)} configs)")

        for gm in ["dag_adaptive", "hybrid"]:
            subset = [r for r in results_with_nn if gm in r["config"]]
            if subset:
                avg_nn = sum(r["nn"] for r in subset) / len(subset)
                best_nn = min(r["nn"] for r in subset)
                print(f"  mode={gm:12s}: avg_NN={avg_nn:.1f}%, best_NN={best_nn:.1f}% ({len(subset)} configs)")


if __name__ == "__main__":
    main()
