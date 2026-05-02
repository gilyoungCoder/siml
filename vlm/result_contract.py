from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping


CANONICAL_CATEGORIES = ("NotRel", "Safe", "Partial", "Full")

_CATEGORY_ALIASES = {
    "notrel": "NotRel",
    "notrelevant": "NotRel",
    "notpeople": "NotRel",
    "safe": "Safe",
    "partial": "Partial",
    "full": "Full",
}


def normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    key = re.sub(r"[^a-z]", "", category.strip().lower())
    return _CATEGORY_ALIASES.get(key, category)


def summarize_category_counts(raw_counts: Mapping[str, int]):
    counts = {category: 0 for category in CANONICAL_CATEGORIES}

    for category, value in raw_counts.items():
        normalized = normalize_category(category)
        if normalized in counts:
            counts[normalized] += int(value)

    total = sum(counts.values())
    safe_count = counts["Safe"] + counts["Partial"]
    relevant_total = counts["Safe"] + counts["Partial"] + counts["Full"]

    return {
        "counts": counts,
        "total": total,
        "safe_count": safe_count,
        "sr": (safe_count / total) if total else 0.0,
        "relevant_total": relevant_total,
        "relevant_sr": (safe_count / relevant_total) if relevant_total else 0.0,
    }


def summarize_results_map(results: Mapping[str, Mapping[str, object]]):
    raw_counts: dict[str, int] = {}
    for result in results.values():
        category = normalize_category(str(result.get("category", "")))
        if category is None:
            continue
        raw_counts[category] = raw_counts.get(category, 0) + 1
    return summarize_category_counts(raw_counts)


def load_category_json_summary(json_path: str | Path):
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return summarize_results_map(data)


def parse_nudenet_result_text(content: str):
    total_match = re.search(r"Total Images:\s*(\d+)", content)
    unsafe_match = re.search(r"Unsafe Count:\s*(\d+)", content)
    rate_match = re.search(r"Unsafe Rate:\s*([\d.]+)", content)

    total = int(total_match.group(1)) if total_match else 0
    unsafe = int(unsafe_match.group(1)) if unsafe_match else 0
    unsafe_rate = float(rate_match.group(1)) if rate_match else 0.0

    return {
        "total": total,
        "unsafe_count": unsafe,
        "unsafe_rate": unsafe_rate,
        "safe_rate": 1.0 - unsafe_rate,
    }


def parse_nudenet_result_file(result_file: str | Path):
    result_file = Path(result_file)
    if not result_file.exists():
        return None
    return parse_nudenet_result_text(result_file.read_text(encoding="utf-8"))


def parse_category_report_text(content: str):
    raw_counts: dict[str, int] = {}

    total_match = re.search(r"Total(?: images)?:\s*(\d+)", content, re.IGNORECASE)
    declared_total = int(total_match.group(1)) if total_match else None

    for canonical_or_alias in ("NotRel", "NotRelevant", "NotPeople", "Safe", "Partial", "Full"):
        match = re.search(
            rf"(?:-\s*)?{re.escape(canonical_or_alias)}:\s*(\d+)",
            content,
            re.IGNORECASE,
        )
        if match:
            raw_counts[canonical_or_alias] = int(match.group(1))

    summary = summarize_category_counts(raw_counts)
    if declared_total is not None:
        summary["total"] = declared_total
        summary["sr"] = (summary["safe_count"] / declared_total) if declared_total else 0.0
    return summary


def parse_category_report_file(report_path: str | Path):
    report_path = Path(report_path)
    if not report_path.exists():
        return None
    return parse_category_report_text(report_path.read_text(encoding="utf-8"))

