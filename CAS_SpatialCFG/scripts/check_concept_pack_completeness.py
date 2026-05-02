#!/usr/bin/env python3
"""Audit multi-concept pack completeness and write OMC-friendly report artifacts."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_BASE = REPO_ROOT / "docs" / "neurips_plan" / "multi_concept" / "concept_packs"
DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "omc_reports"
EXPECTED_CORE_PACKS = [
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
OPTIONAL_ARTIFACTS = [
    "concept_directions.pt",
    "clip_patch_tokens.pt",
    "contrastive_embeddings.pt",
]


@dataclass
class PackAudit:
    name: str
    exists: bool
    required_present: list[str]
    required_missing: list[str]
    optional_present: list[str]
    optional_missing: list[str]
    exemplar_images_exists: bool

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing_pack_dir"
        if self.required_missing:
            return "missing_required_files"
        if self.optional_missing:
            return "ready_for_text_only_or_partial_image_modes"
        return "complete"


def audit_pack(base_dir: Path, pack_name: str) -> PackAudit:
    pack_dir = base_dir / pack_name
    if not pack_dir.is_dir():
        return PackAudit(
            name=pack_name,
            exists=False,
            required_present=[],
            required_missing=REQUIRED_FILES.copy(),
            optional_present=[],
            optional_missing=OPTIONAL_ARTIFACTS.copy(),
            exemplar_images_exists=False,
        )

    required_present = [name for name in REQUIRED_FILES if (pack_dir / name).exists()]
    required_missing = [name for name in REQUIRED_FILES if name not in required_present]
    optional_present = [name for name in OPTIONAL_ARTIFACTS if (pack_dir / name).exists()]
    optional_missing = [name for name in OPTIONAL_ARTIFACTS if name not in optional_present]
    exemplar_images_exists = (pack_dir / "exemplar_images").is_dir()

    return PackAudit(
        name=pack_name,
        exists=True,
        required_present=required_present,
        required_missing=required_missing,
        optional_present=optional_present,
        optional_missing=optional_missing,
        exemplar_images_exists=exemplar_images_exists,
    )


def build_markdown(audits: list[PackAudit], base_dir: Path) -> str:
    lines = [
        "# Current Concept Pack Completeness",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Pack base: `{base_dir}`",
        "",
        "| Pack | Status | Required missing | Optional missing | exemplar_images/ |",
        "| --- | --- | --- | --- | --- |",
    ]

    for audit in audits:
        lines.append(
            f"| {audit.name} | {audit.status} | "
            f"{', '.join(audit.required_missing) if audit.required_missing else 'none'} | "
            f"{', '.join(audit.optional_missing) if audit.optional_missing else 'none'} | "
            f"{'yes' if audit.exemplar_images_exists else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `complete`: required metadata/text files and optional precomputed artifacts are all present.",
            "- `missing_required_files`: pack exists but the text/metadata contract is incomplete.",
            "- `ready_for_text_only_or_partial_image_modes`: required files exist, but at least one optional artifact is still missing.",
            "- `missing_pack_dir`: expected pack directory itself is missing.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Audit concept-pack completeness and emit OMC-friendly reports.")
    parser.add_argument("--pack-base", default=str(DEFAULT_PACK_BASE))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--expected-packs", nargs="+", default=EXPECTED_CORE_PACKS)
    args = parser.parse_args()

    pack_base = Path(args.pack_base)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    audits = [audit_pack(pack_base, pack_name) for pack_name in args.expected_packs]
    md = build_markdown(audits, pack_base)

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pack_base": str(pack_base),
        "expected_packs": args.expected_packs,
        "audits": [
            {
                "name": audit.name,
                "status": audit.status,
                "exists": audit.exists,
                "required_present": audit.required_present,
                "required_missing": audit.required_missing,
                "optional_present": audit.optional_present,
                "optional_missing": audit.optional_missing,
                "exemplar_images_exists": audit.exemplar_images_exists,
            }
            for audit in audits
        ],
    }

    (report_dir / "current_concept_pack_completeness.md").write_text(md, encoding="utf-8")
    (report_dir / "current_concept_pack_completeness.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote concept-pack markdown report to {report_dir / 'current_concept_pack_completeness.md'}")
    print(f"Wrote concept-pack JSON report to {report_dir / 'current_concept_pack_completeness.json'}")


if __name__ == "__main__":
    main()
