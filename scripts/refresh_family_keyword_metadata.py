#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch


def parse_family_file(path: Path) -> dict[str, dict[str, object]]:
    family_map: dict[str, dict[str, object]] = {}
    current_family: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# Family:"):
            current_family = line.replace("# Family:", "", 1).strip().split("->")[0].strip()
            current_family = current_family.split(" (")[0].strip()
            family_map.setdefault(current_family, {"prompts": [], "keywords": []})
            continue
        if line.startswith("# Keywords:"):
            if current_family is None:
                raise ValueError(f"{path}: found keywords before family header")
            keywords = [part.strip() for part in line.replace("# Keywords:", "", 1).split("|") if part.strip()]
            family_map.setdefault(current_family, {"prompts": [], "keywords": []})
            family_map[current_family]["keywords"] = keywords
            continue
        if line.startswith("#"):
            continue
        if current_family is None:
            raise ValueError(f"{path}: found prompt before family header")
        family_map.setdefault(current_family, {"prompts": [], "keywords": []})
        family_map[current_family]["prompts"].append(line)

    return family_map


def refresh_concept(root: Path, concept: str) -> None:
    target_txt = root / "SafeGen" / "configs" / "exemplar_prompts_v2" / f"{concept}_target.txt"
    anchor_txt = root / "SafeGen" / "configs" / "exemplar_prompts_v2" / f"{concept}_anchor.txt"
    clip_grouped = root / "CAS_SpatialCFG" / "exemplars" / "concepts_v2" / concept / "clip_grouped.pt"

    target_info = parse_family_file(target_txt)
    anchor_info = parse_family_file(anchor_txt)
    data = torch.load(clip_grouped, map_location="cpu", weights_only=False)

    family_names = data.get("family_names") or list(data.get("family_metadata", {}).keys())
    family_meta = data.get("family_metadata", {})

    for family in family_names:
        if family not in target_info or family not in anchor_info:
            raise KeyError(f"{concept}: missing family '{family}' in prompt txt")
        meta = dict(family_meta.get(family, {}))
        meta["target_prompts"] = target_info[family]["prompts"]
        meta["anchor_prompts"] = anchor_info[family]["prompts"]
        meta["target_words"] = target_info[family]["keywords"]
        meta["anchor_words"] = anchor_info[family]["keywords"]
        family_meta[family] = meta

    data["family_metadata"] = family_meta
    torch.save(data, clip_grouped)

    print(f"[updated] {concept}: {clip_grouped}")
    for family in family_names:
        meta = family_meta[family]
        print(f"  {family}")
        print(f"    target_words={meta.get('target_words')}")
        print(f"    anchor_words={meta.get('anchor_words')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject family-specific keyword metadata into clip_grouped.pt")
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["sexual", "violent", "disturbing", "illegal"],
        help="Concept names under SafeGen/configs/exemplar_prompts_v2 and CAS_SpatialCFG/exemplars/concepts_v2",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    args = parser.parse_args()

    for concept in args.concepts:
        refresh_concept(args.repo_root, concept)


if __name__ == "__main__":
    main()
