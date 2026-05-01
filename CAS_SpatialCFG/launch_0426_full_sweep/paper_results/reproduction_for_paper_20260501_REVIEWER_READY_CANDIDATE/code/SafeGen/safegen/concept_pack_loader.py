"""
Concept Pack Loader — multi-concept configuration system.

Each concept pack contains metadata, family definitions, keywords, prompts,
and optionally precomputed tensors for a specific unsafe concept.

Concept pack directory layout:
    concept_packs/<concept>/
        metadata.json
        families.json
        target_prompts.txt
        anchor_prompts.txt
        target_keywords_primary.txt
        target_keywords_secondary.txt
        anchor_keywords.txt
        concept_directions.pt       (optional, from prepare_concept_subspace.py)
        clip_exemplar_projected.pt  (optional, from prepare_clip_exemplar.py)

Usage:
    from safegen.concept_pack_loader import load_concept_pack

    pack = load_concept_pack("configs/concept_packs/violence")
    print(pack.target_concepts, pack.cas_threshold)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class ConceptFamily:
    """One family within a concept (e.g., weapon_threat within violence)."""
    name: str
    target_words: List[str]
    anchor_words: List[str]
    mapping_strength: str  # strong / medium / weak
    pilot: bool


@dataclass
class ConceptPack:
    """One loaded concept configuration."""
    name: str
    cas_threshold: float
    probe_source: str          # text / image / both
    guide_mode: str            # anchor_inpaint / hybrid
    families: List[ConceptFamily]
    target_keywords_primary: List[str]
    target_keywords_secondary: List[str]
    anchor_keywords: List[str]
    target_prompts: List[str]
    anchor_prompts: List[str]
    concept_directions: Optional[dict] = None
    clip_embeddings: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    pack_dir: str = ""

    @property
    def target_concepts(self) -> List[str]:
        if self.target_keywords_primary:
            return self.target_keywords_primary
        if self.target_prompts:
            return self.target_prompts
        return [self.name]

    @property
    def anchor_concepts(self) -> List[str]:
        if self.anchor_keywords:
            return self.anchor_keywords
        if self.anchor_prompts:
            return self.anchor_prompts
        return [f"safe {self.name}"]

    @property
    def target_words(self) -> List[str]:
        return _dedupe(self.target_keywords_primary + self.target_keywords_secondary)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_concept_pack(pack_dir: str, device: str = "cpu") -> ConceptPack:
    """
    Load a concept pack from a directory.

    Args:
        pack_dir: Path to concept pack directory
        device: Device for loading .pt tensors

    Returns:
        ConceptPack with all available data
    """
    d = Path(pack_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Concept pack not found: {d}")

    metadata = json.loads((d / "metadata.json").read_text())
    families_data = json.loads((d / "families.json").read_text())

    families = [
        ConceptFamily(
            name=f["name"],
            target_words=f.get("target_words", []),
            anchor_words=f.get("anchor_words", []),
            mapping_strength=f.get("mapping_strength", "medium"),
            pilot=f.get("pilot", False),
        )
        for f in families_data.get("families", [])
    ]

    # Load optional precomputed tensors
    concept_dirs = None
    clip_emb = None
    for pt_name, attr in [("concept_directions.pt", "concept_dirs"), ("clip_exemplar_projected.pt", "clip_emb")]:
        pt_path = d / pt_name
        if pt_path.exists():
            if attr == "concept_dirs":
                concept_dirs = torch.load(pt_path, map_location=device, weights_only=False)
            else:
                clip_emb = torch.load(pt_path, map_location=device, weights_only=False)

    return ConceptPack(
        name=metadata.get("concept", d.name),
        cas_threshold=metadata.get("cas_threshold", 0.5),
        probe_source=metadata.get("recommended_probe_source", "text"),
        guide_mode=metadata.get("recommended_guide_mode", "anchor_inpaint"),
        families=families,
        target_keywords_primary=_read_lines(d / "target_keywords_primary.txt"),
        target_keywords_secondary=_read_lines(d / "target_keywords_secondary.txt"),
        anchor_keywords=_read_lines(d / "anchor_keywords.txt"),
        target_prompts=_read_lines(d / "target_prompts.txt"),
        anchor_prompts=_read_lines(d / "anchor_prompts.txt"),
        concept_directions=concept_dirs,
        clip_embeddings=clip_emb,
        metadata=metadata,
        pack_dir=str(d),
    )


def load_multiple_packs(pack_dirs: List[str], device: str = "cpu") -> List[ConceptPack]:
    """Load multiple concept packs for simultaneous erasing."""
    packs = []
    for pd in pack_dirs:
        print(f"Loading concept pack: {pd}")
        packs.append(load_concept_pack(pd, device=device))
    print(f"Loaded {len(packs)} concept packs: {[p.name for p in packs]}")
    return packs
