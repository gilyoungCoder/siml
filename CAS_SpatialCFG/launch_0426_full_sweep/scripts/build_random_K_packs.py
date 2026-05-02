"""Build random-subsample K-packs from 16-img pool with multiple seeds.
3 concepts × K∈{1,2} × 3 seeds = 18 packs.
"""
import torch, random
from pathlib import Path

CONCEPTS = ["sexual", "violence", "hate"]
KS = [1, 2]
SEEDS = [42, 43, 44]
SRC_BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/exemplars_K_per_concept"
OUT_BASE = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/exemplars_K_random"

for c in CONCEPTS:
    src = torch.load(f"{SRC_BASE}/{c}/clip_grouped_K16.pt", map_location="cpu", weights_only=False)
    out_dir = f"{OUT_BASE}/{c}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for K in KS:
        for seed in SEEDS:
            rng = random.Random(seed * 1000 + K)
            sub = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in src.items()}
            sub_t = {}
            for fname, feats in src["target_clip_features"].items():
                idx = sorted(rng.sample(range(feats.shape[0]), K))
                sub_t[fname] = feats[idx].clone()
            sub["target_clip_features"] = sub_t
            torch.save(sub, f"{out_dir}/clip_grouped_K{K}_seed{seed}.pt")
    print(f"{c}: built K∈{KS} × seed∈{SEEDS} = 6 packs")
print("Done.")
