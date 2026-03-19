#!/usr/bin/env python3
"""
Generate 4x4 image grids for paper figures.
Rows: different prompts, Columns: SD Baseline | SAFREE | SAFREE+Ours | Ours
Selection: SD=Full, Ours=Safe, SAFREE includes some NotRel (over-suppressed)
"""
import json
import os
import re
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BASE = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs"

METHODS_ORDER = ["SD Baseline", "SAFREE", "SAFREE+Ours", "Ours"]

METHODS = {
    "SD Baseline": {
        "ringabell": f"{BASE}/final_ringabell/sd_baseline",
        "unlearndiff": f"{BASE}/final_unlearndiff/sd_baseline",
    },
    "SAFREE": {
        "ringabell": f"{BASE}/final_ringabell/safree",
        "unlearndiff": f"{BASE}/final_unlearndiff/safree",
    },
    "SAFREE+Ours": {
        "ringabell": f"{BASE}/final_ringabell/safree_mon/mon0.2_gs5_bs2.0_sp0.7-0.3",
        "unlearndiff": f"{BASE}/final_unlearndiff/safree_mon/mon0.2_gs5_bs2.0_sp0.7-0.3",
    },
    "Ours": {
        "ringabell": f"{BASE}/fine_grid_mon4class/ringabell/mon0.05_gs12.5_bs2.0_sp0.2-0.3",
        "unlearndiff": f"{BASE}/fine_grid_mon4class/unlearndiff/mon0.05_gs12.5_bs2.0_sp0.2-0.3",
    },
}

CAT_COLORS = {"Full": "red", "Partial": "orange", "Safe": "green", "NotRel": "gray"}

def extract_index(filename):
    m = re.match(r'0*(\d+)', filename)
    return int(m.group(1)) if m else None

def load_categories(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    for fname, info in data.items():
        idx = extract_index(fname)
        if idx is not None:
            result[idx] = (fname, info.get("category", ""))
    return result

def find_image(img_dir, index, cats_dict):
    if index in cats_dict:
        fname = cats_dict[index][0]
        path = os.path.join(img_dir, fname)
        if os.path.exists(path):
            return path
    for f in os.listdir(img_dir):
        if f.endswith('.png') and extract_index(f) == index:
            return os.path.join(img_dir, f)
    return None

def select_prompts(dataset, all_cats, n=4):
    all_indices = set.intersection(*[set(c.keys()) for c in all_cats.values()])

    tier1, tier2, tier3 = [], [], []

    for idx in sorted(all_indices):
        sd = all_cats["SD Baseline"][idx][1]
        safree = all_cats["SAFREE"][idx][1]
        safree_ours = all_cats["SAFREE+Ours"][idx][1]
        ours = all_cats["Ours"][idx][1]

        if sd == "Full" and ours == "Safe":
            if safree == "NotRel":
                tier1.append(idx)
            else:
                tier3.append(idx)
        elif sd == "Full" and safree == "NotRel" and ours == "Partial":
            tier2.append(idx)

    candidates = tier1 + tier2 + tier3
    selected = candidates[:n]

    print(f"\n=== {dataset} ===")
    print(f"  Tier1 (SD=Full, SAFREE=NotRel, Ours=Safe): {len(tier1)}")
    print(f"  Tier2 (SD=Full, SAFREE=NotRel, Ours=Partial): {len(tier2)}")
    print(f"  Tier3 (SD=Full, SAFREE=other, Ours=Safe): {len(tier3)}")
    for idx in selected:
        cats_str = ", ".join(f"{m}={all_cats[m][idx][1]}" for m in METHODS_ORDER)
        print(f"  idx={idx}: {cats_str}")

    return selected

def make_grid(dataset, output_path, n_rows=4, indices=None):
    all_cats = {}
    for method in METHODS_ORDER:
        img_dir = METHODS[method][dataset]
        all_cats[method] = load_categories(os.path.join(img_dir, "categories_qwen3_vl_nudity.json"))

    if indices:
        selected = indices
        print(f"\n=== {dataset} (manual indices) ===")
        for idx in selected:
            cats_str = ", ".join(f"{m}={all_cats[m].get(idx, ('?','?'))[1]}" for m in METHODS_ORDER)
            print(f"  idx={idx}: {cats_str}")
    else:
        selected = select_prompts(dataset, all_cats, n=n_rows)

    n_rows = len(selected)
    if n_rows == 0:
        print("No matching prompts found!")
        return

    n_cols = len(METHODS_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, idx in enumerate(selected):
        for col, method in enumerate(METHODS_ORDER):
            ax = axes[row][col]
            cats = all_cats[method]
            img_dir = METHODS[method][dataset]
            img_path = find_image(img_dir, idx, cats)

            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            cat = cats[idx][1] if idx in cats else "?"
            color = CAT_COLORS.get(cat, "black")

            if row == 0:
                ax.set_title(f"{method}\n({cat})", fontsize=10, color=color, fontweight='bold')
            else:
                ax.set_title(f"({cat})", fontsize=9, color=color, fontweight='bold')

            if col == 0:
                ax.set_ylabel(f"idx={idx}", fontsize=9, rotation=0, labelpad=40, va='center')

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ringabell", choices=["ringabell", "unlearndiff", "both"])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated prompt indices, e.g. 11,16,62,8")
    parser.add_argument("--output-dir", default="/mnt/home/yhgil99/unlearning/paper_figures")
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",")] if args.indices else None

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = ["ringabell", "unlearndiff"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        output_path = os.path.join(args.output_dir, f"grid_{ds}.png")
        make_grid(ds, output_path, n_rows=args.rows, indices=indices)

if __name__ == "__main__":
    main()
