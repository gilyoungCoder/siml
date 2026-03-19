import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NAMES = {0: "not people", 1: "non-nude people", 2: "nude people"}

def load_props(csv_path, min_step=None, max_step=None):
    df = pd.read_csv(csv_path)
    df["step"] = df["step"].astype(int)
    # step x class 카운트 → 비율
    counts = df.groupby(["step", "pred_class"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[0, 1, 2], fill_value=0)
    if min_step is None: min_step = int(counts.index.min())
    if max_step is None: max_step = int(counts.index.max())
    counts = counts.reindex(range(min_step, max_step + 1), fill_value=0)
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return props

def plot_stacked_bars(props, out_png, title="Classifier breakdown by step No Guidance",
                      step_stride=1, bar_width=0.9):
    # 너무 빼곡하면 간격 두고 샘플링
    props = props.iloc[::step_stride].copy()
    x = np.arange(len(props.index))
    labels = props.index.astype(int).tolist()

    y0 = props[0].values
    y1 = props[1].values
    y2 = props[2].values

    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)

    plt.figure(figsize=(14, 6))
    b0 = plt.bar(x, y0, width=bar_width, label=NAMES[0])
    b1 = plt.bar(x, y1, width=bar_width, bottom=y0, label=NAMES[1])
    b2 = plt.bar(x, y2, width=bar_width, bottom=y0 + y1, label=NAMES[2])

    plt.ylim(0, 1)
    plt.xlim(-0.5, len(x) - 0.5)
    plt.xticks(x, labels, rotation=0)
    plt.xlabel("step")
    plt.ylabel("proportion")
    plt.title(title)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("[Saved]", os.path.abspath(out_png))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./no_guidance_cls/cls_dump.csv",
                    help="Path to cls_dump.csv")
    ap.add_argument("--out", default="./no_guidance_cls/cls_breakdown_by_step.png",
                    help="Output PNG path")
    ap.add_argument("--min-step", type=int, default=None)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--step-stride", type=int, default=2,
                    help="plot every Nth step (e.g., 2 or 5 to reduce clutter)")
    args = ap.parse_args()

    props = load_props(args.csv, args.min_step, args.max_step)
    plot_stacked_bars(props, args.out, step_stride=args.step_stride)

if __name__ == "__main__":
    main()
