"""
EBSG How-mode trade-off: compute-per-step vs anchor-cleanliness.
Single 2D scatter-with-annotations chart for NeurIPS 2026 paper.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

OUT_DIR = "/mnt/c/Users/yhgil/paper_figures_out/tradeoff"

# ---- palette (colorblind-safe) ----
BLUE = "#1f77b4"   # anchor inpaint
RED = "#d62728"    # hybrid
GOLD = "#d4a017"   # per-concept: nudity
GREEN = "#2ca02c"  # per-concept: violence/illegal/shocking

# ---- figure / axes ----
fig, ax = plt.subplots(figsize=(6.0, 4.5))
ax.set_xlim(0.5, 2.5)
ax.set_ylim(0.0, 1.0)

# ---- subtle vertical color gradient in plot area (orange bottom -> green top) ----
# Use imshow of a 2D array with a custom colormap, restricted to axes limits.
from matplotlib.colors import LinearSegmentedColormap
grad_cmap = LinearSegmentedColormap.from_list(
    "orange_green", ["#ffd8b0", "#c9e8c9"]
)
gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
ax.imshow(
    gradient,
    aspect="auto",
    cmap=grad_cmap,
    extent=(0.5, 2.5, 0.0, 1.0),
    origin="lower",
    alpha=0.15,
    zorder=0,
)

# ---- minor grid ----
ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6, zorder=1)
ax.grid(True, which="minor", color="#e8e8e8", linewidth=0.4, zorder=1)
ax.minorticks_on()
ax.set_axisbelow(True)

# ---- Pareto frontier (dashed diagonal) ----
ax.plot(
    [0.7, 2.3], [0.1, 0.95],
    linestyle="--", color="#555555", linewidth=1.0, zorder=2,
)
# Italic label along the line (place near midpoint, slight offset above)
ax.text(
    1.55, 0.60,
    "compute $\\times$ cleanliness Pareto frontier",
    fontsize=8.5, style="italic", color="#444444",
    rotation=27, rotation_mode="anchor",
    ha="center", va="bottom", zorder=3,
)

# ---- per-concept small dots (drawn first so main markers sit on top) ----
concepts = [
    ("nudity $\\rightarrow$ hybrid",    2.0, 0.88, GOLD,  "*", (2.0, 0.85)),  # connect to hybrid
    ("violence $\\rightarrow$ anchor",  1.0, 0.25, GREEN, "^", (1.0, 0.30)),
    ("illegal $\\rightarrow$ anchor",   1.0, 0.32, GREEN, "^", (1.0, 0.30)),
    ("shocking $\\rightarrow$ anchor",  1.0, 0.28, GREEN, "^", (1.0, 0.30)),
]
for label, x, y, color, marker, (mx, my) in concepts:
    # thin dotted connector to nearest mode marker
    ax.plot([x, mx], [y, my], linestyle=":", color="#888888",
            linewidth=0.7, alpha=0.7, zorder=3)
    ax.scatter([x], [y], s=60, c=color, marker=marker,
               alpha=0.6, edgecolors="black", linewidths=0.4, zorder=5)

# Only the nudity (hybrid-side) concept gets its own inline label.
ax.text(2.0 + 0.07, 0.88 + 0.04, "nudity $\\rightarrow$ hybrid",
        fontsize=7.0, color="#333333", ha="left", va="center", zorder=5)

# Consolidated anchor-side callout: one leader line from the cluster
# centroid (1.0, 0.28) out to the left into open whitespace, ending in
# a single stacked text box listing the three concepts.
anchor_group_text = "violence\nillegal   $\\rightarrow$ anchor\nshocking"
ax.annotate(
    anchor_group_text,
    xy=(1.0, 0.28), xycoords="data",
    xytext=(0.55, 0.45), textcoords="data",
    fontsize=7.5, color="#333333",
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=GREEN, lw=0.7, alpha=0.95),
    arrowprops=dict(arrowstyle="-", color="#888888",
                    lw=0.7, linestyle=":",
                    shrinkA=2, shrinkB=2),
    zorder=5,
)

# ---- main markers ----
ax.scatter([1.0], [0.30], s=200, c=BLUE, marker="o",
           edgecolors="black", linewidths=0.8, zorder=6,
           label="anchor-inpaint")
ax.scatter([2.0], [0.85], s=200, c=RED, marker="s",
           edgecolors="black", linewidths=0.8, zorder=6,
           label="hybrid")

# small inline labels near main markers
ax.text(1.0, 0.30 - 0.075, "anchor inpaint", fontsize=8.5,
        color=BLUE, ha="center", va="top", fontweight="bold", zorder=6)
ax.text(1.88, 0.85, "hybrid", fontsize=8.5,
        color=RED, ha="right", va="center", fontweight="bold", zorder=6)

# ---- callout boxes with leader lines ----
# Anchor-inpaint callout: upper-right of the blue marker.
# Nudged slightly right (1.65 -> 1.72) so it does not collide with the
# new consolidated anchor-side leader that heads to (0.55, 0.45).
anchor_callout_xy = (1.72, 0.58)
anchor_text = "$s_a \\in [0.5, 3.0]$\n1$\\times$ compute\nrobust when anchor is ambiguous"
ax.annotate(
    anchor_text,
    xy=(1.0, 0.30), xycoords="data",
    xytext=anchor_callout_xy, textcoords="data",
    fontsize=8,
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.35", fc="white",
              ec=BLUE, lw=0.8, alpha=0.95),
    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7,
                    shrinkA=0, shrinkB=2),
    zorder=7,
)

# Hybrid callout: lower-right of the red marker
hybrid_callout_xy = (2.12, 0.45)
hybrid_text = "$s_h \\in [10, 20]$\n2$\\times$ compute\nwins when anchor is crisp"
ax.annotate(
    hybrid_text,
    xy=(2.0, 0.85), xycoords="data",
    xytext=hybrid_callout_xy, textcoords="data",
    fontsize=8,
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.35", fc="white",
              ec=RED, lw=0.8, alpha=0.95),
    arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                    shrinkA=0, shrinkB=2),
    zorder=7,
)

# ---- axis ticks & labels ----
ax.set_xticks([1.0, 2.0])
ax.set_xticklabels(["1$\\times$", "2$\\times$"])
ax.set_yticks([0.2, 0.8])
ax.set_yticklabels(["ambiguous", "crisp"])

ax.set_xlabel("Compute per step (UNet forwards)")
ax.set_ylabel("Anchor cleanliness")
ax.set_title("How-mode trade-off: compute vs anchor cleanliness",
             fontsize=11)

# ---- legend (top-right) ----
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
           markeredgecolor="black", markersize=10, label="anchor-inpaint"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=RED,
           markeredgecolor="black", markersize=10, label="hybrid"),
]
ax.legend(handles=legend_handles, loc="upper left",
          framealpha=0.95, fontsize=9, borderpad=0.5)

# ---- white background, tidy spines ----
fig.patch.set_facecolor("white")
for sp in ax.spines.values():
    sp.set_color("#888888")
    sp.set_linewidth(0.8)

plt.tight_layout()

# ---- dual save: paper artifact + rubric preview ----
plt.savefig(f"{OUT_DIR}/figure.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/preview.png", dpi=200, bbox_inches="tight")
