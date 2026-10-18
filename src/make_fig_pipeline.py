"""Figure 1: study pipeline diagram."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paths import RESULTS
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(10, 3.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333", lw=1))
    ax.text(x + w/2, y + h - 3, title, ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.text(x + w/2, y + h - 8, body, ha="center", va="top", fontsize=7.2, linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#333"))
c = ["#e8eef4", "#eef4e8", "#f6efe3", "#f4e8ee", "#e8f2f2"]
box(1, 21, 17, 16, "1  M5 panel", "3,999 stratified\nitem-store series\nTRAIN | VAL | TEST\n(1,465 | 364 | 140 d)", c[0])
box(21, 21, 18, 16, "2  Forecasters", "9 point models\n1 trained quantile model\n3 calibrated empirical\nvariants", c[1])
box(42, 21, 18, 16, "3  Policies", "P1 normal base stock\nP2 quantile base stock\nP3 (s,S) weekly/daily\nP4 DDMRP buffer", c[2])
box(63, 21, 17, 16, "4  Simulator", "lost sales, lead time 2 d\nholding, margin penalty,\nordering cost\n59 pairings", c[3])
box(83, 21, 16, 16, "5  Outputs", "cost, fill rate,\non-hand per series\nand per period", c[4])
for xa, xb in [(18, 21), (39, 42), (60, 63), (80, 83)]: arrow(xa+0.6, 29, xb-0.6, 29)
box(50, 2, 31, 14, "6  Selection on VAL only", "accuracy-driven vs cost-driven choice;\nper-SKU, menu, shrinkage, meta-model;\ntuned normal-theory rule", c[0])
box(8, 2, 31, 14, "7  Evaluation on TEST", "paired bootstrap over series (2,000);\ndecomposition A to E; regimes;\nsensitivity; second sample draw", c[1])
arrow(91, 20.3, 91, 9.4); ax.plot([91, 81.6], [9, 9], color="#333", lw=1.1)
arrow(81.8, 9, 81.3, 9); arrow(49.4, 9, 40, 9)
fig.savefig(RESULTS / "fig0_pipeline.png"); print("ok")
