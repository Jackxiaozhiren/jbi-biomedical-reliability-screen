"""Regenerate the manuscript-facing pipeline figure from frozen release JSON.

The quantitative callouts are read from the repository's frozen result exports.
The layout matches the collision-free v5 manuscript figure. Running this script
writes ``analysis/master_pipeline.{pdf,png}`` and ``figures/fig5_pipeline.{pdf,png,svg}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"
FIG_DIR = ROOT / "figures"
ANALYSIS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


audit = load_json(RESULTS_DIR / "hetionet_audit_J9_K500.json")
cal10 = next(row for row in audit["calibrated"] if row["claimed_fdr"] == 0.10)
cost11 = next(row for row in audit["cost_aware"] if row["c_fp"] == 1 and row["c_fn"] == 1)
withhold_cost = audit["decision_cost"]["nominal_bh_0.05"][0]["expected_cost"]
cost_reduction = 100.0 * (1.0 - cost11["expected_cost"] / withhold_cost)
wn = load_json(RESULTS_DIR / "realized_fdr_WN18RR_RotatE_J9_K500.json")
fb = load_json(RESULTS_DIR / "realized_fdr_FB15k237_J9_K500.json")
nominal_fdr = min(wn["nominal_bh_0.05"]["realized_fdr"], fb["nominal_bh_0.05"]["realized_fdr"])
topk_10_20 = [
    row["realized_fdr"]
    for data in (wn, fb)
    for row in data["topk"]
    if row["coverage"] in (0.1, 0.2)
]
topk_range = (min(topk_10_20), max(topk_10_20))


plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "svg.fonttype": "none",
    }
)

C_KG = "#E8EDF3"
C_SCREEN = "#2C7FB8"
C_EXPERT = "#F5E6CC"
C_ACTION = "#E2E8D5"
C_KEEP = "#2CA02C"
C_WITHHOLD = "#7F7F7F"
C_AUDIT = "#E74C3C"

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.set_xlim(0.0, 10.0)
ax.set_ylim(0.0, 3.55)
ax.axis("off")


def box(x, y, w, h, face, edgecolor="#333333", lw=0.8, radius=0.12, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(x1, y1, x2, y2, color="#333333", lw=1.4, style="->", ls="-", z=3):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=lw,
        linestyle=ls,
        mutation_scale=10,
        zorder=z,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(patch)
    return patch


def path_arrow(vertices, color="#333333", lw=1.4, style="->", ls="-", z=3):
    path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1))
    patch = FancyArrowPatch(
        path=path,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        linestyle=ls,
        mutation_scale=10,
        zorder=z,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(patch)
    return patch


def txt(x, y, value, size=7, weight="normal", color="#1a1a1a", ha="center", va="center", z=5, **kw):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        clip_on=False,
        **kw,
    )


# 1) KG completion service
box(0.25, 0.98, 2.05, 1.58, C_KG, edgecolor="#5D6D7E")
txt(1.275, 2.27, "KG Completion", size=7.5, weight="bold")
txt(1.275, 2.07, "Service", size=7.5, weight="bold")
txt(1.275, 1.78, "WN18RR  ·  FB15k-237", size=5.8, color="#2C3E50")
txt(1.275, 1.60, "Hetionet core", size=5.8, color="#2C3E50")
txt(1.275, 1.39, "scored candidates", size=5.2, color="#34495E", style="italic")
box(0.45, 1.04, 1.65, 0.28, "#FFFFFF", edgecolor="#AAB7C4", lw=0.5, radius=0.06)
txt(1.275, 1.23, "compatible KGE scorer", size=4.4, color="#5D6D7E")
txt(1.275, 1.11, "tested: TransE / RotatE / ComplEx", size=3.9, color="#5D6D7E")

# 2) Reliability screen
box(2.62, 0.57, 2.58, 2.30, "#EBF5FB", edgecolor=C_SCREEN, lw=1.3, radius=0.14)
box(2.62, 2.43, 2.58, 0.44, C_SCREEN, edgecolor=C_SCREEN, lw=0, radius=0.14)
ax.add_patch(plt.Rectangle((2.62, 2.43), 2.58, 0.12, facecolor=C_SCREEN, edgecolor="none", zorder=4))
txt(3.91, 2.65, "RELIABILITY  SCREEN  ★", size=6.8, weight="bold", color="white")
txt(3.91, 2.49, "this paper", size=5.2, color="white", alpha=0.92)
txt(3.91, 2.12, "rank-extremeness  p(h,r,t)", size=6.0, weight="bold", color="#1A3C5E")
txt(3.91, 1.90, "K >= 500  shared-reference", size=5.2, color="#2C3E50")

for i, (label, sub) in enumerate(
    [
        ("nominal BH", "alpha = 0.05"),
        ("calibrated", f"gamma = 0.10 -> {cal10['realized_fdr']:.3f} realized"),
        ("cost-aware", f"1:1 -> {cost_reduction:.0f}% lower cost"),
    ]
):
    y = 1.60 - i * 0.30
    col = "#2471A3" if i == 0 else (C_SCREEN if i == 1 else C_KEEP)
    ax.add_patch(plt.Circle((2.91, y), 0.055, color=col, zorder=5))
    txt(3.05, y, label, size=5.25, weight="bold", ha="left", color="#1A1A1A")
    txt(3.72, y, sub, size=4.05, ha="left", color="#2C3E50")

box(2.84, 0.62, 1.04, 0.27, "#FFFFFF", edgecolor=C_KEEP, lw=0.9, radius=0.07)
txt(3.36, 0.755, "KEEP", size=6, weight="bold", color=C_KEEP)
box(4.08, 0.62, 1.04, 0.27, "#FFFFFF", edgecolor=C_WITHHOLD, lw=0.7, radius=0.07)
txt(4.60, 0.755, "WITHHOLD", size=5.3, weight="bold", color=C_WITHHOLD)

# 3) Triage decision layer
box(5.55, 0.98, 2.05, 1.58, C_EXPERT, edgecolor="#8D6E63")
txt(6.575, 2.27, "Triage", size=7.5, weight="bold")
txt(6.575, 2.07, "Decision Layer", size=7.5, weight="bold")
txt(6.575, 1.78, "drug-repurposing pipeline", size=5.8, color="#4E342E")
txt(6.575, 1.60, "follow-up prioritization", size=5.8, color="#4E342E")
txt(6.575, 1.38, "acts only on KEEP", size=5.2, color="#5D4037", style="italic")
box(5.75, 1.04, 1.65, 0.28, "#FFFFFF", edgecolor="#BCAAA4", lw=0.5, radius=0.06)
txt(6.575, 1.20, f"measured FDR  {cal10['realized_fdr']:.3f} @ {cal10['coverage']:.1%}", size=4.5, color="#5D4037")

# 4) Downstream action and audit
box(7.95, 0.98, 2.05, 1.58, C_ACTION, edgecolor="#6B7B5E")
txt(8.975, 2.27, "Downstream", size=7.5, weight="bold")
txt(8.975, 2.07, "Action & Audit", size=7.5, weight="bold")
txt(8.975, 1.78, "wet-lab experiment", size=5.8, color="#33402B")
txt(8.975, 1.60, "candidate prioritization", size=5.8, color="#33402B")
txt(8.975, 1.38, "cost incurred  -  audit", size=5.2, color="#3E4A35", style="italic")
box(8.15, 1.04, 1.65, 0.28, "#FFFFFF", edgecolor="#A3B18A", lw=0.5, radius=0.06)
txt(8.975, 1.20, "realized vs claimed", size=4.5, color="#3E4A35")

# Main flow arrows and labels
arrow(2.30, 1.78, 2.62, 1.78, color="#5D6D7E", lw=1.6)
txt(2.46, 1.94, "scores", size=4.7, color="#5D6D7E", style="italic")
arrow(3.36, 0.62, 5.55, 1.22, color=C_KEEP, lw=1.5)
arrow(4.60, 0.62, 4.60, 0.34, color=C_WITHHOLD, lw=1.0, ls=(0, (3, 2)))
txt(4.60, 0.22, "withheld: no action", size=4.5, color=C_WITHHOLD, style="italic")
arrow(7.60, 1.78, 7.95, 1.78, color="#8D6E63", lw=1.6)
txt(7.78, 1.94, "keep", size=4.8, color=C_KEEP, weight="bold")

# Audit feedback loop: route above the boxes and terminate at the top-right
# edge of the Reliability Screen, making the feedback destination explicit.
path_arrow(
    [(8.98, 2.56), (8.98, 3.08), (4.95, 3.08), (4.95, 2.87)],
    color=C_AUDIT,
    lw=1.1,
    ls=(0, (4, 2)),
    style="-|>",
)
txt(6.50, 3.42, "realized-vs-claimed FDR audit  (labeled pools, Wilson CI)", size=5.0, color=C_AUDIT, style="italic", weight="bold")
txt(6.50, 3.25, f"honest error: nominal {nominal_fdr:.3f} at 0.05  -  calibrated {cal10['realized_fdr']:.3f} at 0.10  -  cost -{cost_reduction:.0f}% at 1:1", size=4.25, color="#7B241C")

txt(5.0, 0.08, f"Without the screen: raw top-k  FDR {topk_range[0]:.2f}-{topk_range[1]:.2f} at 10-20% coverage  ->  wasted experiments.   With the screen: keep only the audited subset.", size=4.55, color="#5D6D7E", style="italic")

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
out = ANALYSIS_DIR / "master_pipeline.pdf"
png = ANALYSIS_DIR / "master_pipeline.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
fig.savefig(png, bbox_inches="tight", pad_inches=0.03, dpi=220)
fig.savefig(FIG_DIR / "fig5_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(FIG_DIR / "fig5_pipeline.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
fig.savefig(FIG_DIR / "fig5_pipeline.svg", bbox_inches="tight", pad_inches=0.03)
print(f"Saved {out} ({out.stat().st_size} bytes)")
print(f"Saved {png} ({png.stat().st_size} bytes)")
