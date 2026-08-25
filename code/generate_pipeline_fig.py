"""Generate P1-6: business-process embedding pipeline figure (Fig 5).

Usage: python code/generate_pipeline_fig.py

Produces:
  analysis/master_pipeline.pdf  (and copy to paper/figures/fig5_pipeline.pdf)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
FIG_DIR = ROOT / "figures"
ANALYSIS.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style — consistent with regenerate_figures.py
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

# Colors — muted, colorblind-safe, print-safe
C_KG = "#E8EDF3"  # light steel
C_SCREEN = "#2C7FB8"  # calibrated blue (existing palette)
C_SCREEN_LIGHT = "#D6EAF8"
C_EXPERT = "#F5E6CC"  # warm sand
C_ACTION = "#E2E8D5"  # sage
C_KEEP = "#2CA02C"
C_WITHHOLD = "#7F7F7F"
C_AUDIT = "#E74C3C"

# Figure — wide pipeline, ~text-width
fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")

# --- Helpers ---
def box(x, y, w, h, face, edgecolor="#333333", lw=0.8, radius=0.12, alpha=1.0, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edgecolor, linewidth=lw, alpha=alpha, zorder=z,
    )
    ax.add_patch(p)
    return p

def arrow(x1, y1, x2, y2, color="#333333", lw=1.4, style="->", ls="-", z=3):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw, linestyle=ls,
        mutation_scale=10, zorder=z, shrinkA=1, shrinkB=1,
    )
    ax.add_patch(a)
    return a

def txt(x, y, s, size=7, weight="normal", color="#1a1a1a", ha="center", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)

# --- 1) KG Completion Service ---
box(0.15, 0.85, 2.05, 1.55, C_KG, edgecolor="#5D6D7E")
txt(1.18, 2.08, "KG Completion", size=7.5, weight="bold")
txt(1.18, 1.88, "Service", size=7.5, weight="bold")
txt(1.18, 1.60, "WN18RR  ·  FB15k-237", size=5.8, color="#2C3E50")
txt(1.18, 1.42, "Hetionet core", size=5.8, color="#2C3E50")
txt(1.18, 1.18, "scored candidates", size=5.5, color="#34495E", style="italic")
# small badge
box(0.35, 0.97, 1.65, 0.22, "#FFFFFF", edgecolor="#AAB7C4", lw=0.5, radius=0.06)
txt(1.18, 1.08, "any KGE  (TransE / RotatE ...)", size=5.2, color="#5D6D7E")

# --- 2) Reliability Screen (highlighted) ---
# outer glow / highlight
box(2.55, 0.45, 2.45, 2.30, "#EBF5FB", edgecolor=C_SCREEN, lw=1.3, radius=0.14, alpha=0.95)
# header bar
box(2.55, 2.30, 2.45, 0.45, C_SCREEN, edgecolor=C_SCREEN, lw=0, radius=0.14)
# fix bottom corners of header (overlay white strip)
ax.add_patch(plt.Rectangle((2.55, 2.30), 2.45, 0.12, facecolor=C_SCREEN, edgecolor="none", zorder=4))
txt(3.78, 2.52, "RELIABILITY  SCREEN  ★", size=6.8, weight="bold", color="white")
txt(3.78, 2.34, "this paper", size=5.2, color="white", alpha=0.92)

txt(3.78, 1.95, "rank-extremeness  p(h,r,t)", size=6.0, weight="bold", color="#1A3C5E")
txt(3.78, 1.73, "K >= 500  shared-reference", size=5.3, color="#2C3E50")
# three rules
for i, (label, sub) in enumerate([
    ("nominal BH", "a = 0.05"),
    ("calibrated", "g = 0.10 -> 0.086 realized"),
    ("cost-aware", "1:1 -> 0.055 vs 0.100"),
]):
    y = 1.45 - i * 0.30
    col = C_SCREEN if i == 1 else ("#2471A3" if i == 0 else C_KEEP)
    ax.add_patch(plt.Circle((2.82, y), 0.05, color=col, zorder=5))
    txt(2.95, y, label, size=5.4, weight="bold", ha="left", va="center", color="#1A1A1A")
    txt(4.90, y, sub, size=4.7, ha="right", va="center", color="#2C3E50")
# keep/withhold split at bottom of screen box — lowered to avoid overlap
box(2.75, 0.50, 0.96, 0.26, "#FFFFFF", edgecolor=C_KEEP, lw=0.9, radius=0.07)
txt(3.23, 0.63, "KEEP", size=6, weight="bold", color=C_KEEP)
box(3.85, 0.50, 0.96, 0.26, "#FFFFFF", edgecolor=C_WITHHOLD, lw=0.7, radius=0.07, alpha=0.95)
txt(4.33, 0.63, "WITHHOLD", size=5.4, weight="bold", color=C_WITHHOLD)

# --- 3) Expert System Decision Layer ---
box(5.35, 0.85, 2.05, 1.55, C_EXPERT, edgecolor="#8D6E63")
txt(6.38, 2.08, "Expert System", size=7.5, weight="bold")
txt(6.38, 1.88, "Decision Layer", size=7.5, weight="bold")
txt(6.38, 1.60, "drug-repurposing triage", size=5.8, color="#4E342E")
txt(6.38, 1.42, "clinical DSS", size=5.8, color="#4E342E")
txt(6.38, 1.18, "acts only on KEEP", size=5.5, color="#5D4037", style="italic")
box(5.55, 0.97, 1.65, 0.22, "#FFFFFF", edgecolor="#BCAAA4", lw=0.5, radius=0.06)
txt(6.38, 1.08, "measured FDR  0.086 @ 4.5%", size=5.2, color="#5D4037")

# --- 4) Downstream Action & Audit ---
box(7.75, 0.85, 2.05, 1.55, C_ACTION, edgecolor="#6B7B5E")
txt(8.78, 2.08, "Downstream", size=7.5, weight="bold")
txt(8.78, 1.88, "Action & Audit", size=7.5, weight="bold")
txt(8.78, 1.60, "wet-lab experiment", size=5.8, color="#33402B")
txt(8.78, 1.42, "treatment recommendation", size=5.8, color="#33402B")
txt(8.78, 1.18, "cost incurred  -  audit", size=5.5, color="#3E4A35", style="italic")
box(7.95, 0.97, 1.65, 0.22, "#FFFFFF", edgecolor="#A3B18A", lw=0.5, radius=0.06)
txt(8.78, 1.08, "realized vs claimed", size=5.2, color="#3E4A35")

# --- Arrows: main pipeline ---
# KG -> Screen
arrow(2.20, 1.62, 2.55, 1.62, color="#5D6D7E", lw=1.6)
txt(2.38, 1.78, "scores", size=5.0, color="#5D6D7E", style="italic")
# Screen KEEP -> Expert — from center of KEEP box to Expert box
arrow(3.23, 0.50, 5.35, 1.00, color=C_KEEP, lw=1.5)
# Withhold branch (dashed, down-out) — from center of WITHHOLD bottom edge
arrow(4.33, 0.50, 4.33, 0.28, color=C_WITHHOLD, lw=1.0, ls=(0, (3, 2)))
txt(4.33, 0.16, "withheld: no action", size=4.6, color=C_WITHHOLD, style="italic")
# Expert -> Action
arrow(7.40, 1.62, 7.75, 1.62, color="#8D6E63", lw=1.6)
txt(7.57, 1.78, "keep", size=5.0, color=C_KEEP, weight="bold")

# --- Audit feedback loop (curved, top) ---
# From Action back to Screen (audit)
arrow(8.78, 2.40, 3.78, 2.85, color=C_AUDIT, lw=1.1, ls=(0, (4, 2)), style="-|>")
txt(6.28, 2.95, "realized-vs-claimed FDR audit  (labeled pools, Wilson CI)", size=5.0, color=C_AUDIT, style="italic", weight="bold")
# small annotation near audit arrow
txt(6.28, 2.75, "honest error: nominal 0.033 at 0.05  -  calibrated 0.086 at 0.10  -  cost -45% at 1:1", size=4.6, color="#7B241C")

# --- Bottom caption bar ---
txt(5.0, 0.05, "Without the screen: raw top-k  FDR 0.42-0.67 at 10-20% coverage  ->  wasted experiments.   With the screen: keep only the audited subset.", size=5.2, color="#5D6D7E", style="italic")

fig.tight_layout(pad=0.6)
out = ANALYSIS / "master_pipeline.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}  ({out.stat().st_size} bytes)")
# vector + raster preview
png = ANALYSIS / "master_pipeline.png"
fig.savefig(png, bbox_inches="tight", dpi=220)
print(f"Saved {png}")

# Copy to the release figures directory with canonical name
dst = FIG_DIR / "fig5_pipeline.pdf"
shutil.copy(out, dst)
print(f"Copied -> {dst}")
dst_png = FIG_DIR / "fig5_pipeline.png"
shutil.copy(png, dst_png)
print(f"Copied -> {dst_png}")
