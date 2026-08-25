#!/usr/bin/env python3
"""E1 external-validation enrichment figure (Phase 3, JBI submission).

Data figure per academic-plotting Workflow 2: two-panel horizontal bar chart
of external-evidence enrichment lift (log axis) for the CtD (drug-disease,
ClinicalTrials.gov) and CbG (drug-gene, ChEMBL) decision edges, across the
frozen primary operating point and the pre-registered sensitivity cells.

All numbers are read programmatically from results/external_validation_*.json
(no hand-copied values). Low-power cells (protocol §8: <50 evidence hits)
are hatched.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
})

CTD = json.loads((ROOT / "results" / "external_validation_ctd.json").read_text())
CBG = json.loads((ROOT / "results" / "external_validation_cbg.json").read_text())

# (json key, display label, kind)  kind: primary | sensitivity | lowpower
ctd_cells = [
    ("full|main|full", "Main operating point\n(calibrated target 0.10)", "primary"),
    ("full|main|W1", "Window 2017\u20132020", "sens"),
    ("full|main|W2", "Window 2021\u20132026", "sens"),
    ("full|t20|full", "Threshold \u03c4 = 0.20", "sens"),
    ("full|relcond|full", "Per-relation threshold", "sens"),
    ("full|cost11|full", "Cost-aware 1:1", "sens"),
    ("restricted|main|full", "Alignment-restricted space", "sens"),
]
cbg_cells = [
    ("full|main|tier1_full", "Main op. point, Tier-1 (≤10 nM)", "primary"),
    ("full|main|full", "Main op. point, ≤100 nM", "sens"),
    ("full|main|tier1_W1", "Tier-1, window 2018–2020", "lowpower"),
    ("full|main|tier1_W2", "Tier-1, window 2021–2026", "sens"),
    ("full|t20|tier1_full", "Threshold τ = 0.20, Tier-1", "sens"),
    ("full|relcond|tier1_full", "Per-relation, Tier-1", "sens"),
    ("restricted|main|tier1_full", "Alignment-restricted, Tier-1", "sens"),
]

PRIMARY = "#D55E00"   # vermillion (Okabe-Ito) - the paper's operating point
SENS = "#0072B2"      # blue (Okabe-Ito) - sensitivity cells
LOWP = "#8C8C8C"      # gray - low-power cells (hatched)


def panel(ax, data, cells, title):
    labels, lifts, colors, hatches, anns = [], [], [], [], []
    for key, label, kind in cells:
        c = data[key]
        lift = c["lift"]
        lift = lift if lift == lift and lift not in (float("inf"),) else np.nan
        low = bool(c.get("low_power"))
        labels.append(label)
        lifts.append(lift)
        colors.append(PRIMARY if kind == "primary" else (LOWP if low else SENS))
        hatches.append("///" if low else None)
        p = c.get("fisher_p_one_sided")
        p_s = f"p={p:.0e}" if p is not None else ""
        anns.append(f"{c['keep_hits']}/{c['withhold_hits']} hits  {p_s}")
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, lifts, height=0.62, color=colors, edgecolor="white",
                   linewidth=0.5)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#444", lw=1.0, ls="--", zorder=0)
    ax.text(1.0, y[-1] - 0.75, "no enrichment (lift=1)", fontsize=7.5,
            color="#444", ha="center", va="top")
    for yi, lv, ann in zip(y, lifts, anns):
        ax.text(lv * 1.12, yi, ann, va="center", fontsize=7.2, color="#333")
    ax.set_xlabel("Enrichment lift  (hit rate KEEP / hit rate WITHHOLD)")
    ax.set_title(title, loc="left")
    ax.set_xlim(0.8, max(lifts) * 12)


fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.4))
panel(axes[0], CTD, ctd_cells,
      "A  Drug\u2013disease (CtD): post-2017 trials")
panel(axes[1], CBG, cbg_cells,
      "B  Drug\u2013gene (CbG): post-2017 potent binding")
fig.tight_layout(w_pad=2.4)
fig.savefig(FIGS / "fig_enrichment.pdf")
fig.savefig(FIGS / "fig_enrichment.png", dpi=300)
print("saved:", FIGS / "fig_enrichment.pdf", "and .png")
