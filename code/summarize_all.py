"""Master results report: consolidates all experiment outputs into one digest.

Scans results/ for spike_*, realized_fdr_*, and hetionet_audit_*.json and
renders a single paper-ready markdown report plus figures:

  * K-resolution analysis (floor-saturation and BH behavior vs K) from spike
    jsons -- the paper's "does candidate resolution matter" figure;
  * per-audit claimed-vs-realized / method-comparison / decision-cost tables;
  * per-audit figures (reused from analyze_results).

Usage: python code/summarize_all.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
ANALYSIS.mkdir(exist_ok=True)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_results as ar


def k_resolution_table() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "spike_*.json"))):
        d = json.load(open(f))
        rows.append({
            "K": d.get("k"), "m": d.get("m"), "floor_fraction": d.get("n_at_floor") / d.get("m", 1),
            "pi0": d.get("pi0"), "bh_rejected_rate": d.get("rejected_rate"),
            "bh_threshold": d.get("bh_threshold"), "scores_per_sec": d.get("scores_per_sec"),
        })
    return sorted(rows, key=lambda r: (r["K"] or 0))


def render_k_resolution(rows: list[dict], out_prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    Ks = [r["K"] for r in rows]
    ax.plot(Ks, [r["floor_fraction"] for r in rows], marker="o", label="floor fraction")
    ax2 = ax.twinx()
    ax2.plot(Ks, [r["bh_rejected_rate"] for r in rows], marker="s", color="#c0392b",
             label="BH rejected rate")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("candidates K (log)")
    ax.set_ylabel("fraction at floor p", color="#2c7fb8")
    ax2.set_ylabel("BH rejected rate", color="#c0392b")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=7)
    fig.tight_layout()
    p = ANALYSIS / f"{out_prefix}_k_resolution.pdf"
    fig.savefig(p); print(f"Saved {p}")


def main():
    parts = ["# Master results report", ""]

    spike = k_resolution_table()
    if spike:
        parts.append("## K-resolution (WN18RR TransE spike)")
        parts.append("| K | m | floor fraction | pi0 | BH rejected rate | BH threshold | scores/s |")
        parts.append("|---|---|---|---|---|---|---|")
        for r in spike:
            parts.append(f"| {r['K']} | {r['m']} | {r['floor_fraction']:.3f} | "
                         f"{r['pi0']:.3f} | {r['bh_rejected_rate']:.3f} | "
                         f"{r['bh_threshold']:.4f} | {r['scores_per_sec']:.0f} |")
        parts.append("")
        render_k_resolution(spike, "master")

    audits = sorted(glob.glob(str(ROOT / "results" / "hetionet_audit_*.json")))
    for f in audits:
        a = json.load(open(f))
        prefix = Path(f).stem.replace("hetionet_audit_", "")
        parts.append(f"\n## Hetionet audit ({prefix})")
        tbl = ar.claimed_vs_realized_table(a)
        parts.append("\n### Claimed vs realized FDR")
        parts.append("| method | claimed | realized | precision | recall | coverage |")
        parts.append("|---|---|---|---|---|---|")
        for r in tbl:
            parts.append(f"| {r['method']} | {r['claimed']} | {r['realized_fdr']:.4f} | "
                         f"{r['precision']:.4f} | {r['recall']:.4f} | {r['coverage']:.1%} |")
        ar.render_figures(a, "master_" + prefix)

    for f in sorted(glob.glob(str(ROOT / "results" / "realized_fdr_*.json"))):
        if "hetionet" in Path(f).name:
            continue
        a = json.load(open(f))
        prefix = Path(f).stem
        parts.append(f"\n## Realized-FDR ({prefix})")
        tbl = ar.claimed_vs_realized_table(a)
        parts.append("| method | claimed | realized | precision | recall | coverage |")
        parts.append("|---|---|---|---|---|---|")
        for r in tbl:
            parts.append(f"| {r['method']} | {r['claimed']} | {r['realized_fdr']:.4f} | "
                         f"{r['precision']:.4f} | {r['recall']:.4f} | {r['coverage']:.1%} |")

    out = ANALYSIS / "MASTER_REPORT.md"
    out.write_text("\n".join(parts))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
