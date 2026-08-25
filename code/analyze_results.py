"""Aggregate experiment results into paper-ready tables and figures (P5).

Reads the audit / realized-FDR / spike JSONs produced by the experiment
scripts and renders:
  * a claimed-vs-realized FDR comparison table (the core audit finding);
  * a cross-method comparison at comparable operating points;
  * the decision-cost table across cost ratios;
  * figures: realized-vs-claimed FDR, realized-FDR-vs-recall by method, and
    decision-cost vs cost-ratio curves.

Usage: python code/analyze_results.py [--audit results/hetionet_audit_J9_K500.json ...]
  (defaults to globbing the newest audit + realized_fdr + spike jsons)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
ANALYSIS.mkdir(exist_ok=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def claimed_vs_realized_table(audit: dict) -> list[dict]:
    """Rows of (method, claimed_fdr, realized_fdr, precision, recall, coverage)."""
    rows = []
    # nominal BH
    if "nominal_bh_0.05" in audit:
        r = audit["nominal_bh_0.05"]
        rows.append({"method": "nominal BH", "claimed": 0.05,
                     "realized_fdr": r["realized_fdr"], "precision": r["precision"],
                     "recall": r["recall"], "coverage": r["coverage"]})
    # calibrated targets
    for row in audit.get("calibrated", []):
        rows.append({"method": "calibrated", "claimed": row["claimed_fdr"],
                     "realized_fdr": row["realized_fdr"], "precision": row["precision"],
                     "recall": row["recall"], "coverage": row["coverage"]})
    # cost-aware
    for row in audit.get("cost_aware", []):
        rows.append({"method": f"cost-aware {row['c_fp']}:{row['c_fn']}",
                     "claimed": None, "realized_fdr": row["realized_fdr"],
                     "precision": row["precision"], "recall": row["recall"],
                     "coverage": row["coverage"]})
    return rows


def method_comparison(audit: dict) -> list[dict]:
    """Best (lowest-FDR) operating point per method at coverage >= 5%.

    Works for both the full audit schema and the older realized_fdr_*.json
    schema (which lacks calibrator/conformal/relik/cost sections).
    """
    candidates = []
    for row in audit.get("calibrated", []):
        if row.get("coverage", 0) > 0.0:
            candidates.append({"method": "calibrated", **row})
    for row in audit.get("topk", []):
        if row.get("coverage", 0) >= 0.05:
            candidates.append({"method": "raw top-k", **row})
    for row in (audit.get("calibrator", {}).get("platt", [])
                + audit.get("calibrator", {}).get("isotonic", [])):
        if row.get("coverage", 0) >= 0.05:
            candidates.append({"method": row["method"], **row})
    for row in audit.get("conformal", []):
        if row.get("coverage", 0) >= 0.05:
            candidates.append({"method": "split-conformal", **row})
    for row in audit.get("relik", []):
        if row.get("coverage", 0) >= 0.05:
            candidates.append({"method": "ReliK", **row})
    if "calibrated_per_relation" in audit and "_pooled_all_" in audit["calibrated_per_relation"]:
        pr = audit["calibrated_per_relation"]["_pooled_all_"]
        candidates.append({"method": "per-relation calibrated", "coverage": pr["coverage"],
                           "realized_fdr": pr["realized_fdr"], "precision": pr["precision"],
                           "recall": pr["recall"]})
    # dedupe: keep lowest realized FDR per method among covered rows
    best = {}
    for c in candidates:
        if c["method"] not in best or c["realized_fdr"] < best[c["method"]]["realized_fdr"]:
            best[c["method"]] = c
    return [best[k] for k in sorted(best)]


def decision_cost_table(audit: dict) -> list[dict]:
    rows = []
    for method, cost_rows in audit.get("decision_cost", {}).items():
        for cr in cost_rows:
            rows.append({"method": method, "c_fp": cr["c_fp"], "c_fn": cr["c_fn"],
                         "expected_cost": cr["expected_cost"]})
    for cr in audit.get("cost_aware", []):
        rows.append({"method": "cost-aware", "c_fp": cr["c_fp"], "c_fn": cr["c_fn"],
                     "expected_cost": cr["expected_cost"]})
    return rows


def render_tables(audit: dict, out_prefix: str = "hetionet"):
    tables = {
        "claimed_vs_realized": claimed_vs_realized_table(audit),
        "method_comparison": method_comparison(audit),
        "decision_cost": decision_cost_table(audit),
    }
    md = [f"# {out_prefix} screening audit summary", ""]
    md.append("## Claimed vs realized FDR")
    md.append("| method | claimed | realized FDR | precision | recall | coverage |")
    md.append("|---|---|---|---|---|---|")
    for r in tables["claimed_vs_realized"]:
        md.append(f"| {r['method']} | {r['claimed']} | {r['realized_fdr']:.4f} | "
                  f"{r['precision']:.4f} | {r['recall']:.4f} | {r['coverage']:.1%} |")
    md.append("")
    md.append("## Method comparison (coverage >= 5%)")
    md.append("| method | realized FDR | precision | recall | coverage |")
    md.append("|---|---|---|---|---|")
    for r in tables["method_comparison"]:
        md.append(f"| {r['method']} | {r['realized_fdr']:.4f} | {r['precision']:.4f} | "
                  f"{r['recall']:.4f} | {r['coverage']:.1%} |")
    md.append("")
    md.append("## Decision cost (expected cost per candidate)")
    md.append("| method | c_fp | c_fn | expected cost |")
    md.append("|---|---|---|---|")
    for r in tables["decision_cost"]:
        md.append(f"| {r['method']} | {r['c_fp']} | {r['c_fn']} | {r['expected_cost']:.4f} |")
    out = ANALYSIS / f"{out_prefix}_summary.md"
    out.write_text("\n".join(md))
    print(f"Saved {out}")
    return tables


def render_figures(audit: dict, out_prefix: str = "hetionet"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"nominal BH": "#c0392b", "calibrated": "#2c7fb8", "raw top-k": "#7f7f7f",
               "cost-aware": "#2ca02c", "split-conformal": "#9467bd",
               "ReliK": "#d62728", "per-relation calibrated": "#17becf"}
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})

    # Fig 1: claimed vs realized FDR (calibrated + nominal)
    rows = claimed_vs_realized_table(audit)
    cal_rows = [r for r in rows if r["method"] == "calibrated" and r["claimed"] and r["coverage"] > 0]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(cal_rows))
    claimed = [r["claimed"] for r in cal_rows]
    realized = [r["realized_fdr"] for r in cal_rows]
    w = 0.35
    ax.bar(x - w / 2, claimed, w, label="claimed (alpha)", color="#2c7fb8", alpha=0.85)
    ax.bar(x + w / 2, realized, w, label="realized", color="#c0392b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"α={r['claimed']:.2f}" for r in cal_rows], rotation=0)
    ax.set_ylabel("FDR")
    ax.set_title("Claimed vs realized FDR (screening layer)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p1 = ANALYSIS / f"{out_prefix}_claimed_vs_realized.pdf"
    fig.savefig(p1); print(f"Saved {p1}")

    # Fig 2: realized FDR vs recall by method
    cmp_rows = method_comparison(audit)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for r in cmp_rows:
        col = palette.get(r["method"], "#333333")
        ax.scatter(r["recall"], r["realized_fdr"], label=r["method"], color=col, s=45, zorder=3)
    ax.axhline(0.10, color="#2c7fb8", ls="--", lw=1, label="claim α=0.10")
    ax.set_xlabel("recall (true links kept)")
    ax.set_ylabel("realized FDR")
    ax.set_title("Screening methods: error vs recall")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p2 = ANALYSIS / f"{out_prefix}_fdr_vs_recall.pdf"
    fig.savefig(p2); print(f"Saved {p2}")

    # Fig 3: decision cost vs cost ratio
    cost_rows = decision_cost_table(audit)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for method in dict.fromkeys(c["method"] for c in cost_rows):
        xs = [(r["c_fp"], r["c_fn"]) for r in cost_rows if r["method"] == method]
        ys = [r["expected_cost"] for r in cost_rows if r["method"] == method]
        ratios = [c[0] / c[1] for c in xs]
        ax.plot(ratios, ys, marker="o", label=method, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("cost ratio c_fp / c_fn (log)")
    ax.set_ylabel("expected decision cost")
    ax.set_title("Decision cost vs cost asymmetry")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p3 = ANALYSIS / f"{out_prefix}_decision_cost.pdf"
    fig.savefig(p3); print(f"Saved {p3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", nargs="*", default=None)
    args = ap.parse_args()
    if args.audit:
        audits = [load(Path(p)) for p in args.audit]
    else:
        files = sorted(glob.glob(str(ROOT / "results" / "hetionet_audit_*.json")))
        audits = [load(Path(f)) for f in files] if files else []
    if not audits:
        print("No audit JSONs found; run code/experiment_hetionet.py after training.")
        return
    for a in audits:
        prefix = "hetionet"
        render_tables(a, prefix)
        render_figures(a, prefix)


if __name__ == "__main__":
    main()
