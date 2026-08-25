"""Regenerate the four paper figures from the authoritative frozen JSONs.

- fig1/2/3 (Hetionet): analyze_results.render_figures on hetionet_audit_J9_K500.json
- fig4 (K-resolution): summarize_all.k_resolution_table + render_k_resolution on the
  re-run spike_k{100,500,1000}_transE.json (converged TransE)

Writes master_*.pdf into analysis/ and copies them into paper/figures/ with the
names used by 04_experiments.tex. Data provenance is the frozen exports only.

Usage: python code/regenerate_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_results as ar          # noqa: E402
from summarize_all import k_resolution_table, render_k_resolution  # noqa: E402

MAPPING = {
    "master_claimed_vs_realized.pdf": "fig1_claimed_vs_realized.pdf",
    "master_fdr_vs_recall.pdf": "fig2_fdr_vs_recall.pdf",
    "master_decision_cost.pdf": "fig3_decision_cost.pdf",
    "master_k_resolution.pdf": "fig4_k_resolution.pdf",
}

audit = json.load(open(ROOT / "results" / "hetionet_audit_J9_K500.json"))
ar.render_figures(audit, "master")
rows = k_resolution_table()
render_k_resolution(rows, "master")
print("K-resolution rows:", [(r["K"], round(r["floor_fraction"], 3), round(r["pi0"], 3)) for r in rows])

fig_dir = ROOT / "paper" / "figures"
for src_name, dst_name in MAPPING.items():
    src = ROOT / "analysis" / src_name
    dst = fig_dir / dst_name
    shutil.copy(src, dst)
    print(f"copied {src_name} -> {dst_name}")
