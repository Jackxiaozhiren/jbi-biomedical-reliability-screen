"""Compute Wilson 95% score intervals for realized-FDR / precision operating points.

Data source: the frozen audit JSONs only (kept counts and FP counts), so the
intervals are a pure re-derivation, not new data. Used by the paper to quantify
the uncertainty of its central realized-vs-claimed estimates (single split +
single sampling seed; binomial sampling variance only).

Usage: python code/compute_wilson_ci.py
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Z = 1.959963984540054  # 95%


def wilson_score_ci(n: int, x: int) -> tuple[float, float]:
    """Wilson score interval for a proportion x/n (x successes out of n)."""
    if n <= 0:
        return (0.0, 0.0)
    p = x / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def row(label: str, n_kept: int, n_neg: int, n_pos: int) -> None:
    fdr = n_neg / n_kept if n_kept else 0.0
    lo, hi = wilson_score_ci(n_kept, n_neg)
    print(f"{label:<38} n={n_kept:<6} FDR={fdr:.4f}  Wilson95=({lo:.4f},{hi:.4f})  "
          f"TP={n_pos:<6} prec={n_pos/n_kept if n_kept else 0:.4f}")


print("=" * 100)
print("HETIONET (seed-42 frozen), global operating points")
print("=" * 100)
h = json.load(open(ROOT / "results" / "hetionet_audit_J9_K500.json"))
for r in h["calibrated"]:
    if r["n_screened_in"] > 0:
        row(f"calibrated target {r['claimed_fdr']}", r["n_screened_in"],
            r["n_kept_negative"], r["n_kept_positive"])
for r in h["cost_aware"]:
    n_neg = round(r["n_screened_in"] * r["realized_fdr"])
    row(f"cost-aware {r['c_fp']}:{r['c_fn']}", r["n_screened_in"],
        n_neg, r["n_screened_in"] - n_neg)
print()

print("=" * 100)
print("HETIONET per-relation calibrated 0.10 (frozen)")
print("=" * 100)
for rel, v in h["calibrated_per_relation"].items():
    if rel == "_pooled_all_":
        continue
    row(rel, v["n_screened_in"], v["n_kept_negative"], v["n_kept_positive"])
print()

print("=" * 100)
print("WN18RR RotatE (frozen)")
print("=" * 100)
w = json.load(open(ROOT / "results" / "realized_fdr_WN18RR_RotatE_J9_K500.json"))
nb = w["nominal_bh_0.05"]
row("WN18RR nominal BH 0.05", nb["n_screened"], 0, 0)
# recompute FP from realized_fdr
n_neg = round(nb["realized_fdr"] * nb["n_screened"])
row("WN18RR nominal BH 0.05 (FP-derived)", nb["n_screened"], n_neg, nb["n_screened"] - n_neg)
for r in w["calibrated"]:
    if r["coverage"] > 0:
        n_neg_r = round(r["realized_fdr"] * (r["coverage"] * w["pool_size"] / 2))
        n_kept_r = round(r["coverage"] * w["pool_size"] / 2)
        row(f"WN18RR calibrated {r['claimed_fdr']}", n_kept_r, n_neg_r, n_kept_r - n_neg_r)
print()

print("=" * 100)
print("FB15k-237 RotatE (frozen)")
print("=" * 100)
f = json.load(open(ROOT / "results" / "realized_fdr_FB15k237_J9_K500.json"))
nb2 = f["nominal_bh_0.05"]
n_neg2 = round(nb2["realized_fdr"] * nb2["n_screened"])
row("FB15k nominal BH 0.05 (FP-derived)", nb2["n_screened"], n_neg2, nb2["n_screened"] - n_neg2)
for r in f["calibrated"]:
    if r["coverage"] > 0:
        n_kept_r = round(r["coverage"] * f["pool_size"] / 2)
        n_neg_r = round(r["realized_fdr"] * n_kept_r)
        row(f"FB15k calibrated {r['claimed_fdr']}", n_kept_r, n_neg_r, n_kept_r - n_neg_r)
print()

print("=" * 100)
print("WN18RR TransE weak model (frozen, new)")
print("=" * 100)
t = json.load(open(ROOT / "results" / "realized_fdr_WN18RR_TransE_J9_K500.json"))
nb3 = t["nominal_bh_0.05"]
n_neg3 = round(nb3["realized_fdr"] * nb3["n_screened"])
row("WN18RR-TransE nominal BH 0.05", nb3["n_screened"], n_neg3, nb3["n_screened"] - n_neg3)
for r in t["calibrated"]:
    if r["coverage"] > 0:
        n_kept_r = round(r["coverage"] * t["pool_size"] / 2)
        n_neg_r = round(r["realized_fdr"] * n_kept_r)
        row(f"WN18RR-TransE calibrated {r['claimed_fdr']}", n_kept_r, n_neg_r, n_kept_r - n_neg_r)
